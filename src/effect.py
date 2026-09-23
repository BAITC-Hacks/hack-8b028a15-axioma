"""Historical inventory-policy simulation, not an independent test or a pilot.

Forecasts are frozen at each review using only then-completed months.
Monthly observed sales are spread uniformly over days: daily timing is synthetic.
"""
from copy import deepcopy
from dataclasses import dataclass, asdict, replace
from io import BytesIO
import json
import zipfile
import numpy as np
import pandas as pd
from .data import number
from .engine import Settings, calculate
from .planning import inventory_plan, receipt_schedule

POLICIES={'mean':'Среднее за 6 месяцев','adaptive':'Axioma: автовыбор','pooled':'Axioma: общая ML-модель'}
MODES={'lost':'Потерянная продажа','backorder':'Отложенный заказ'}

@dataclass(frozen=True)
class Experiment:
    start: str = '2026-06-01'
    end: str = '2026-08-31'
    lead_days: int = 21
    review_days: int = 14
    safety_days: int = 7
    initial_days: float = 21.
    max_skus: int = 30

    def validate(self):
        start=pd.Timestamp(self.start);end=pd.Timestamp(self.end)
        if start.day!=1 or end!=end+pd.offsets.MonthEnd(0) or end<start or (end-start).days>366:
            raise ValueError('Период должен содержать полные месяцы, не более года')
        for name,lo,hi in [('lead_days',1,365),('review_days',1,90),('safety_days',0,90),('max_skus',1,5000)]:
            value=getattr(self,name)
            if not isinstance(value,(int,np.integer)) or not lo<=value<=hi:raise ValueError(f'Некорректный параметр {name}')
        if not np.isfinite(self.initial_days) or not 0<=self.initial_days<=365:raise ValueError('Некорректный начальный запас')


def training_data(dataset,skus,review_date):
    """Drop retrospective inputs whose historical availability cannot be proven."""
    train=deepcopy(dataset)
    cutoff=pd.Timestamp(review_date).replace(day=1).normalize()
    train.products=train.products[train.products.sku.isin(skus)].copy()
    train.products=train.products.drop(columns=['source_growth','source_growth_percent'],errors='ignore')
    train.products['stock']=0. # Forecast-only; actual simulated balance is separate.
    for attr in ['sales','transactions']:
        frame=getattr(train,attr)
        setattr(train,attr,frame[frame.sku.isin(skus)&frame.date.lt(cutoff)].copy())
    train.seasonal={}
    for attr in ['stocks','stockouts','transit']:setattr(train,attr,getattr(train,attr).iloc[:0].copy())
    return train


def forecast_bank(dataset,skus,experiment):
    """No fitted model ever receives evaluation-month transactions/sales."""
    bank={};forecast_rows=[]
    for day in pd.date_range(experiment.start,experiment.end,freq=f'{experiment.review_days}D'):
        train=training_data(dataset,skus,day)
        base=Settings(as_of=str(day.date()),lead_days=experiment.lead_days,review_days=experiment.review_days,
                      safety_days=experiment.safety_days,compensate_stockout=False,approximate_stockout=False,use_source_growth=False)
        for policy in POLICIES:
            cfg=replace(base,forecast_method='legacy' if policy=='mean' else policy)
            if policy=='mean':cfg=replace(cfg,remove_outliers=False,seasonality=False,trend=False)
            _,histories,_=calculate(train,cfg)
            for sku,h in histories.items():
                curve=pd.DataFrame(h.attrs['future'])
                rates=curve['Прогноз в день'].to_numpy(dtype=float)
                bank[(policy,day,sku)]=rates
                forecast_rows.append(dict(policy=policy,review_date=str(day.date()),sku=sku,
                    training_before=str(day.replace(day=1).date()),method=h.attrs['model_info']['method'],
                    forecast_first_review_days=float(rates[:experiment.review_days].sum())))
    return bank,pd.DataFrame(forecast_rows)


def simulate_path(sku,dates,demand,initial,pack,moq,experiment,policy,mode,bank,opening=None):
    """Beginning-day receipt -> review/order -> serve backlog -> today's demand."""
    if mode not in MODES:raise ValueError('Unknown shortage mode')
    demand=np.asarray(demand,dtype=float)
    if len(dates)!=len(demand) or not np.isfinite(demand).all() or (demand<0).any():raise ValueError('Некорректный спрос')
    if not np.isfinite(initial) or initial<0:raise ValueError('Некорректный начальный запас')
    on_hand=float(initial);backlog=0.;events=[];orders=[];pending=[]
    if opening is not None and not opening.empty:
        if not {'sku','eta','quantity','known_on'}.issubset(opening):raise ValueError('Для начального пути нужны sku, eta, quantity, known_on')
        for row in opening[opening.sku.eq(sku)].itertuples():
            eta=pd.Timestamp(row.eta).normalize();known=pd.Timestamp(row.known_on).normalize()
            if pd.isna(eta) or pd.isna(known) or known>dates[0] or eta<dates[0] or not np.isfinite(row.quantity) or row.quantity<0:
                raise ValueError('Начальный путь должен быть известен до начала эксперимента и иметь будущую дату')
            pending.append(dict(eta=eta,quantity=float(row.quantity),source='opening'))
    initial_incoming=sum(x['quantity'] for x in pending)
    for i,day in enumerate(dates):
        arriving=[r for r in pending if r['eta']==day]
        received=sum(r['quantity'] for r in arriving)
        opening_received=sum(r['quantity'] for r in arriving if r['source']=='opening')
        pending=[r for r in pending if r['eta']!=day]
        begin=on_hand;backlog_begin=backlog;on_hand+=received
        quantity=0.
        if i%experiment.review_days==0:
            rates=bank[(policy,day,sku)]
            future=pd.date_range(day,periods=len(rates))
            incoming=pd.DataFrame(pending,columns=['eta','quantity','source'])
            # Backlog is a claim on stock, not a lost sale that should be reordered.
            plan=inventory_plan(rates,receipt_schedule(incoming,future),on_hand-backlog,
                                experiment.lead_days,float(np.mean(rates))*experiment.safety_days,pack,moq)
            quantity=plan['recommended']
            if quantity>0:
                eta=day+pd.Timedelta(days=experiment.lead_days)
                pending.append(dict(eta=eta,quantity=quantity,source='generated'))
                orders.append(dict(sku=sku,policy=policy,mode=mode,ordered_on=str(day.date()),eta=str(eta.date()),quantity=quantity,
                                   stock_at_decision=on_hand,backlog_at_decision=backlog,pack=pack,moq=moq,
                                   arrives_after_end=eta>dates[-1],source='policy_generated'))
        served_backlog=min(on_hand,backlog);on_hand-=served_backlog;backlog-=served_backlog
        requested=float(demand[i]);served_now=min(on_hand,requested);on_hand-=served_now
        unfilled=max(0.,requested-served_now)
        lost=unfilled if mode=='lost' else 0.
        if mode=='backorder':backlog+=unfilled
        events.append(dict(sku=sku,policy=policy,mode=mode,date=str(day.date()),demand=requested,stock_start=begin,
                           received=received,opening_received=opening_received,generated_received=received-opening_received,
                           ordered=quantity,served_now=served_now,served_backlog=served_backlog,lost=lost,unfilled_today=unfilled,
                           backlog_start=backlog_begin,backlog_end=backlog,stock_end=on_hand))
        assert abs(begin+received-served_now-served_backlog-on_hand)<1e-7
        assert abs(backlog_begin+requested-served_now-lost-served_backlog-backlog)<1e-7
        assert on_hand>=-1e-8 and backlog>=-1e-8
    daily=pd.DataFrame(events)
    total=float(daily.demand.sum());served=float(daily.served_now.sum()+daily.served_backlog.sum())
    row=dict(sku=sku,policy=policy,mode=mode,days=len(dates),initial_stock=float(initial),initial_incoming=initial_incoming,
             demand=total,shortage_days=int(daily.unfilled_today.gt(1e-8).sum()),backlog_days=int(daily.backlog_end.gt(1e-8).sum()),
             immediate_fill_rate=float(daily.served_now.sum()/total) if total>0 else np.nan,
             eventual_fill_rate=served/total if total>0 else np.nan,
             average_stock=float(daily.stock_end.mean()),stock_unit_days=float(daily.stock_end.sum()),
             backlog_unit_days=float(daily.backlog_end.sum()),lost_units=float(daily.lost.sum()),
             orders=len(orders),ordered_units=float(daily.ordered.sum()),end_stock=on_hand,end_backlog=backlog,
             end_pipeline=sum(r['quantity'] for r in pending),
             end_generated_pipeline=sum(r['quantity'] for r in pending if r['source']=='generated'),
             generated_received=float(daily.generated_received.sum()),opening_received=float(daily.opening_received.sum()))
    return row,daily,orders


def compare_inventory(dataset,experiment=Experiment(),opening_transit=None):
    experiment.validate();start=pd.Timestamp(experiment.start);end=pd.Timestamp(experiment.end)
    monthly=dataset.sales.groupby(['sku','date'],as_index=False).quantity.sum(min_count=1)
    monthly=monthly[np.isfinite(monthly.quantity)&monthly.quantity.ge(0)]
    history=monthly[monthly.date.lt(start)]
    eligible=sorted(set(history.groupby('sku').date.nunique().loc[lambda s:s>=6].index)&set(dataset.products.sku))
    selected=eligible[:experiment.max_skus] # Chosen BEFORE seeing target demand.
    months=pd.date_range(start,end,freq='MS');dates=pd.date_range(start,end)
    evaluation=monthly[monthly.sku.isin(selected)&monthly.date.isin(months)]
    coverage=evaluation.groupby('sku').date.nunique()
    evaluated=[s for s in selected if coverage.get(s,0)==len(months)]
    skipped=[dict(sku=s,reason='Нет всех пригодных месячных наблюдений периода; пропуск, NaN и отрицательное значение не равны нулю') for s in selected if s not in evaluated]
    meta=dict(kind='Ретроспективная симуляция',settings=asdict(experiment),supplier=dataset.supplier,
              eligible_skus=len(eligible),selected_skus=len(selected),evaluated_skus=len(evaluated),excluded_skus=len(skipped),
              selection='Первые коды в строковой сортировке с >=6 месяцами ДО старта; пропуски целевых месяцев раскрыты отдельно',
              invalid_months='Отсутствующие, неконечные и отрицательные месячные итоги не участвуют в покрытии; пропуски внутри обучающей истории модель заполняет нулями как явное допущение',
              demand='Месячные наблюдаемые продажи равномерно распределены по дням; это не истинный спрос и не фактические дни дефицита',
              initial_stock='Средний дневной спрос последних 6 завершённых месяцев до старта × initial_days; одинаково для всех политик',
              opening_transit='Только явно переданный путь с known_on <= start; текущий путь из файлов не используется',
              product_rules='Текущие MOQ/кратность/единицы заморожены одинаково для всех политик как допущение; исторические изменения неизвестны',
              training_population='Общая модель обучается на выбранных до просмотра целевого периода SKU; выборка не представляет весь каталог',
              model_status='Модели не изменялись и не выбирались по этому периоду; июнь–август уже просматривались, независимая проверка не заявляется',
              zero_demand_fill='Процент обслуживания не определён при нулевом спросе; такие товары отдельно учтены в знаменателях',
              sources=dataset.sources)
    if not evaluated:return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(skipped,columns=['sku','reason']),meta,pd.DataFrame()
    # Keep the selected training population even if target observation is missing.
    bank,forecast_info=forecast_bank(dataset,selected,experiment)
    products=dataset.products.set_index('sku');results=[];traces=[];orders=[]
    for sku in evaluated:
        p=products.loc[sku]
        old=history[history.sku.eq(sku)].set_index('date').quantity
        completed=pd.date_range(end=start-pd.offsets.MonthBegin(1),periods=6,freq='MS')
        past=old.reindex(completed,fill_value=0).clip(lower=0)
        initial=float((past/past.index.days_in_month).mean())*experiment.initial_days
        target=evaluation[evaluation.sku.eq(sku)].set_index('date').quantity.clip(lower=0)
        rates={month:float(target[month])/month.days_in_month for month in months}
        demand=[rates[day.replace(day=1)] for day in dates]
        for policy in POLICIES:
            for mode in MODES:
                row,trace,events=simulate_path(sku,dates,demand,initial,max(number(p.pack,1),1),max(number(p.moq,1),1),
                                              experiment,policy,mode,bank,opening_transit)
                row.update(name=p['name'],unit=p.unit,history_months=len(old),history_zero_fraction=float(past.eq(0).mean()),pack=max(number(p.pack,1),1),moq=max(number(p.moq,1),1))
                trace['unit']=p.unit
                for event in events:event['unit']=p.unit
                results.append(row);traces.append(trace);orders.extend(events)
    results=pd.DataFrame(results)
    baseline=results[results.policy.eq('mean')][['sku','mode','immediate_fill_rate','average_stock','shortage_days']]
    baseline=baseline.rename(columns={c:'baseline_'+c for c in ['immediate_fill_rate','average_stock','shortage_days']})
    results=results.merge(baseline,on=['sku','mode'],validate='many_to_one')
    results['fill_change_pp']=(results.immediate_fill_rate-results.baseline_immediate_fill_rate)*100
    results['stock_change']=results.average_stock-results.baseline_average_stock
    results['worse_service']=results.immediate_fill_rate.lt(results.baseline_immediate_fill_rate-1e-8)
    results['more_stock']=results.average_stock.gt(results.baseline_average_stock+1e-8)
    results['tradeoff']=results.fill_change_pp.gt(1e-6)&results.more_stock
    meta['positive_demand_skus']=int(results[results.policy.eq('mean')&results['mode'].eq('lost')].demand.gt(0).sum())
    return results,pd.concat(traces,ignore_index=True),pd.DataFrame(orders,columns=['sku','policy','mode','ordered_on','eta','quantity','stock_at_decision','backlog_at_decision','pack','moq','arrives_after_end','source','unit']),pd.DataFrame(skipped,columns=['sku','reason']),meta,forecast_info


def aggregate_metrics(results):
    rows=[]
    if results.empty:return pd.DataFrame()
    for (policy,mode),g in results.groupby(['policy','mode']):
        scored=g[g.demand.gt(0)]
        rows.append(dict(policy=policy,mode=mode,evaluated_skus=len(g),positive_demand_skus=len(scored),zero_demand_skus=len(g)-len(scored),
                         macro_immediate_fill_pct=float(scored.immediate_fill_rate.mean()*100) if len(scored) else np.nan,
                         macro_eventual_fill_pct=float(scored.eventual_fill_rate.mean()*100) if len(scored) else np.nan,
                         mean_shortage_days=float(g.shortage_days.mean()),worse_service_skus=int(g.worse_service.sum()),
                         better_service_skus=int(g.fill_change_pp.gt(1e-6).sum()),less_stock_skus=int(g.stock_change.lt(-1e-8).sum()),
                         lower_stock_worse_service_skus=int((g.worse_service&g.stock_change.lt(-1e-8)).sum()),
                         more_stock_skus=int(g.more_stock.sum()),service_stock_tradeoff_skus=int(g.tradeoff.sum())))
    return pd.DataFrame(rows)


def forecast_diagnostics(forecasts,daily,review_days):
    """Non-overlapping first-review forecast windows: per-SKU WAPE and signed bias."""
    if forecasts.empty or daily.empty:return pd.DataFrame()
    observed=daily[daily.policy.eq('mean')&daily['mode'].eq('lost')].copy()
    observed['date']=pd.to_datetime(observed.date)
    last=observed.date.max();rows=[]
    for f in forecasts.itertuples():
        start=pd.Timestamp(f.review_date);end=start+pd.Timedelta(days=review_days-1)
        if end>last:continue # Do not score partial final windows against a full forecast.
        actual=observed[observed.sku.eq(f.sku)&observed.date.between(start,end)]
        if len(actual)!=review_days:continue
        rows.append(dict(sku=f.sku,policy=f.policy,actual=float(actual.demand.sum()),forecast=f.forecast_first_review_days))
    if not rows:return pd.DataFrame()
    scores=[]
    for (sku,policy),g in pd.DataFrame(rows).groupby(['sku','policy']):
        actual=g.actual.sum();error=g.forecast-g.actual
        scores.append(dict(sku=sku,policy=policy,windows=len(g),actual=actual,
                           wape_pct=float(error.abs().sum()/actual*100) if actual>0 else np.nan,
                           signed_bias_pct=float(error.sum()/actual*100) if actual>0 else np.nan,
                           zero_actual_positive_forecast=bool(actual==0 and g.forecast.sum()>1e-8)))
    return pd.DataFrame(scores)


def scenario_costs(rows,unit_price,holding_per_unit_day,order_line_cost,lost_per_unit,backlog_per_unit_day,source,currency='KZT'):
    values=[unit_price,holding_per_unit_day,order_line_cost,lost_per_unit,backlog_per_unit_day]
    if not source.strip() or not currency.strip() or any(not np.isfinite(v) or v<0 for v in values):
        raise ValueError('Укажите источник, валюту и конечные неотрицательные ставки')
    if rows.sku.nunique()!=1:raise ValueError('Цены задаются для одного SKU в его единицах учёта')
    if set(rows['mode'])-set(rows.loc[rows.policy.eq('mean'),'mode']):raise ValueError('Для сравнения затрат нужна простая политика в каждом режиме')
    costs=[]
    for row in rows.itertuples():
        for holding_factor in [.5,1.,1.5]:
            for shortage_factor in [.5,1.,1.5]:
                storage=row.stock_unit_days*holding_per_unit_day*holding_factor
                ordering=row.orders*order_line_cost
                shortage=(row.lost_units*lost_per_unit if row.mode=='lost' else row.backlog_unit_days*backlog_per_unit_day)*shortage_factor
                costs.append(dict(sku=row.sku,unit=row.unit,policy=row.policy,mode=row.mode,source=source,currency=currency,
                    unit_price=unit_price,holding_per_unit_day=holding_per_unit_day,order_line_cost=order_line_cost,
                    lost_per_unit=lost_per_unit,backlog_per_unit_day=backlog_per_unit_day,
                    holding_factor=holding_factor,shortage_factor=shortage_factor,storage_cost=storage,ordering_cost=ordering,
                    shortage_cost=shortage,operating_cost=storage+ordering+shortage,
                    generated_purchase_commitment=row.ordered_units*unit_price,received_generated_value=row.generated_received*unit_price,
                    residual_stock_value=row.end_stock*unit_price,pending_generated_value=row.end_generated_pipeline*unit_price))
    result=pd.DataFrame(costs)
    base=result[result.policy.eq('mean')][['sku','mode','holding_factor','shortage_factor','operating_cost']].rename(columns={'operating_cost':'baseline_operating_cost'})
    result=result.merge(base,on=['sku','mode','holding_factor','shortage_factor'],validate='many_to_one')
    result['cost_change_vs_mean']=result.operating_cost-result.baseline_operating_cost
    return result


def _safe_csv(frame):
    copy=frame.copy()
    for col in copy.select_dtypes(include=['object','str']):
        copy[col]=copy[col].map(lambda v:"'"+v if isinstance(v,str) and v.startswith(('=','+','-','@')) else v)
    return copy.to_csv(index=False).encode('utf-8-sig')


def protocol_zip(results,daily,orders,excluded,meta,forecasts,costs=None):
    data=BytesIO()
    with zipfile.ZipFile(data,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,frame in [('results',results),('daily_balance',daily),('generated_orders',orders),('excluded',excluded),('forecast_origins',forecasts),('aggregates',aggregate_metrics(results)),('forecast_diagnostics',forecast_diagnostics(forecasts,daily,meta['settings']['review_days']))]:
            z.writestr(name+'.csv',_safe_csv(frame))
        z.writestr('metadata.json',json.dumps(meta,ensure_ascii=False,indent=2,default=str,allow_nan=False))
        if costs is not None:z.writestr('scenario_costs.csv',_safe_csv(costs))
    return data.getvalue()


def pilot_zip(products,input_batch,parameters,sources):
    """Blank, counterbalanced measurement assignments. Not fabricated observations."""
    data=BytesIO()
    assignments=[]
    for pair in [1,2]:
        for position,workflow in enumerate(['Excel','Axioma'] if pair==1 else ['Axioma','Excel'],1):
            assignments.append(dict(pair=pair,position=position,workflow=workflow,dataset_version=input_batch,
                matched_task='',sku_subset='',operator='',started_at='',finished_at='',active_minutes='',waiting_minutes='',
                reviewed_lines='',corrected_lines='',errors='',change_reasons='',verified_by='',verified_at='',status='Не выполнено'))
    manifest=dict(status='Пилот не проведён; задания для заполнения',dataset_version=input_batch,parameters=parameters,sources=sources,
                  instruction='Заполнить разные сопоставимые задания и SKU до начала. Одинаковые данные и правила в паре; порядок Excel/Axioma чередуется. Пустые поля не являются результатом пилота.')
    with zipfile.ZipFile(data,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('README.txt','Подготовка пилота, не результат внедрения.\nДо старта назначить проверяющего и сопоставимые задания; зафиксировать файлы и параметры.\nВ каждой паре одни входы и правила. Между парами другие сопоставимые SKU; порядок Excel/Axioma чередуется.\nФиксировать активное время отдельно от ожидания, правки и причины, ошибки. Проверяющий подтверждает выгрузку.\nУспех: отсутствие критических ошибок и меньшее активное время при не худшем качестве; пороги согласовать ДО начала.\nПодробный протокол: docs/pilot.md.\n')
        z.writestr('assignments.csv',_safe_csv(pd.DataFrame(assignments)))
        z.writestr('frozen_product_rules.csv',_safe_csv(products))
        z.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2,default=str))
    return data.getvalue()
