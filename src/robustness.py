"""Finite scenario grid and a stock-count queue; no probabilities or retraining."""
from dataclasses import dataclass
from itertools import product
import numpy as np
import pandas as pd
from .planning import inventory_plan, receipt_schedule
from .data import number

ALL_ORDER='Пополнение во всех проверенных сценариях'
DEPENDS='Решение зависит от допущений'
NO_ORDER='Пополнение не требуется в проверенных сценариях'
INSUFFICIENT='Недостаточно данных для сценариев'


@dataclass(frozen=True)
class ScenarioSettings:
    stock_percent: float = 20.
    demand_percent: float = 20.
    delay_days: int = 7

    def validate(self):
        if any(not np.isfinite(x) or not 0<=x<=100 for x in (self.stock_percent,self.demand_percent)):
            raise ValueError('Отклонения остатка и спроса должны быть от 0 до 100%')
        if not isinstance(self.delay_days,(int,np.integer)) or not 0<=self.delay_days<=90:
            raise ValueError('Задержка должна быть целым числом от 0 до 90 дней')


def stock_evidence(dataset, as_of):
    """Latest nonfuture snapshot and signed sales since it. Missing rows != zero."""
    asof=pd.Timestamp(as_of).normalize()
    snapshots=dataset.stocks.dropna(subset=['date','quantity']).copy()
    snapshots=snapshots[snapshots.date.le(asof)&snapshots.quantity.ge(0)]
    snapshots=snapshots.sort_values('date').drop_duplicates('sku',keep='last')
    if snapshots.empty:return {}
    evidence=snapshots.set_index('sku').to_dict('index')
    if not dataset.transactions.empty:
        tx=dataset.transactions.merge(snapshots[['sku','date']].rename(columns={'date':'snapshot_date'}),on='sku',how='inner')
        tx=tx[tx.date.ge(tx.snapshot_date)&tx.date.lt(asof)]
        amounts=tx.groupby('sku').quantity.agg(['sum','count'])
        for sku,row in amounts.iterrows():
            if row['count']>0:evidence[sku]['recorded_sales']=float(row['sum'])
    return evidence


def stock_variants(row, evidence, as_of, percent):
    current=float(row.stock) if pd.notna(row.stock) else np.nan
    age=np.nan;snapshot_date='';anchors=[]
    if np.isfinite(current) and current>=0:
        anchors=[(current,str(row.get('stock_origin','Остаток из входных данных'))) ]
        # Timestamp of an input "current" balance is not necessarily known.
        if row.get('stock_origin')=='Ручной пересчёт':age=0.
    elif evidence:
        snapshot_date=str(pd.Timestamp(evidence['date']).date())
        age=(pd.Timestamp(as_of).normalize()-pd.Timestamp(evidence['date']).normalize()).days
        snapshot=float(evidence['quantity'])
        anchors=[(snapshot,f'Снимок {snapshot_date} без изменений')]
        if 'recorded_sales' in evidence:
            anchors.append((max(0.,snapshot-evidence['recorded_sales']),f'Снимок {snapshot_date} минус продажи {evidence["recorded_sales"]:g}; без поступлений/резервов'))
    values={}
    for amount,basis in anchors:
        for factor in sorted({1-percent/100,1.,1+percent/100}):
            value=round(max(0.,amount*factor),8)
            label=f'{basis}; остаток ×{factor:g}'
            values.setdefault(value,[]).append(label)
    return [(v,' / '.join(labels)) for v,labels in sorted(values.items())],age,snapshot_date


def _changes(frame, dimension, value):
    others=[c for c in ['scenario_stock','demand_multiplier','delay_days'] if c!=dimension]
    return bool(frame.groupby(others,dropna=False)[value].nunique().gt(1).any())


def analyse_scenarios(dataset, cfg, result, histories, options=ScenarioSettings()):
    """Reuse the chosen forecast curve; vary stock, rate and dated receipts only.

    dataset contains original/manager-confirmed stocks, not a filled scenario.
    result can contain the explicitly selected stock scenario used for approval.
    """
    options.validate()
    asof=pd.Timestamp(cfg.as_of).normalize()
    evidence=stock_evidence(dataset,asof)
    transit={k:g for k,g in dataset.transit.groupby('sku')} if not dataset.transit.empty else {}
    products=dataset.products.set_index('sku')
    result_by_sku=result.set_index('sku')
    summaries=[];details=[];delays=[]
    factors=sorted({1-options.demand_percent/100,1.,1+options.demand_percent/100})
    delay_values=sorted({0,options.delay_days})
    assumptions=f'Остаток ±{options.stock_percent:g}%; спрос ±{options.demand_percent:g}%; ожидаемые поступления +0/+{options.delay_days} дн.; срок нового заказа {cfg.lead_days} дн. неизменен. Только проверенная сетка, не доверительный интервал.'
    for sku,p in products.iterrows():
        variants,age,snapshot_date=stock_variants(p,evidence.get(sku),asof,options.stock_percent)
        hist=histories.get(sku)
        future=pd.DataFrame(hist.attrs.get('future',[])) if hist is not None else pd.DataFrame()
        common=dict(sku=sku,scenario_snapshot_date=snapshot_date,scenario_snapshot_age_days=age,
                    scenario_assumptions=assumptions,scenario_basis='; '.join(label for _,label in variants),
                    stock_min=min((v for v,_ in variants),default=np.nan),stock_max=max((v for v,_ in variants),default=np.nan),
                    scenario_count=0,order_min=np.nan,order_max=np.nan,urgency_any=False,urgency_all=False,
                    stock_changes_order=False,stock_changes_urgency=False,relative_order_span=0.,check_stock=False,count_priority=3)
        if future.empty or not variants:
            why='Нет завершённой истории продаж' if future.empty else 'Нет текущего остатка и пригодного снимка'
            common.update(robustness=INSUFFICIENT,sensitivity_reason=why,check_stock=not np.isfinite(p.stock),
                          count_priority=2,count_reason=why+'; нужен пересчёт остатка' if not variants else why+'; запросите историю продаж')
            summaries.append(common)
            continue
        dates=pd.DatetimeIndex(pd.to_datetime(future['Дата']))
        rates=future['Прогноз в день'].to_numpy(dtype=float)
        incoming=transit.get(sku,dataset.transit.iloc[:0])
        schedules={d:receipt_schedule(incoming,dates,d) for d in set(delay_values+[7])}
        cat=float(cfg.category_factors.get(str(p.category),1.))
        pack=max(number(p.pack,1),1.);moq=max(number(p.moq,1),1.)
        records=[]
        for (stock,basis),factor,delay in product(variants,factors,delay_values):
            scenario_rates=rates*factor
            safety=float(scenario_rates.mean())*cfg.safety_days*cat
            plan=inventory_plan(scenario_rates,schedules[delay],stock,cfg.lead_days,safety,pack,moq)
            first=plan['first_shortage_index']
            records.append(dict(sku=sku,scenario_stock=stock,stock_assumption=basis,demand_multiplier=factor,delay_days=delay,
                                scenario_forecast=float(scenario_rates.sum()),scenario_safety=safety,
                                scenario_in_transit=float(schedules[delay].sum()),scenario_order=plan['recommended'],
                                scenario_expedite=plan['expedite_need'],scenario_urgent=plan['urgent'],
                                scenario_shortage=str(dates[first].date()) if first is not None else '',needs_order=plan['recommended']>0))
        frame=pd.DataFrame(records)
        low=float(frame.scenario_order.min());high=float(frame.scenario_order.max())
        status=ALL_ORDER if low>0 else NO_ORDER if high==0 else DEPENDS
        stock_switch=_changes(frame,'scenario_stock','needs_order')
        urgency_switch=_changes(frame,'scenario_stock','scenario_urgent')
        drivers=[label for dim,label in [('scenario_stock','остаток'),('demand_multiplier','спрос'),('delay_days','дата поступления')]
                 if _changes(frame,dim,'scenario_order') or _changes(frame,dim,'scenario_urgent')]
        reason='Изменяются '+', '.join(drivers) if drivers else 'Количество и срочность одинаковы во всей проверенной сетке'
        suffix=f'; снимку {age:g} дн.' if np.isfinite(age) and age>0 else '; дата текущего остатка не указана' if not np.isfinite(age) else '; остаток уточнён сегодня'
        if stock_switch or urgency_switch:
            priority=1;count_reason=('Уточнение остатка меняет заказ да/нет' if stock_switch else 'Уточнение остатка меняет срочность')+suffix
        elif not np.isfinite(p.stock):priority=2;count_reason='Текущий остаток неизвестен; проверка нужна перед утверждением'+suffix
        else:priority=3;count_reason='Решение да/нет и срочность не меняются от проверенных остатков'+suffix
        common.update(robustness=status,scenario_count=len(frame),order_min=low,order_max=high,
                      urgency_any=bool(frame.scenario_urgent.any()),urgency_all=bool(frame.scenario_urgent.all()),
                      stock_changes_order=stock_switch,stock_changes_urgency=urgency_switch,
                      relative_order_span=(high-low)/max(high,1.),sensitivity_reason=reason,
                      check_stock=priority<3,count_priority=priority,count_reason=count_reason)
        summaries.append(common);details.extend(records)
        # Counterfactual at the SAME selected stock and demand, always exactly +7d.
        base=result_by_sku.loc[sku]
        selected_stock=float(base.stock) if pd.notna(base.stock) else np.nan
        safety=float(rates.mean())*cfg.safety_days*cat
        before=inventory_plan(rates,schedules[0],selected_stock,cfg.lead_days,safety,pack,moq)
        after=inventory_plan(rates,schedules[7],selected_stock,cfg.lead_days,safety,pack,moq)
        first=after['first_shortage_index']
        delays.append(dict(sku=sku,delay_stock=selected_stock,delay_order_before=before['recommended'],delay_order_after=after['recommended'],
                           delay_order_change=after['recommended']-before['recommended'],delay_expedite_before=before['expedite_need'],
                           delay_expedite_after=after['expedite_need'],delay_new_urgent=bool(not before['urgent'] and after['urgent']),
                           delay_first_shortage=str(dates[first].date()) if first is not None else '',delay_evaluable=np.isfinite(selected_stock)))
    summary=pd.DataFrame(summaries)
    detail=pd.DataFrame(details)
    delayed=pd.DataFrame(delays,columns=['sku','delay_stock','delay_order_before','delay_order_after','delay_order_change','delay_expedite_before','delay_expedite_after','delay_new_urgent','delay_first_shortage','delay_evaluable'])
    return summary,detail,delayed


def recount_queue(result):
    """Lexicographic priority. Quantities of different SKUs are never summed."""
    queue=result[result.check_stock].copy()
    return queue.sort_values(['count_priority','stock_changes_urgency','stock_changes_order','scenario_snapshot_age_days','relative_order_span','sku'],
                             ascending=[True,False,False,False,False,True],na_position='last').reset_index(drop=True)
