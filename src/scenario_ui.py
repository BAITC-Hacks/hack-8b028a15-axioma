"""Scenario workspace, preserving the audited primary ordering workflow."""
import hashlib
import json
import numpy as np
import pandas as pd
import streamlit as st
from .robustness import ScenarioSettings, analyse_scenarios, recount_queue


@st.cache_data(show_spinner=False)
def scenario_grid(ds,cfg,result,histories,options,version):
    return analyse_scenarios(ds,cfg,result,histories,options)


def render_scenarios(ds,cfg,result,histories,store,dataset_key,version,export_excel,labels):
    st.subheader('Что заказать, что ускорить и что сначала пересчитать')
    st.write('Проверяем сочетания отклонений остатка, спроса и дат поступлений. Модель повторно не обучается. Диапазон показывает только проверенную сетку, а не вероятность или доверительный интервал.')
    c1,c2,c3=st.columns(3)
    stock_pct=c1.number_input('Отклонение остатка, ±%',min_value=0.,max_value=100.,value=20.,step=5.)
    demand_pct=c2.number_input('Отклонение спроса, ±%',min_value=0.,max_value=100.,value=20.,step=5.)
    delay_days=c3.number_input('Задержка в сетке, дней',min_value=0,max_value=90,value=7)
    options=ScenarioSettings(stock_pct,demand_pct,int(delay_days))
    key=hashlib.sha256((result.to_json(date_format='iso')+json.dumps(vars(options),sort_keys=True)+version).encode()).hexdigest()
    requested=st.button('Проверить устойчивость решений',type='primary')
    if requested or len(result)<=10:
        with st.spinner('Перебираю сценарии для товаров…'):
            summary,detail,delay=scenario_grid(ds,cfg,result,histories,options,version)
        st.session_state['scenario_result']=(key,summary,detail,delay)
    saved=st.session_state.get('scenario_result')
    if not saved or saved[0]!=key:
        st.info('Запустите проверку: появятся диапазоны заказа, влияние задержки на 7 дней и очередь пересчёта склада. Для большого каталога расчёт может занять несколько минут.')
        return result
    _,summary,detail,delay=saved
    combined=result.merge(summary,on='sku',validate='one_to_one')
    queue=recount_queue(combined)
    a,b,c=st.columns(3)
    a.metric('Товаров со сценариями',int(summary.scenario_count.gt(0).sum()))
    b.metric('Проверить остаток',len(queue))
    c.metric('Срочно хотя бы в одном сценарии',int(summary.urgency_any.sum()))
    action=st.radio('Выберите действие',['Подготовить заказ','Ускорить поставку','Проверить остаток','Все товары'],horizontal=True)
    if action=='Подготовить заказ':selected=combined[combined.recommended.gt(0)]
    elif action=='Ускорить поставку':selected=combined[combined.urgency_any|combined.expedite_need.gt(0)]
    elif action=='Проверить остаток':
        selected=queue
        st.caption('Сначала остаток, меняющий заказ да/нет или срочность, затем неизвестный остаток. Внутри приоритета учитываются давность снимка и относительная чувствительность. Количества разных товаров не складываются.')
    else:selected=combined
    cols=['sku','name','count_priority','count_reason','recommended','unit','order_min','order_max','robustness'] if action=='Проверить остаток' else ['sku','name','recommended','unit','order_min','order_max','robustness']
    st.dataframe(selected[[c for c in cols if c in selected]].rename(columns=labels),hide_index=True,width='stretch',column_config={'Код 1С':st.column_config.TextColumn(width='small'),'Товар':st.column_config.TextColumn(width='medium'),'Приоритет пересчёта':st.column_config.NumberColumn('Приоритет',width='small'),'Почему пересчитать':st.column_config.TextColumn(width='large'),'Минимум заказа в сценариях':st.column_config.NumberColumn('Заказ от',width='small'),'Максимум заказа в сценариях':st.column_config.NumberColumn('Заказ до',width='small')})
    st.caption('Группы могут пересекаться. Сценарный диапазон не заменяет количество выбранного расчёта и не утверждает заказ.')
    if st.checkbox('Показать задержку на 7 дней'):
        known=delay[delay.delay_evaluable]
        st.markdown(f'**Стали срочными: {int(known.delay_new_urgent.sum())} товаров. Изменился заказ: {int(known.delay_order_change.abs().gt(1e-8).sum())}.**')
        st.caption('Сдвинуты только ожидаемые будущие поступления. Спрос, выбранный остаток и индивидуальный срок нового заказа не меняются. Путь без даты и просроченные поставки не считаются прибывшими.')
        st.dataframe(known.rename(columns=labels),hide_index=True,width='stretch')
        st.download_button('Скачать сравнение задержки',export_excel(known,{'Задержка поступлений, дней':7}),file_name='axioma-delay-7.xlsx')
    if not selected.empty:
        sku=st.selectbox('Открыть решение по товару',selected.sku.tolist(),format_func=lambda k:f"{k} · {combined.set_index('sku').loc[k,'name']}")
        item=combined.set_index('sku').loc[sku];original=ds.products.set_index('sku').loc[sku]
        st.markdown('**Исходные данные → Допущения → Расчёт → Действие**')
        st.write(f'Свободный остаток: {original.stock:g} {item.unit}.' if pd.notna(original.stock) else 'Актуальный остаток неизвестен; он не подменён нулём.')
        st.write('Основание:',item.get('stock_basis','Остаток из входных данных'))
        st.write(item.scenario_assumptions)
        st.write(item.robustness);st.write(item.sensitivity_reason)
        if item.scenario_count:
            st.write(f'Заказ от {item.order_min:g} до {item.order_max:g} {item.unit}; проверено {item.scenario_count:g} сочетаний.')
            with st.expander('Каждое проверенное сочетание'):
                trace=detail[detail.sku.eq(sku)]
                st.dataframe(trace.rename(columns=labels),hide_index=True,width='stretch')
                st.download_button('Скачать сценарии этого товара',export_excel(trace,{**vars(cfg),'scenario_grid':vars(options)}),file_name=f'axioma-scenarios-{sku}.xlsx')
        st.write(item.reason)
        if item.expedite_need>0:st.error(f'До новой поставки {item.order_arrival} не хватает до {item.expedite_need:g} {item.unit}. Нужно ускорение или перемещение.')
        elif item.urgency_any:st.warning('Срочность возникает в части проверенных сценариев; уточните остаток и даты.')
        if item.check_stock:st.info(item.count_reason)
        with st.expander('Внести подтверждённый остаток после пересчёта'):
            counted=st.number_input(f'Подтверждённый свободный остаток · {sku}',min_value=0.,max_value=1e12,value=float(original.stock) if pd.notna(original.stock) else 0.,key=f'count-{dataset_key[:12]}-{cfg.as_of}-{sku}')
            st.caption(f'Укажите фактический свободный остаток на {cfg.as_of}, в {item.unit}. Подтверждение сохранится на ноутбуке для этих файлов и даты. Затем проверьте заказ на вкладке «Рекомендации».')
            if st.button('Применить уточнённый остаток'):
                kind='counts:'+cfg.as_of
                counts=store.preferences(dataset_key,kind,{})
                counts[sku]=float(counted);store.save_preferences(dataset_key,kind,counts)
                st.rerun()
    else:st.info('В этой группе нет товаров.')
    return combined
