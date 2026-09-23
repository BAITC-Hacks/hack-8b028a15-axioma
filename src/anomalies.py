"""Auditable detection of isolated large purchases, with manager overrides."""
import hashlib
import numpy as np
import pandas as pd

EVENT_COLUMNS=['event_id','sku','date','document','original','regular','excluded','reason','decision','recurring_months','span_days']

def clean_transactions(tx, decisions=None):
    if tx.empty:return tx.copy(),pd.DataFrame(columns=EVENT_COLUMNS)
    decisions=decisions or {}
    if any(v not in ('auto','regular','one_off') for v in decisions.values()):raise ValueError('Неизвестное решение по всплеску')
    t=tx.copy();t['quantity']=pd.to_numeric(t.quantity,errors='raise').astype(float)
    if not np.isfinite(t.quantity).all():raise ValueError('Некорректное количество в детализации')
    t['day']=pd.to_datetime(t.date).dt.normalize()
    t['client_id']=t.client_id.fillna('').astype(str)
    # First preserve invoices, then group known clients across split invoices.
    docs=t.groupby(['sku','document','day','client_id'],as_index=False,dropna=False).quantity.sum()
    docs['identity']=np.where(docs.client_id.ne(''),'client:'+docs.client_id,'invoice:'+docs.document.astype(str))
    events=[];result=[]
    for sku,g in docs.groupby('sku',sort=False):
        g=g.copy()
        purchases=g.groupby(['identity','day'],as_index=False).quantity.sum()
        pos=purchases.loc[purchases.quantity.gt(0),'quantity']
        if len(pos)<8:result.append(g);continue
        median=float(pos.median());q1,q3=pos.quantile([.25,.75])
        threshold=max(5*median,float(q3+3*(q3-q1)),1.)
        large=purchases[purchases.quantity>threshold]
        # Wholesale demand can be frequent even when customer IDs are absent.
        # Retain a persistent large-order tier; still reject unusually large
        # events relative to that tier. Three consecutive project days alone
        # cannot satisfy this rule.
        wholesale_months=int(large.day.dt.to_period('M').nunique())
        wholesale_span=int((large.day.max()-large.day.min()).days) if len(large) else 0
        wholesale=len(large)>=12 and wholesale_months>=4 and wholesale_span>=120
        if wholesale:
            lq1,lq3=large.quantity.quantile([.25,.75]);lmed=float(large.quantity.median())
            wholesale_cap=max(5*lmed,float(lq3+3*(lq3-lq1)))
        else:wholesale_cap=0
        recurrence={}
        for identity,history in large.groupby('identity'):
            months=history.day.dt.to_period('M').nunique()
            span=int((history.day.max()-history.day.min()).days)
            recurrence[identity]=(int(months),span)
        for row in large.itertuples(index=False):
            months,span=recurrence[row.identity]
            recurring_client=row.identity.startswith('client:') and months>=3 and span>=90
            recurring_tier=wholesale and row.quantity<=wholesale_cap
            recurring=recurring_client or recurring_tier
            stable=f'{sku}|{row.day.isoformat()}|{row.identity}|{row.quantity:.8g}'
            event_id=hashlib.sha256(stable.encode()).hexdigest()[:20]
            choice=decisions.get(event_id,'auto')
            keep=choice=='regular' or (choice=='auto' and recurring)
            ix=g.identity.eq(row.identity)&g.day.eq(row.day)&g.quantity.gt(0)
            total=float(g.loc[ix,'quantity'].sum())
            replacement=float(row.quantity) if keep else median
            if not keep and total>0:
                # Preserve negative returns while replacing positive purchase volume.
                negative=float(g.loc[g.identity.eq(row.identity)&g.day.eq(row.day)&g.quantity.lt(0),'quantity'].sum())
                g.loc[ix,'quantity']*=max(0.,median-negative)/total
            label='Группа клиента' if row.identity.startswith('client:') else row.identity.removeprefix('invoice:')
            reason=('Подтверждено менеджером: регулярная продажа' if choice=='regular' else
                    'Подтверждено менеджером: разовый заказ' if choice=='one_off' else
                    'Регулярные крупные покупки клиента в разных месяцах' if recurring_client else
                    'Устойчивый поток крупных накладных по SKU' if recurring_tier else
                    'Разовый всплеск клиента за день' if row.identity.startswith('client:') else 'Крупная разовая накладная')
            events.append(dict(event_id=event_id,sku=sku,date=row.day,document=label,
                               original=float(row.quantity),regular=replacement,excluded=max(0.,float(row.quantity)-replacement),
                               reason=reason,decision=choice,recurring_months=months,span_days=span))
        result.append(g)
    cleaned=pd.concat(result,ignore_index=True).drop(columns='identity').rename(columns={'day':'date'})
    return cleaned,pd.DataFrame(events,columns=EVENT_COLUMNS)
