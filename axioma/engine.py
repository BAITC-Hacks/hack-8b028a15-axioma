"""Deterministic, auditable demand forecast and order-up-to policy."""
from dataclasses import dataclass, field
import math
import numpy as np
import pandas as pd
from .model import Dataset


@dataclass
class Policy:
    as_of: str = '2026-09-22'
    safety_days: dict = field(default_factory=lambda: {'1': 14, '2': 10, '3': 7, 'A': 14, 'B': 10, 'C': 7, 'unknown': 7})
    remove_outliers: bool = True
    compensate_stockouts: bool = True


def clean_transactions(sales: pd.DataFrame, enabled=True):
    """Flag rare extreme document or client-day totals, preserving recurrent bulk demand."""
    frame = sales.copy().reset_index(drop=True)
    frame['date'] = pd.to_datetime(frame['date']).dt.normalize()
    frame['raw_qty'] = pd.to_numeric(frame['qty'])
    frame['qty'] = frame['raw_qty'].clip(lower=0)  # Returns are not new demand.
    frame['outlier'] = False
    if not enabled or len(frame) < 8:
        return frame
    groupings = []
    if 'document' in frame:
        groupings.append(['date', 'document'])
    if 'client_id' in frame and frame.client_id.fillna('').ne('').any():
        groupings.append(['date', 'client_id'])
    if not groupings:
        frame['_event'] = np.arange(len(frame))
        groupings.append(['_event'])
    for keys in groupings:
        valid = frame
        if 'client_id' in keys:
            valid = frame[frame.client_id.fillna('').ne('')]
        amounts = valid.groupby(keys, dropna=False)['qty'].sum()
        positive = amounts[amounts > 0]
        if len(positive) < 8:
            continue
        median = float(positive.median())
        mad = float((positive - median).abs().median())
        threshold = max(median * 5, median + 6 * 1.4826 * mad, 20)
        extreme = amounts[amounts > threshold]
        # Three distinct high-volume dates indicate recurring demand, not one-off demand.
        if extreme.empty:
            continue
        if 'date' in keys:
            distinct_days = extreme.index.get_level_values('date').nunique()
        else:
            distinct_days = frame.loc[frame['_event'].isin(extreme.index), 'date'].nunique()
        if distinct_days >= 3:
            continue
        marked = pd.MultiIndex.from_frame(frame[keys]).isin(extreme.index) if len(keys) > 1 else frame[keys[0]].isin(extreme.index)
        frame.loc[marked, 'outlier'] = True
    frame.loc[frame.outlier, 'qty'] = 0.0
    return frame


def demand_history(sales, stockouts, policy):
    as_of = pd.Timestamp(policy.as_of).normalize()
    valid = sales[pd.to_datetime(sales.date) <= as_of].copy()
    if valid.empty:
        return pd.DataFrame(), pd.DataFrame()
    cleaned = clean_transactions(valid, policy.remove_outliers)
    positives = cleaned.loc[cleaned.raw_qty > 0, 'date']
    # Old returns do not prove observation of zero demand in intervening years.
    start = positives.min() if not positives.empty else cleaned.date.min()
    dates = pd.date_range(start, as_of, freq='D')
    history = pd.DataFrame(index=dates)
    history['raw'] = cleaned.groupby('date')['raw_qty'].sum().reindex(dates, fill_value=0)
    history['clean'] = cleaned.groupby('date')['qty'].sum().reindex(dates, fill_value=0)
    history['stockout'] = False
    for row in stockouts.itertuples():
        history.loc[(history.index >= pd.Timestamp(row.start)) & (history.index <= pd.Timestamp(row.end)), 'stockout'] = True
    history['restored'] = history['clean']
    history['imputed'] = 0.0
    if policy.compensate_stockouts and history.stockout.any():
        available = history[~history.stockout]
        for day in history.index[history.stockout]:
            # Same calendar month and weekday, otherwise neighbouring in-stock days.
            peers = available[(available.index.month == day.month) & (available.index.dayofweek == day.dayofweek)]
            if len(peers) < 4:
                peers = available.loc[(available.index >= day - pd.Timedelta(days=56)) & (available.index <= day + pd.Timedelta(days=56))]
            estimate = float(peers.clean.mean()) if len(peers) else 0.0
            increment = max(0.0, estimate - float(history.loc[day, 'clean']))
            history.loc[day, 'imputed'] = increment
            history.loc[day, 'restored'] += increment
    return history, cleaned


def forecast(history, product, as_of, horizon, monthly_prior=None):
    future = pd.date_range(as_of + pd.Timedelta(days=1), periods=horizon, freq='D')
    if history.empty:
        return pd.Series(0.0, index=future), 1.0, 'Нет истории', {}
    monthly = history.restored.resample('MS').mean()
    counts = history.restored.resample('MS').count()
    full = monthly[counts == monthly.index.days_in_month]
    prior_count = 0
    if monthly_prior is not None and not monthly_prior.empty:
        prior = monthly_prior.copy()
        prior['month'] = pd.to_datetime(prior.month)
        # No overlap, no duplicate demand and no future/incomplete month.
        cutoff = history.index.min().to_period('M').start_time
        prior = prior[(prior.month < cutoff) & (prior.month + pd.offsets.MonthEnd(0) <= as_of)]
        if not prior.empty:
            sums = prior.groupby('month').qty.sum().clip(lower=0)
            daily_prior = sums / sums.index.days_in_month
            prior_count = len(daily_prior)
            full = pd.concat([daily_prior, full]).sort_index()
    factors = {m: 1.0 for m in range(1, 13)}
    if len(full) >= 18:
        # Detrend before estimating month effects: otherwise steady growth is
        # incorrectly learned as a January-to-December seasonal increase.
        axis = full.index.year * 12 + full.index.month
        axis = np.asarray(axis - axis.min(), dtype=float)
        slope, intercept = np.polyfit(axis, full.to_numpy(dtype=float), 1)
        level = np.maximum(slope * axis + intercept, max(float(full.mean()) * .1, .001))
        ratios = full / level
        seasonal = ratios.groupby(ratios.index.month).median().clip(0.15, 4.0)
        factors.update({int(k): float(v) for k, v in seasonal.dropna().items()})
        norm = np.mean(list(factors.values()))
        factors = {m: x / norm for m, x in factors.items()}
    deseason = history.restored / pd.Series([factors[d.month] for d in history.index], index=history.index)
    baseline = float(deseason.tail(90).mean())
    inferred = 1.0
    if len(full) >= 6:
        adjusted = full / pd.Series([factors[d.month] for d in full.index], index=full.index)
        previous = float(adjusted.iloc[-6:-3].mean())
        recent = adjusted.iloc[-3:]
        # Trend accepted only if all three months move in the same direction.
        if previous > 0 and ((recent > previous * 1.05).all() or (recent < previous * .95).all()):
            inferred = float(np.clip(recent.mean() / previous, .5, 2))
    growth = product.get('growth_pct', np.nan)
    if pd.notna(growth):
        multiplier = max(0.0, 1.0 + float(growth) / 100)
        method = 'Заданный прирост заменяет автоматический тренд'
    else:
        # Baseline already contains recent growth: only half of the observed ratio extrapolates.
        multiplier = math.sqrt(inferred)
        method = 'Сезонность + устойчивый тренд' if multiplier != 1 else 'Сезонность + регулярный спрос'
    values = [baseline * factors[d.month] * multiplier for d in future]
    if prior_count:
        method += f'; ранняя месячная история: {prior_count} мес.'
    return pd.Series(values, index=future), multiplier, method, factors


def calculate(dataset: Dataset, policy: Policy | None = None):
    dataset.validate()
    policy = policy or Policy()
    if any(not np.isfinite(v) or v < 0 for v in policy.safety_days.values()):
        raise ValueError('Страховой запас в днях должен быть конечным неотрицательным числом.')
    as_of = pd.Timestamp(policy.as_of).normalize()
    result, details = [], {}
    sale_groups = {key: group for key, group in dataset.sales.groupby(['supplier', 'sku'], sort=False)}
    outage_groups = {key: group for key, group in dataset.stockouts.groupby(['supplier', 'sku'], sort=False)}
    incoming_groups = {key: group for key, group in dataset.incoming.groupby(['supplier', 'sku'], sort=False)}
    monthly_groups = {key: group for key, group in dataset.monthly_sales.groupby(['supplier', 'sku'], sort=False)}
    for _, product in dataset.products.iterrows():
        supplier, sku = str(product.supplier), str(product.sku)
        sales = sale_groups.get((supplier, sku), dataset.sales.iloc[:0])
        outages = outage_groups.get((supplier, sku), dataset.stockouts.iloc[:0])
        incoming = incoming_groups.get((supplier, sku), dataset.incoming.iloc[:0]).copy()
        history, events = demand_history(sales, outages, policy)
        lead, review = int(product.lead_days), int(product.review_days)
        horizon = max(1, lead + review)
        prior = monthly_groups.get((supplier, sku), dataset.monthly_sales.iloc[:0])
        future, multiplier, method, factors = forecast(history, product, as_of, horizon, prior)
        safety_days = policy.safety_days.get(str(product.category), policy.safety_days.get('unknown', 7))
        safety = float(future.mean() * safety_days)
        if not incoming.empty:
            incoming['eta'] = pd.to_datetime(incoming.eta)
        due = incoming[(incoming.eta > as_of) & (incoming.eta <= future.index[-1])] if not incoming.empty else incoming
        in_time = float(due.qty.sum())
        overdue = float(incoming.loc[incoming.eta <= as_of, 'qty'].sum()) if not incoming.empty else 0.0
        stock = float(product.stock) if pd.notna(product.stock) else np.nan
        needs_conversion = bool(product.get('unit_conversion_required', False)) if pd.notna(product.get('unit_conversion_required', False)) else False
        known = pd.notna(stock) and not history.empty and not needs_conversion
        need = max(0.0, float(future.sum()) + safety - stock - in_time) if known else np.nan
        factor, pack, moq = float(product.purchase_factor), float(product.pack), float(product.moq)
        buy_qty = math.ceil(max(need / factor, moq) / pack - 1e-10) * pack if known and need > 0 else (0.0 if known else np.nan)
        stock_qty = buy_qty * factor
        risk_date = None
        if pd.notna(stock):
            balance = stock
            arrivals = due.groupby('eta').qty.sum() if not due.empty else pd.Series(dtype=float)
            for day, demand in future.items():
                balance += float(arrivals.get(day, 0)) - float(demand)
                if balance < -1e-9:
                    risk_date = day
                    break
        urgency = 'Уточнить данные' if not known else ('Срочно' if risk_date is not None and risk_date <= as_of + pd.Timedelta(days=lead) else ('Плановый' if buy_qty > 0 else 'Не заказывать'))
        removed = float(events.loc[events.outlier, 'raw_qty'].clip(lower=0).sum()) if not events.empty else 0.0
        lost = float(history.imputed.sum()) if not history.empty else 0.0
        notes = str(product.get('assumptions', '') or '')
        if overdue:
            notes += f' Просроченное поступление {overdue:g} не вычтено: подтвердите получение.'
        if not known:
            explanation = 'Расчёт заблокирован: ' + ('нет актуального остатка. ' if pd.isna(stock) else '') + ('нет истории продаж. ' if history.empty else '') + ('не подтверждён перевод единиц закупки. ' if needs_conversion else '') + notes
        else:
            explanation = (f'Прогноз на {horizon} дн. {future.sum():.1f} + запас категории {product.category} '
                           f'({safety_days:g} дн.) {safety:.1f} − остаток {stock:g} − поступления в срок {in_time:g} '
                           f'= потребность {need:.1f} {product.unit}. Заказ {buy_qty:g} ед. закупки '
                           f'× {factor:g}; минимум {moq:g}, кратность {pack:g}. '
                           f'Коэффициент роста {multiplier:.3f}. Исключено {removed:g}; восстановлено {lost:.1f}. {notes}')
        result.append(dict(supplier=supplier, sku=sku, article=product.get('article', sku), name=product['name'],
                           category=str(product.category), unit=product.unit, purchase_unit=product.get('purchase_unit', product.unit),
                           stock=stock, incoming=in_time, demand=round(float(future.sum()), 2), safety=round(safety, 2),
                           recommended_qty=buy_qty, stock_units=stock_qty, purchase_factor=factor, pack=pack, moq=moq,
                           urgency=urgency, shortage_date=str(risk_date.date()) if risk_date is not None else '',
                           excluded_qty=removed, restored_qty=round(lost, 2), growth_factor=round(multiplier, 4),
                           method=method, explanation=explanation, assumptions=notes))
        details[(supplier, sku)] = {'history': history, 'forecast': future, 'events': events, 'seasonality': factors, 'monthly_prior': prior}
    if not result:
        raise ValueError('Справочник товаров пуст.')
    return pd.DataFrame(result).sort_values(['supplier', 'sku']).reset_index(drop=True), details
