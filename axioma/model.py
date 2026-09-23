from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class Dataset:
    products: pd.DataFrame
    sales: pd.DataFrame
    incoming: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=['supplier', 'sku', 'eta', 'qty']))
    stockouts: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=['supplier', 'sku', 'start', 'end']))
    warnings: list[str] = field(default_factory=list)
    source: str = 'Синтетические данные'
    source_checks: pd.DataFrame = field(default_factory=pd.DataFrame)
    monthly_sales: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=['supplier', 'sku', 'month', 'qty']))

    def validate(self):
        for frame, required_cols in [(self.incoming, ['supplier', 'sku', 'eta', 'qty']),
                                     (self.stockouts, ['supplier', 'sku', 'start', 'end'])]:
            if not set(required_cols) <= set(frame.columns):
                raise ValueError(f'Отсутствуют поля дополнительной таблицы: {sorted(set(required_cols) - set(frame.columns))}')
        required = {'supplier', 'sku', 'name', 'stock', 'category', 'lead_days', 'review_days', 'unit', 'moq', 'pack', 'purchase_factor'}
        missing = required - set(self.products.columns)
        if missing:
            raise ValueError(f'В products отсутствуют поля: {sorted(missing)}')
        if self.products.duplicated(['supplier', 'sku']).any():
            raise ValueError('Повторяющиеся пары поставщик/код 1С в products.')
        if not {'supplier', 'sku', 'date', 'qty'} <= set(self.sales.columns):
            raise ValueError('В sales нужны supplier, sku, date, qty.')
        for frame in [self.products, self.sales, self.incoming, self.stockouts]:
            for col in ['supplier', 'sku']:
                if frame[col].isna().any() or frame[col].astype(str).str.strip().eq('').any():
                    raise ValueError(f'Пустое значение {col}.')
                frame[col] = frame[col].astype(str).str.strip()
        if self.products.duplicated(['supplier', 'sku']).any():
            raise ValueError('Повторяющиеся пары поставщик/код 1С после нормализации.')
        for col in ['lead_days', 'review_days', 'moq']:
            values = pd.to_numeric(self.products[col], errors='coerce')
            if values.isna().any() or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f'{col}: требуется неотрицательное число.')
            self.products[col] = values
        for col in ['pack', 'purchase_factor']:
            values = pd.to_numeric(self.products[col], errors='coerce')
            if values.isna().any() or not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError(f'{col}: требуется число больше нуля.')
            self.products[col] = values
        for col in ['lead_days', 'review_days']:
            if ((self.products[col] % 1 != 0) | (self.products[col] > 730)).any():
                raise ValueError(f'{col}: требуется целое число дней от 0 до 730.')
        self.products['stock'] = pd.to_numeric(self.products['stock'], errors='raise')
        if not np.isfinite(self.products['stock'].dropna()).all():
            raise ValueError('Остаток должен быть конечным числом.')
        if 'growth_pct' in self.products:
            self.products['growth_pct'] = pd.to_numeric(self.products['growth_pct'], errors='raise')
            values = self.products['growth_pct'].dropna()
            if not np.isfinite(values).all() or (values < -100).any():
                raise ValueError('Прирост должен быть конечным числом не меньше −100%.')
        if 'unit_conversion_required' in self.products:
            values = self.products['unit_conversion_required'].fillna(False)
            mapped = values.astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False})
            if mapped.isna().any():
                raise ValueError('unit_conversion_required: допустимы true/false.')
            self.products['unit_conversion_required'] = mapped
        if (pd.to_numeric(self.products['stock'], errors='coerce').dropna() < 0).any():
            raise ValueError('Отрицательный доступный остаток: требуется уточнение.')
        for frame, cols in [(self.sales, ['date']), (self.incoming, ['eta']), (self.stockouts, ['start', 'end'])]:
            for col in cols:
                if col in frame and pd.to_datetime(frame[col], errors='coerce').isna().any():
                    raise ValueError(f'Некорректная или отсутствующая дата {col}.')
                frame[col] = pd.to_datetime(frame[col]).dt.normalize()
        if not self.stockouts.empty:
            if (pd.to_datetime(self.stockouts['end']) < pd.to_datetime(self.stockouts['start'])).any():
                raise ValueError('Конец stockout раньше начала.')
        for label, frame in [('sales', self.sales), ('incoming', self.incoming)]:
            values = pd.to_numeric(frame['qty'], errors='coerce')
            if values.isna().any() or not np.isfinite(values).all():
                raise ValueError(f'{label}: некорректное количество.')
            frame['qty'] = values
        if not self.incoming.empty and (self.incoming['qty'] < 0).any():
            raise ValueError('Отрицательное количество в пути.')
        keys = set(zip(self.products.supplier, self.products.sku))
        for label, frame in [('sales', self.sales), ('incoming', self.incoming), ('stockouts', self.stockouts)]:
            if not frame.empty and set(zip(frame.supplier, frame.sku)) - keys:
                raise ValueError(f'{label}: есть коды без карточки товара.')
        if not self.monthly_sales.empty:
            if not {'supplier', 'sku', 'month', 'qty'} <= set(self.monthly_sales):
                raise ValueError('monthly_sales: нужны supplier, sku, month, qty.')
            self.monthly_sales['month'] = pd.to_datetime(self.monthly_sales['month'], errors='raise')
            self.monthly_sales['qty'] = pd.to_numeric(self.monthly_sales['qty'], errors='raise')
            if not np.isfinite(self.monthly_sales['qty']).all():
                raise ValueError('Некорректное количество месячной истории.')
        return self
