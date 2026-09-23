"""Local SQLite audit log: immutable runs, explicit approval and idempotency."""
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone
from dataclasses import asdict
import hashlib,json,math,sqlite3,uuid
import pandas as pd

def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)

def records(frame):
    return json.loads(frame.to_json(orient='records',date_format='iso',force_ascii=False))

def fingerprint(supplier, sources):
    ordered=sorted(sources,key=lambda s:canonical(s))
    return hashlib.sha256(canonical({'supplier':supplier,'sources':ordered}).encode()).hexdigest()

class Store:
    def __init__(self,path='data/axioma.sqlite3'):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS runs(
              id TEXT PRIMARY KEY, created_at TEXT NOT NULL, supplier TEXT NOT NULL,
              dataset_key TEXT NOT NULL, settings_json TEXT NOT NULL, rows_json TEXT NOT NULL, sources_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS approvals(
              id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id), created_at TEXT NOT NULL,
              reviewer TEXT NOT NULL, note TEXT NOT NULL, rows_json TEXT NOT NULL, dedupe TEXT UNIQUE NOT NULL);
            CREATE TABLE IF NOT EXISTS preferences(
              dataset_key TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL,
              updated_at TEXT NOT NULL, PRIMARY KEY(dataset_key,kind));
            ''')

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path,timeout=15)
        db.row_factory=sqlite3.Row;db.execute('PRAGMA foreign_keys=ON')
        try:
            with db:yield db
        finally:db.close()

    def save_run(self,supplier,dataset_key,settings,frame,sources):
        if frame.empty:raise ValueError('Нельзя сохранить пустой расчёт')
        ident='RUN-'+uuid.uuid4().hex[:12]
        with self.connect() as db:
            db.execute('INSERT INTO runs VALUES(?,?,?,?,?,?,?)',
                       (ident,datetime.now(timezone.utc).isoformat(),supplier,dataset_key,
                        canonical(asdict(settings)),canonical(records(frame)),canonical(sources)))
        return ident

    def get_run(self,run_id):
        with self.connect() as db:r=db.execute('SELECT * FROM runs WHERE id=?',(run_id,)).fetchone()
        if r is None:raise ValueError('Расчёт не найден')
        result=dict(r)
        for field in ['settings','rows','sources']:result[field]=json.loads(result.pop(field+'_json'))
        return result

    def approve(self,run_id,selections,reviewer,note=''):
        if not str(reviewer).strip():raise ValueError('Укажите ответственного за утверждение')
        if not selections:raise ValueError('Выберите хотя бы одну позицию')
        run=self.get_run(run_id);catalog={r['sku']:r for r in run['rows']};rows=[];seen=set()
        for choice in selections:
            sku=str(choice['sku']);q=choice['quantity']
            if sku in seen:raise ValueError('Повторяется код товара')
            seen.add(sku)
            if sku not in catalog:raise ValueError('Товар отсутствует в сохранённом расчёте')
            base=catalog[sku]
            if base.get('recommended') is None or base.get('stock') is None:raise ValueError('Для товара не хватает данных')
            if isinstance(q,bool) or not isinstance(q,(int,float)) or not math.isfinite(q) or q<=0:raise ValueError('Количество должно быть положительным конечным числом')
            moq=base.get('moq') or 1.;pack=base.get('pack') or 1.
            if q<moq or not math.isclose(q/pack,round(q/pack),abs_tol=1e-8):raise ValueError('Количество не соответствует минимуму или кратности')
            reason=str(choice.get('reason','')).strip()
            if not math.isclose(q,float(base['recommended']),abs_tol=1e-8) and not reason:
                raise ValueError('При изменении количества укажите причину')
            rows.append({**base,'approved_quantity':float(q),'adjustment_reason':reason})
        rows.sort(key=lambda r:r['sku'])
        dedupe=hashlib.sha256(canonical({'run':run_id,'rows':rows,'reviewer':reviewer.strip(),'note':note}).encode()).hexdigest()
        ident='ORD-'+uuid.uuid4().hex[:12]
        with self.connect() as db:
            old=db.execute('SELECT id FROM approvals WHERE dedupe=?',(dedupe,)).fetchone()
            if old:return old['id']
            db.execute('INSERT INTO approvals VALUES(?,?,?,?,?,?,?)',
                       (ident,run_id,datetime.now(timezone.utc).isoformat(),reviewer.strip(),str(note),canonical(rows),dedupe))
        return ident

    def list_runs(self,supplier=None):
        with self.connect() as db:
            rows=db.execute('SELECT id,created_at,supplier,dataset_key FROM runs '+('WHERE supplier=? ' if supplier else '')+'ORDER BY created_at DESC LIMIT 100',(supplier,) if supplier else ()).fetchall()
        return [dict(r) for r in rows]

    def list_orders(self,supplier=None):
        with self.connect() as db:
            rows=db.execute('SELECT a.*,r.supplier FROM approvals a JOIN runs r ON r.id=a.run_id '+('WHERE r.supplier=? ' if supplier else '')+'ORDER BY a.created_at DESC LIMIT 100',(supplier,) if supplier else ()).fetchall()
        result=[]
        for row in rows:
            r=dict(row);r['rows']=json.loads(r.pop('rows_json'));result.append(r)
        return result

    def save_preferences(self,dataset_key,kind,payload):
        with self.connect() as db:
            db.execute('INSERT INTO preferences VALUES(?,?,?,?) ON CONFLICT(dataset_key,kind) DO UPDATE SET payload=excluded.payload,updated_at=excluded.updated_at',
                       (dataset_key,kind,canonical(payload),datetime.now(timezone.utc).isoformat()))

    def preferences(self,dataset_key,kind,default=None):
        with self.connect() as db:r=db.execute('SELECT payload FROM preferences WHERE dataset_key=? AND kind=?',(dataset_key,kind)).fetchone()
        return json.loads(r['payload']) if r else default
