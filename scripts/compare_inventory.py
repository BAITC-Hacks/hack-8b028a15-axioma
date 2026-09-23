"""Reproduce inventory simulations; publish aggregate results, never source files."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data import demo_data, parse_files
from src.effect import Experiment, compare_inventory, aggregate_metrics, forecast_diagnostics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--folder',type=Path);p.add_argument('--supplier',default='Демо')
    p.add_argument('--start',default='2026-06-01');p.add_argument('--end',default='2026-08-31')
    p.add_argument('--lead',type=int,default=21);p.add_argument('--review',type=int,default=14)
    p.add_argument('--safety',type=int,default=7);p.add_argument('--initial-days',type=float,default=21)
    p.add_argument('--limit',type=int,default=30);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();files=[]
    if args.folder:
        files=[(f.name,f.read_bytes()) for f in sorted(args.folder.glob('*.xlsx'))]
        if not files:p.error('No XLSX inputs')
        ds=parse_files(files,args.supplier)
    else:ds=demo_data()
    start=perf_counter()
    exp=Experiment(args.start,args.end,args.lead,args.review,args.safety,args.initial_days,args.limit)
    r,d,o,e,m,f=compare_inventory(ds,exp)
    diagnostics=forecast_diagnostics(f,d,exp.review_days)
    forecast_summary=[]
    if not diagnostics.empty:
        for policy,g in diagnostics.groupby('policy'):
            valid=g[g.actual.gt(0)]
            forecast_summary.append(dict(policy=policy,evaluated_skus=len(g),positive_actual_skus=len(valid),
                zero_actual_skus=int(g.actual.eq(0).sum()),macro_wape_pct=float(valid.wape_pct.mean()) if len(valid) else None,
                macro_signed_bias_pct=float(valid.signed_bias_pct.mean()) if len(valid) else None,
                overforecast_skus=int(valid.signed_bias_pct.gt(1e-8).sum()),underforecast_skus=int(valid.signed_bias_pct.lt(-1e-8).sum()),
                zero_actual_positive_forecast_skus=int(g.zero_actual_positive_forecast.sum())))
    # Raw SKU/transaction traces are deliberately absent from the public report.
    code_root=Path(__file__).resolve().parents[1]
    code_hash=hashlib.sha256(b''.join(x.read_bytes() for x in sorted((code_root/'src').glob('*.py')))).hexdigest()
    m.pop('sources',None)
    report=dict(measured_at=datetime.now(timezone.utc).isoformat(),python=platform.python_version(),code_sha256=code_hash,
        data_type='Отчёты компании' if files else 'Синтетический пример',
        input_sha256=hashlib.sha256(b''.join(name.encode()+hashlib.sha256(raw).digest() for name,raw in files)).hexdigest() if files else 'demo_data()',
        settings=asdict(exp),metadata=m,elapsed_seconds=round(perf_counter()-start,3),
        aggregates=json.loads(aggregate_metrics(r).to_json(orient='records')),
        forecast_diagnostics=forecast_summary,
        methods=[] if f.empty else json.loads(f.groupby(['policy','method']).size().reset_index(name='forecast_origins').to_json(orient='records')),
        disclosure='June–August already used during development; no independent final test, no true latent demand, no measured financial savings. No mixed-unit stock totals.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(supplier=ds.supplier,elapsed_seconds=report['elapsed_seconds'],aggregates=report['aggregates'],forecast_diagnostics=forecast_summary),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
