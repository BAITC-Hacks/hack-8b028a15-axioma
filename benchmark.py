"""Reproducible chronological benchmark. Company files never leave this machine."""
from pathlib import Path
import argparse,json,time
from src.data import parse_files,unpack_excel_archive,demo_data
from src.backtest import evaluate_dataset

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',nargs='?');p.add_argument('--periods',type=int,default=6)
    p.add_argument('--max-skus',type=int);p.add_argument('--output',default='data/benchmark');p.add_argument('--end',default='2026-08-01');a=p.parse_args()
    start=time.perf_counter()
    if a.archive:
        sup='IEK' if 'iek' in Path(a.archive).name.lower() else 'Systeme Electric'
        ds=parse_files(unpack_excel_archive(Path(a.archive).read_bytes()),sup)
    else:ds=demo_data()
    print(f'Data loaded: {ds.supplier}; {len(ds.products)} SKU',flush=True)
    summary,detail,meta=evaluate_dataset(ds,a.end,a.periods,a.max_skus)
    meta['supplier']=ds.supplier;meta['elapsed_seconds']=round(time.perf_counter()-start,2)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    summary.to_csv(out/'summary.csv',index=False,encoding='utf-8-sig')
    detail.to_csv(out/'detail.csv',index=False,encoding='utf-8-sig')
    (out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary.to_string(index=False),flush=True);print(json.dumps(meta,ensure_ascii=False),flush=True)
