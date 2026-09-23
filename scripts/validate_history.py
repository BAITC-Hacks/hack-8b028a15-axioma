"""Run historical evaluation; raw private files are read locally only."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data import demo_data,parse_files
from src.validation import backtest

parser=argparse.ArgumentParser()
parser.add_argument('--folder',type=Path)
parser.add_argument('--supplier',default='Демо')
parser.add_argument('--limit',type=int,default=30)
parser.add_argument('--pooled',action='store_true')
parser.add_argument('--output',type=Path,help='Write aggregate metrics only; no raw company data')
args=parser.parse_args()
ds=parse_files([(p.name,p.read_bytes()) for p in sorted(args.folder.glob('*.xlsx'))],args.supplier) if args.folder else demo_data()
detail,scores,meta=backtest(ds,max_skus=args.limit,include_pooled=args.pooled)
valid=scores.dropna(subset=['model_wape','baseline_wape'])
meta.update(supplier=args.supplier,scored_skus=len(valid),macro_wape_model=float(valid.model_wape.mean()) if len(valid) else None,macro_wape_baseline=float(valid.baseline_wape.mean()) if len(valid) else None,
    macro_wape_adaptive=float(valid.adaptive_wape.mean()) if len(valid) else None,
    adaptive_wins=int((valid.adaptive_wape<valid.baseline_wape).sum()),
    adaptive_median_wape=float(valid.adaptive_wape.median()) if len(valid) else None)
if args.pooled:
    meta.update(macro_wape_pooled=float(valid.pooled_wape.mean()),pooled_wins=int((valid.pooled_wape<valid.baseline_wape).sum()))
zero=scores[scores.actual_total.eq(0)]
meta['zero_actual_skus']=len(zero)
meta['zero_actual_forecast_positive']={m:int(detail[detail.sku.isin(zero.sku)].groupby('sku')[m].sum().gt(0).sum()) for m in ['model','baseline','adaptive']+(['pooled'] if args.pooled else [])}
encoded=json.dumps(meta,ensure_ascii=False,indent=2,allow_nan=False)
print(encoded)
if args.output:
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(encoded+'\n',encoding='utf-8')
