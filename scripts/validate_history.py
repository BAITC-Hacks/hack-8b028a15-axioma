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
args=parser.parse_args()
ds=parse_files([(p.name,p.read_bytes()) for p in sorted(args.folder.glob('*.xlsx'))],args.supplier) if args.folder else demo_data()
detail,scores,meta=backtest(ds,max_skus=args.limit,include_pooled=args.pooled)
valid=scores.dropna(subset=['model_wape','baseline_wape'])
meta.update(supplier=args.supplier,scored_skus=len(valid),macro_wape_model=float(valid.model_wape.mean()) if len(valid) else None,macro_wape_baseline=float(valid.baseline_wape.mean()) if len(valid) else None,
    macro_wape_adaptive=float(valid.adaptive_wape.mean()) if len(valid) else None,
    adaptive_wins=int((valid.adaptive_wape<valid.baseline_wape).sum()),
    adaptive_median_wape=float(valid.adaptive_wape.median()) if len(valid) else None)
print(json.dumps(meta,ensure_ascii=False,indent=2,allow_nan=False))
if args.pooled:print('Pooled macro WAPE:',float(valid.pooled_wape.mean()))
