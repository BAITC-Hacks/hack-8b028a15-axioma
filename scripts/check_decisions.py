"""Reproduce decision-grid checks; print aggregates, never company transaction rows."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data import demo_data, parse_files
from src.engine import Settings, calculate
from src.robustness import ScenarioSettings, analyse_scenarios, recount_queue


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--folder',type=Path)
    parser.add_argument('--supplier',default='Демо')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--method',choices=['auto','pooled','adaptive','classic'],default='pooled')
    args=parser.parse_args()
    started=time.perf_counter()
    files=[(p.name,p.read_bytes()) for p in sorted(args.folder.glob('*.xlsx'))] if args.folder else []
    if args.folder and not files:parser.error('No XLSX files in folder')
    ds=parse_files(files,args.supplier) if files else demo_data()
    cfg=Settings(forecast_method=args.method,use_source_growth=False)
    result,histories,_=calculate(ds,cfg)
    summary,detail,delay=analyse_scenarios(ds,cfg,result,histories,ScenarioSettings())
    assert len(summary)==len(ds.products) and summary.sku.is_unique
    if not detail.empty:
        assert np.isfinite(detail.scenario_order).all() and detail.scenario_order.ge(0).all()
        linked=detail.merge(ds.products[['sku','pack','moq']],on='sku',validate='many_to_one')
        positive=linked[linked.scenario_order.gt(0)]
        assert positive.scenario_order.ge(positive.moq).all()
        assert np.isclose(positive.scenario_order%positive.pack,0).all()
    for _,row in summary.iterrows():
        subset=detail[detail.sku.eq(row.sku)] if not detail.empty else detail
        assert len(subset)==row.scenario_count
        if len(subset):
            assert row.order_min==subset.scenario_order.min()
            assert row.order_max==subset.scenario_order.max()
    known=delay[delay.delay_evaluable].merge(result[['sku','recommended','expedite_need']],on='sku')
    assert np.allclose(known.delay_order_before,known.recommended)
    assert np.allclose(known.delay_expedite_before,known.expedite_need,atol=.0051)
    report=dict(dataset='real' if files else 'synthetic',supplier=args.supplier,files=len(files),products=len(ds.products),
                settings=vars(cfg),grid=vars(ScenarioSettings()),scenarios=len(detail),
                decisions={str(k):int(v) for k,v in summary.robustness.value_counts().items()},
                count_queue=len(recount_queue(summary)),delay_evaluable=len(known),new_urgent_after_7_days=int(known.delay_new_urgent.sum()),
                changed_order_after_7_days=int(known.delay_order_change.abs().gt(1e-8).sum()),
                assertions='passed',elapsed_seconds=round(time.perf_counter()-started,2))
    payload=json.dumps(report,ensure_ascii=False,indent=2)
    print(payload)
    if args.output:args.output.write_text(payload+'\n',encoding='utf-8')

if __name__=='__main__':main()
