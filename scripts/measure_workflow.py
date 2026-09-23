"""Measure actual local processing, not unobserved staff time or money saved."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
from time import perf_counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.data import parse_files
from src.engine import Settings, calculate
from src.robustness import analyse_scenarios, ScenarioSettings


def measure(folder,supplier,repeats):
    runs=[];counts=None
    for iteration in range(repeats):
        start=perf_counter()
        files=[(p.name,p.read_bytes()) for p in sorted(folder.glob('*.xlsx'))]
        if len(files)!=6:raise ValueError(f'{supplier}: expected six XLSX files')
        ds=parse_files(files,supplier);loaded=perf_counter()
        cfg=Settings(forecast_method='pooled',use_source_growth=False)
        result,histories,_=calculate(ds,cfg);forecasted=perf_counter()
        summary,detail,delay=analyse_scenarios(ds,cfg,result,histories,ScenarioSettings());planned=perf_counter()
        merged=result.merge(summary,on='sku',validate='one_to_one')
        data=BytesIO()
        with pd.ExcelWriter(data,engine='openpyxl') as writer:
            merged.to_excel(writer,index=False,sheet_name='Recommendations')
        exported=perf_counter()
        # This is the complete recommendations workbook, not manager approval.
        assert len(pd.read_excel(BytesIO(data.getvalue())))==len(ds.products)
        current=dict(products=len(result),scenarios=len(detail),positive_base_orders=int(result.recommended.gt(0).sum()),
                     unknown_base_orders=int(result.recommended.isna().sum()),
                     stock_changes_order=int(summary.stock_changes_order.sum()),stock_changes_urgency=int(summary.stock_changes_urgency.sum()),
                     new_urgent_delay=int(delay.delay_new_urgent.sum()),delay_evaluable=int(delay.delay_evaluable.sum()))
        if counts is not None:assert counts==current
        counts=current
        runs.append(dict(import_seconds=round(loaded-start,3),forecast_seconds=round(forecasted-loaded,3),
                         scenarios_seconds=round(planned-forecasted,3),export_seconds=round(exported-planned,3),
                         total_seconds=round(exported-start,3)))
        print(f'{supplier}: run {iteration+1}/{repeats}, {runs[-1]["total_seconds"]}s',flush=True)
    # Same data and parameters, change only the outlier-filter switch.
    no_filter=calculate(ds,replace(cfg,remove_outliers=False))[0].set_index('sku')
    base=result.set_index('sku')
    known=base.recommended.notna() & no_filter.recommended.notna()
    delta=base.loc[known,'recommended']-no_filter.loc[known,'recommended']
    comparison=dict(evaluable_orders=int(known.sum()),changed_orders=int((~np.isclose(delta,0)).sum()),
                    smaller_orders=int((delta < -1e-8).sum()),larger_orders=int((delta > 1e-8).sum()),
                    interpretation='Different quantities are not proof of better decisions or savings. No cross-SKU quantity sum.')
    return dict(supplier=supplier,counts=counts,runs=runs,median_seconds=round(median(r['total_seconds'] for r in runs),3),
                min_seconds=min(r['total_seconds'] for r in runs),max_seconds=max(r['total_seconds'] for r in runs),outlier_filter_comparison=comparison)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repeats',type=int,default=3);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not 1<=a.repeats<=10:p.error('repeats must be 1..10')
    reports=[measure(a.root/folder,supplier,a.repeats) for folder,supplier in [('Systeme electric','Systeme Electric'),('IEK','IEK')]]
    try:
        revision=subprocess.run(['git','rev-parse','HEAD'],text=True,capture_output=True)
        commit=revision.stdout.strip() if revision.returncode==0 else 'source archive; commit unavailable'
    except OSError:commit='source archive; git unavailable'
    payload=dict(measured_at=datetime.now(timezone.utc).isoformat(),commit=commit,
                 environment=dict(os=platform.system(),machine=platform.machine(),python=platform.python_version(),pandas=pd.__version__,numpy=np.__version__),
                 scope='XLSX read + forecast + finite scenarios + full recommendation XLSX in memory. Excludes UI, user review and network. No Streamlit cache; OS cache may be warm.',
                 settings=vars(Settings(forecast_method='pooled',use_source_growth=False)),grid=vars(ScenarioSettings()),repeats=a.repeats,reports=reports,
                 unmeasured=['manual Excel preparation time','manager review time','money saved','post-deployment stockouts'])
    a.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Saved aggregate benchmark:',a.output)
if __name__=='__main__':main()
