"""Optional local verification of participant-provided files. No data is published."""
import argparse,json,time
from pathlib import Path
from src.data import parse_files,unpack_excel_archive
from src.engine import calculate,Settings

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('archives',nargs='+');parser.add_argument('--report',default='data/verification.json');args=parser.parse_args()
    reports=[]
    for path in args.archives:
        supplier='IEK' if 'iek' in Path(path).name.lower() else 'Systeme Electric'
        start=time.perf_counter();files=unpack_excel_archive(Path(path).read_bytes());ds=parse_files(files,supplier)
        unknown=[s for s in ds.sources if s['Тип']=='Не распознан']
        assert len(ds.sources)==6 and not unknown,ds.sources
        assert not ds.products.sku.duplicated().any()
        assert len(ds.transactions)>1000
        result,history,anomalies=calculate(ds,Settings())
        assert len(result)==len(ds.products)
        assert result.recommended.dropna().ge(0).all()
        complete=result.dropna(subset=['recommended'])
        positive=complete[complete.recommended.gt(0)]
        assert (positive.recommended+1e-8>=positive.moq).all()
        assert ((positive.recommended%positive.pack).abs()<1e-8).all()
        reports.append(dict(supplier=supplier,files=len(files),products=len(ds.products),transactions=len(ds.transactions),
                            monthly_rows=len(ds.sales),current_stock_known=int(ds.products.stock.notna().sum()),
                            recommendations=int(result.recommended.gt(0).sum()),missing_stock=int(result.recommended.isna().sum()),
                            anomalies=len(anomalies),seasonality_months=len(ds.seasonal),
                            supplier_growth_known=int(ds.products.source_growth.notna().sum()),
                            elapsed_seconds=round(time.perf_counter()-start,2),warnings=ds.warnings))
        print(json.dumps(reports[-1],ensure_ascii=False),flush=True)
    target=Path(args.report);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8')
