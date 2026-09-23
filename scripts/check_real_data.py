"""Local-only inspection; partner files never copied to the repository."""
import argparse
from pathlib import Path
import json
from time import perf_counter
from axioma.importers import import_supplier_zips
from axioma.engine import calculate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archives', nargs='+')
    args = parser.parse_args()
    started = perf_counter()
    dataset = import_supplier_zips([(Path(p).name, Path(p).read_bytes()) for p in args.archives])
    imported = perf_counter()
    result, _ = calculate(dataset)
    print(json.dumps({'products': len(dataset.products), 'sales': len(dataset.sales), 'incoming': len(dataset.incoming),
                      'blocked': int(result.recommended_qty.isna().sum()), 'orderable': int((result.recommended_qty > 0).sum()),
                      'suppliers': result.supplier.unique().tolist(), 'import_seconds': round(imported-started, 1),
                      'calculation_seconds': round(perf_counter()-imported, 1),
                      'source_checks': dataset.source_checks.to_dict('records'), 'warnings': dataset.warnings}, ensure_ascii=False, indent=2))

