"""Run from the repository root: python -m scripts.generate_demo."""
from pathlib import Path
from axioma.demo import make_demo

if __name__ == '__main__':
    target = Path('data/demo')
    target.mkdir(parents=True, exist_ok=True)
    dataset = make_demo()
    for name in ['products', 'sales', 'incoming', 'stockouts']:
        getattr(dataset, name).to_csv(target / f'{name}.csv', index=False, encoding='utf-8-sig')
    print('Synthetic demo generated in data/demo')

