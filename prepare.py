import argparse
from pathlib import Path
from urllib.request import urlretrieve

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/input.txt')
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urlretrieve('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt', path)
    print(f'{path}: {len(path.read_text()):,} characters')
