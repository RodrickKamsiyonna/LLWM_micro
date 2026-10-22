"""Prepare UTF-8 text corpora for character-level LLWM training."""

import argparse
from pathlib import Path
from urllib.request import urlretrieve

DATA_PATHS = {
    "shakespeare": Path("data/input.txt"),
    "wikitext": Path("data/wikitext.txt"),
}
SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
WIKITEXT_REPO = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"


def default_data_path(dataset):
    """Return the conventional path for the selected corpus."""
    return DATA_PATHS[dataset]


def prepare_dataset(dataset="shakespeare", output=None, force=False):
    """Download the selected corpus if missing; return its local text-file path.

    WikiText uses the Hugging Face *train* split only. train.py preserves its
    historical 90/10 character split for training and validation.
    """
    if dataset not in DATA_PATHS:
        raise ValueError(f"unknown dataset: {dataset!r}")
    path = Path(output) if output is not None else default_data_path(dataset)
    if path.is_file() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        if dataset == "shakespeare":
            urlretrieve(SHAKESPEARE_URL, temporary)
        else:
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise RuntimeError(
                    "WikiText requires Hugging Face datasets; "
                    "install it with: pip install -r requirements.txt"
                ) from exc
            lines = load_dataset(
                WIKITEXT_REPO, WIKITEXT_CONFIG, split="train"
            )
            # Each row is a *line of text*, not a token ID. Retain original
            # case/punctuation and line boundaries before char-level encoding.
            with temporary.open("w", encoding="utf-8", newline="\n") as file:
                for record in lines:
                    line = record["text"]
                    file.write(line)
                    if not line.endswith("\n"):
                        file.write("\n")
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"downloaded {dataset} corpus is empty")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser(description="Download character-level text datasets")
    parser.add_argument(
        "--dataset", choices=tuple(DATA_PATHS), default="shakespeare",
        help="shakespeare (default) or wikitext (WikiText-2 raw train split)",
    )
    parser.add_argument(
        "--output", default=None, help="optional output .txt path"
    )
    parser.add_argument(
        "--force", action="store_true", help="download again even if file exists"
    )
    args = parser.parse_args()
    path = prepare_dataset(args.dataset, args.output, force=args.force)
    print(f"{args.dataset}: {path} ({len(path.read_text(encoding='utf-8')):,} characters)")


if __name__ == "__main__":
    main()
