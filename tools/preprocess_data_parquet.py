"""
Parquet variant of tools/preprocess_data.py — adds a `parquet` reader subcommand using
datatrove's ParquetReader (the bundled preprocess_data.py only exposes `hf` and `jsonl`).

Example:
    python3 tools/preprocess_data_parquet.py \
        --tokenizer-name-or-path /path/to/tokenizer.json \
        --eos-token "</s>" \
        --output-folder datasets/block/tokenized --n-tasks 32 \
        parquet --dataset /path/to/parquet_folder --column text --glob-pattern "*.parquet"
"""

import argparse

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import HuggingFaceDatasetReader, JsonlReader, ParquetReader
from datatrove.pipeline.tokens import DocumentTokenizer


def get_args():
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title="Tokenizer")
    group.add_argument("--tokenizer-name-or-path", type=str, required=True,
                       help="Path to a tokenizer.json / tokenizer dir / hub id.")
    group.add_argument("--eos-token", type=str, default=None,
                       help="EOS token to append after each document. Default: None")

    group = parser.add_argument_group(title="Output data")
    group.add_argument("--output-folder", type=str, required=True,
                       help="Path to the output folder to store the tokenized documents")
    group = parser.add_argument_group(title="Miscellaneous configs")
    group.add_argument("--logging-dir", type=str, default=None)
    group.add_argument("--n-tasks", type=int, default=8)

    sp = parser.add_subparsers(dest="readers", required=True,
                               description="Type of dataset to process: hf | jsonl | parquet")

    p1 = sp.add_parser(name="hf")
    p1.add_argument("--dataset", type=str, required=True)
    p1.add_argument("--column", type=str, default="text")
    p1.add_argument("--split", type=str, default="train")

    p2 = sp.add_parser(name="jsonl")
    p2.add_argument("--dataset", type=str, required=True)
    p2.add_argument("--column", type=str, default="text")
    p2.add_argument("--glob-pattern", type=str, default=None)

    p3 = sp.add_parser(name="parquet")
    p3.add_argument("--dataset", type=str, required=True,
                    help="Folder containing .parquet files (or a single .parquet)")
    p3.add_argument("--column", type=str, default="text")
    p3.add_argument("--glob-pattern", type=str, default=None)

    return parser.parse_args()


def main(args):
    if args.readers == "hf":
        datatrove_reader = HuggingFaceDatasetReader(
            dataset=args.dataset, text_key=args.column, dataset_options={"split": args.split},
        )
    elif args.readers == "parquet":
        datatrove_reader = ParquetReader(
            data_folder=args.dataset, text_key=args.column, glob_pattern=args.glob_pattern,
        )
    else:
        datatrove_reader = JsonlReader(
            data_folder=args.dataset, text_key=args.column, glob_pattern=args.glob_pattern,
        )

    LocalPipelineExecutor(
        pipeline=[
            datatrove_reader,
            DocumentTokenizer(
                output_folder=args.output_folder,
                tokenizer_name_or_path=args.tokenizer_name_or_path,
                eos_token=args.eos_token,
                shuffle_documents=False,  # datatrove 0.5.0 renamed `shuffle` -> `shuffle_documents`
                max_tokens_per_file=1e9,
            ),
        ],
        tasks=args.n_tasks,
        logging_dir=args.logging_dir,
    ).run()


if __name__ == "__main__":
    main(get_args())
