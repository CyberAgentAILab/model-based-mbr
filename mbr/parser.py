import argparse


def get_mbr_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="dataset name e.g. wmt19.de-en, xsum, nocaps")
    parser.add_argument(
        "--n_lines",
        type=int,
        default=4,
        help="number of source inputs to evaluate. default is 4 so that it can be used for debugging",
    )

    parser.add_argument(
        "--model",
        default="None",
        help="default is None which is to select predefined model for each dataset",
    )
    parser.add_argument(
        "--prompt", default="None", help="only applicable for Language models"
    )

    # Sampling algorithm
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="For sample.py, number of samples to generate for sampling algorithm. "
        + "For mbr_engine.py, this is the number of samples to generate for each source input",
    )
    parser.add_argument(
        "--bsz", type=int, default=4, help="batch size for sampling algorithm"
    )
    parser.add_argument(
        "--eps", type=float, default=0.02, help="epsilon for sampling algorithm"
    )
    parser.add_argument(
        "--topk", type=int, default=0, help="topk for sampling algorithm"
    )
    parser.add_argument(
        "--topp", type=float, default=1.0, help="topp for sampling algorithm"
    )

    # MBR Algorithm
    parser.add_argument("--sample_dir", help="directory to save samples")
    parser.add_argument("--algorithm", default="None", help="mbr algorithm")
    parser.add_argument(
        "--recompute_matrix",
        action="store_true",
        help="whether to recompute similarity matrix",
    )

    # Utility function
    parser.add_argument(
        "--sim",
        default="bertscore",
        help="similarity function (utility function) for MBR",
    )
    # Evaluation function
    parser.add_argument(
        "--eval", default="bleu", help="quality metric for evaluating the output"
    )

    return parser
