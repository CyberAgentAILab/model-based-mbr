import os
import argparse
import json

import numpy as np
import pandas as pd

from utility_func import *
from utils import (
    load_dataset,
    load_matrix,
    load_samples_from_file,
    result_dir,
    matrix_dir,
)  # , approx_dir, diverse_dir
from parser import get_mbr_parser

from policy.mbr import compute_score_matrix, compute_mbr


def compute_score(df, d_best, trg, compute_evaluate, src=None):
    d_hyp = df.iloc[d_best]["text"]
    d_score = compute_evaluate(d_hyp, trg, src)
    return d_score


if __name__ == "__main__":
    """
    This script is the "main function" of the experiment.
    """
    parser = get_mbr_parser()
    args = parser.parse_args()

    dataset = args.dataset
    model_name = args.model

    sample_dir = args.sample_dir

    n_lines = args.n_lines
    n_samples = args.n_samples

    epsilon = args.eps
    topk = args.topk
    topp = args.topp

    sim = args.sim
    eval_func = args.eval

    # Algorithm config
    algorithm = args.algorithm
    recompute_matrix = args.recompute_matrix

    compute_similarity, similarity = load_similarity(sim)
    compute_distance = load_distance(sim, compute_similarity)
    compute_evaluate, evaluator = load_evaluate(eval_func, sim, similarity)

    src_lines = load_dataset(dataset)  # src is used only by comet and clip.
    trg_lines = load_dataset(dataset, ref=True)

    # client = boto3.client("s3")

    model_n = os.path.basename(model_name)

    os.makedirs(os.path.join(matrix_dir, dataset, model_n), exist_ok=True)

    files = sorted(os.listdir(sample_dir))

    filtered_files = load_samples_from_file(files, epsilon, topk, topp)

    assert len(filtered_files) > 0

    print("first 10 files=", filtered_files[:10])

    rows = []

    for filename in filtered_files:

        sample_id = int(filename.split("_")[0])
        assert "{:04}".format(sample_id) in filename

        if sample_id >= n_lines:
            break

        src_input = src_lines[sample_id]
        trg = trg_lines[sample_id]

        df = pd.read_csv(os.path.join(sample_dir, filename))

        assert len(df) >= n_samples
        df = df[:n_samples]

        df.fillna(
            "", inplace=True
        )  # TODO: This is needed to remove empty strings. In reality empty strings can be ignored. probably it's better to drop.
        hyp = df.iloc[:]["text"]

        if not recompute_matrix:
            matrix = load_matrix(
                os.path.join(matrix_dir, dataset, model_n), filename, sim, n_samples
            )
        else:
            matrix = None
        if matrix is None:
            matrix_filename = filename + "_" + sim + "_" + str(n_samples)
            matrix_path = os.path.join(matrix_dir, dataset, model_n, matrix_filename)

            matrix = compute_score_matrix(
                hyp, compute_similarity, [src_input] * len(hyp)
            )
            np.savetxt(matrix_path, matrix)

        if algorithm in ["None"]:

            # MBR: Monte Carlo Estimate
            ed_best = compute_mbr(matrix=matrix)
            ed_score = compute_score(df, ed_best, trg, compute_evaluate, src=src_input)

            # MBMBR: Model-Based Estimate
            aed_best = compute_mbr(matrix=matrix, weights=df["probability"])
            aed_score = compute_score(
                df, aed_best, trg, compute_evaluate, src=src_input
            )

            # MBMBR_L: Model-Based Estimate with Length Normalization
            logprob = np.log(df["probability"])
            # using the number of words as length is good enough and more applicable.
            seq_lengths = df["text"].str.split(" ").apply(lambda seq: max(1, len(seq)))
            ln_aed_best = compute_mbr(
                matrix=matrix, weights=np.exp(logprob / seq_lengths)
            )
            ln_aed_score = compute_score(
                df, ln_aed_best, trg, compute_evaluate, src=src_input
            )

            row = [
                sample_id,
                ed_score,
                ed_best,
                aed_score,
                aed_best,
                ln_aed_score,
                ln_aed_best,
            ]
        else:
            assert False
        rows.append(row)

    if algorithm == "None":
        columns = [
            "sample_id",
            "ed_score",
            "ed_best",
            "aed_score",
            "aed_best",
            "ln_aed_score",
            "ln_aed_best",
        ]
        postfix = ""
    else:
        assert False

    df = pd.DataFrame(rows, columns=columns)

    filename = "{}_{}_{:03}_{:.2f}_{:02d}_{:.2f}_{}_{}{}.csv".format(
        dataset, model_n, n_samples, epsilon, topk, topp, sim, eval_func, postfix
    )

    df_path = os.path.join(result_dir, filename)
    os.makedirs(result_dir, exist_ok=True)
    df.to_csv(df_path, index=False)
