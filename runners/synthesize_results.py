"""Aggregates results from the metrics_eval_best_weights.json in a parent folder"""

import argparse
import json
import os
from tabulate import tabulate


parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments',
                    help='Directory containing results of experiments')
parser.add_argument('--result_file', default='test_results.json',
                    help='Name of the files to search and collect result')
parser.add_argument('--save_file', default='test_results.csv',
                    help='Name of the file to save the results summary to.')


def aggregate_metrics(parent_dir, metrics):
    """Aggregate the metrics of all experiments in folder `parent_dir`.

    Assumes that `parent_dir` contains multiple experiments, with their results stored in
    `parent_dir/subdir/metrics_dev.json`

    Args:
        parent_dir: (string) path to directory containing experiments results
        metrics: (dict) subdir -> {'accuracy': ..., ...}
    """
    # Get the metrics for the folder if it has results from an experiment
    metrics_file = os.path.join(parent_dir, '{}'.format(args.result_file))
    if os.path.isfile(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics[parent_dir] = json.load(f)

    # Check every subdirectory of parent_dir recursively
    for subdir in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, subdir)):
            continue
        else:
            aggregate_metrics(os.path.join(parent_dir, subdir), metrics)


def metrics_to_table(metrics):
    # Get the headers from the first subdir. Assumes everything has the same metrics
    headers = metrics[list(metrics.keys())[0]].keys()
    table = [[subdir] + [values[h] for h in headers] for subdir, values in metrics.items()]
    res = tabulate(table, headers, tablefmt='pipe')

    return res


def metrics_to_table_v2(metrics):
    # Get the headers from the first subdir. Assumes everything has the same metrics
    headers = metrics[list(metrics.keys())[0]].keys()

    # list of headers without "mean" and "std"
    headers_clean = ["_".join(h.split("_")[:-1]) for h in headers][0:-1:2]

    table = []
    for subdir, values in metrics.items():
        # construct separate mean, std and header list
        mean_list = []
        std_list = []
        for h in headers:
            if h.split("_")[-1] == "mean":
                mean_list += [values[h]]
            elif h.split("_")[-1] == "std":
                std_list += [values[h]]

        table_line = [subdir] + ["{mean:.3g}({std:.3g})".format(mean=mean_list[i], std=std_list[i]) for i in range(len(headers_clean))]
        table += [table_line]
    res = tabulate(table, headers_clean, tablefmt='pipe')
    return res



if __name__ == "__main__":
    args = parser.parse_args()

    # Aggregate metrics from args.parent_dir directory
    metrics = dict()
    aggregate_metrics(args.parent_dir, metrics)
    table = metrics_to_table_v2(metrics)

    # Display the table to terminal
    print(table)

    # Save results in parent_dir/results.md
    save_file = os.path.join(args.parent_dir, "{}".format(args.save_file))
    with open(save_file, 'w') as f:
        f.write(table)
