import argparse
import os
import pandas as pd


def main(args):
    # collect all analysis results
    df_ls = []
    for model_dir in os.listdir(args.parent_dir):
        model_dir_path = f'{args.parent_dir}/{model_dir}'

        if os.path.isdir(model_dir_path):
            model_df = pd.read_csv(f'{model_dir_path}/{args.suffix}')
            model_df.index = [model_dir]  # add model dir as index

            # for tuning: add hyperparameters to columns
            if args.tuning:
                model_dir_split = model_dir.split('_')
                for idx, name in enumerate(model_dir_split):
                    if idx % 2 == 0:
                        number = model_dir_split[idx + 1]
                        model_df.insert(0, name, [number])
            df_ls += [model_df]
    df = pd.concat(df_ls)

    # save the table in CSV
    if args.save_path is None:
        args.save_path = args.parent_dir + '/table.csv'
    df.to_csv(args.save_path)
    print(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--parent_dir')
    parser.add_argument('-f', '--suffix',
                        help='File path ithin model_dir',
                        default='analysis/analysis_results.csv')
    parser.add_argument('-s', '--save_path',
                        help='Path to save table CSV')
    parser.add_argument('-t', '--tuning',
                        action='store_true',
                        help='For when model_dirs are named as hyperparameters joined by _')

    args = parser.parse_args()
    main(args)
