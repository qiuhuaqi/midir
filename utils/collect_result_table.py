import os
import argparse
from glob import glob
import pandas as pd


def main(args):
    # collect all analysis results
    df_ls = []
    for model_dir in os.listdir(args.parent_dir):
        model_dir_path = f'{args.parent_dir}/{model_dir}'
        if os.path.isdir(model_dir_path):
            if 'analysis' in os.listdir(model_dir_path):
                model_dir_path = model_dir_path
            else:
                # try versions
                model_dir_path = glob(f'{model_dir_path}/*')[-1]
            if 'test' in os.listdir(model_dir_path):
                model_df = pd.read_csv(f'{model_dir_path}/test/{args.suffix}')
                model_df.index = [model_dir]  # add model dir as index
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
                        help='File path within model_dir',
                        default='analysis/analysis_results.csv')
    parser.add_argument('-s', '--save_path',
                        help='Path to save table CSV')

    args = parser.parse_args()
    main(args)
