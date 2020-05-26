import argparse
import glob
import os
import sys
import json

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
sns.set(style="darkgrid")

default_lr = 0.0001

def parse_filepath(fp):
    # run-0_env-BreakoutNoFrameskip--v4_learningrate-0.0003_memorycapacity-20000_seed-0
    try:
        loss = pd.read_csv(f"{fp}/loss.csv")
        with open(f"{fp}/params.json", "r") as json_file:
            params = json.load(json_file)
        for k,v in params.items():
            loss[k] = v
        return loss
    except FileNotFoundError as e:
        print(f"Error in parsing filepath {fp}: {e}")
        return None


def collate_results(results_dir):
    dfs = []
    for run in glob.glob(results_dir + '/*'):
        print(f"Found {run}")
        run_df = parse_filepath(run)
        if run_df is None:
            continue
        dfs.append(run_df)
    return pd.concat(dfs, axis=0)


def plot(data, hue, style, seed, savepath=None, show=True):
    print(f"Plotting using hue={hue}, style={lr}, {seed}")

    # If asking for multiple envs, use facetgrid and adjust height
    height = 3 if len(envs) > 2 else 5
    col_wrap = 2 if len(envs) > 1 else 1

    if isinstance(seed, list) or seed == 'average':
        g = sns.relplot(x='step',
                        y='loss',
                        data=data,
                        hue=hue,
                        style=style,
                        kind='line',
                        legend='full',
                        height=height,
                        aspect=1.5,
                        col='env',
                        col_wrap=col_wrap,
                        facet_kws={'sharey': False})

    elif seed == 'all':
        g = sns.relplot(x='frame',
                        y='average_reward',
                        data=data,
                        hue=hue,
                        units='seed',
                        style=style,
                        estimator=None,
                        kind='line',
                        legend='full',
                        height=height,
                        aspect=1.5,
                        col='env',
                        col_wrap=col_wrap,
                        facet_kws={'sharey': False})
    else:
        raise ValueError(f"{seed} not a recognized choice")

    if savepath is not None:
        g.savefig(savepath)

    if show:
        plt.show()


def parse_args():
    # Parse input arguments
    # Use --help to see a pretty description of the arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # yapf: disable
    parser.add_argument('--results-dir', help='Directory for results', default='results/pretraining',
            required=False, type=str)
    parser.add_argument('--create-csv', help='Create csv, overwrites if exists',
            action='store_true')

    parser.add_argument('--query', help='DF query string', type=str)
    parser.add_argument('--hue', help='Hue variable', type=str, )
    parser.add_argument('--style', help='Style variable', type=str)
    parser.add_argument('--seed', help='How to handle seeds', type=str, nargs='*',
            default='average')

    parser.add_argument('--no-plot', help='No plots', action='store_true')
    parser.add_argument('--no-show', help='Does not show plots', action='store_true')
    parser.add_argument('--save-path', help='Save the plot here', type=str)
    # yapf: enable

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.create_csv:
        print("Recreating csv in results directory")
        df = collate_results(args.results_dir)
        df.to_csv(os.path.join(args.results_dir, 'combined.csv'))

    if not args.no_plot:
        assert args.query is not None, "Must pass in query if plotting"
        assert args.hue is not None, "Must pass in hue if plotting"
        assert args.style is not None, "Must pass in style if plotting"

        if args.save_path:
            os.makedirs(os.path.split(args.save_path)[0], exist_ok=True)
        df = pd.read_csv(os.path.join(args.results_dir, 'combined.csv'))
        print(f"Filtering with {args.query}")
        import pdb; pdb.set_trace()
        df = df.query(args.query)
        plot(df, args.hue, args.style, args.seed, args.save_path, not args.no_show)
