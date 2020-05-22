import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import torch
import glob
import os
sns.set(style="darkgrid")


def parse_filepath(fp):
    fp = os.path.split(fp)[1]
    tags = fp.split('_')
    if len(tags) == 4:
        metadata = {
            "env": tags[0],
            "lr": tags[1].split('-')[1],
            "arch": tags[2].split('-')[1],
            "seed": tags[3].split('-')[1],
        }
        return metadata
    else:
        print(f"Error in parsing filepath {fp}")
        return None


def collate_results(results_dir):
    df = []
    for run in glob.glob(results_dir + '/*'):
        print(f"Found {run}")
        metadata = parse_filepath(run)
        if metadata is None:
            continue
        metrics_path = os.path.join(run, 'metrics.pth')
        metrics = torch.load(metrics_path)
        del metrics['best_avg_reward']
        metrics = [dict(zip(metrics, t)) for t in zip(*metrics.values())]
        for datapoint in metrics:
            row = {
                'frame': datapoint['steps'],
                'average_reward': sum(datapoint['rewards']) / len(datapoint['rewards']),
            }
            row.update(metadata)
            for eval_idx, eval_reward in enumerate(datapoint['rewards']):
                row[f"reward_{eval_idx}"] = eval_reward
            df.append(row)
    return pd.DataFrame(df)


def plot(data, envs, lr, arch, seed_type, savepath=None, show=True):
    # Filter based on input
    if envs:
        data = data[data['env'].isin(envs)]
    else:
        envs = list(data['env'].unique())
    if lr:
        data = data[data['lr'].isin(lr)]
    else:
        lr = list(data['lr'].unique())
    if arch:
        data = data[data['arch'].isin(arch)]
    else:
        arch = list(data['arch'].unique())

    print(f"Plotting using {envs}, {lr}, {arch}, {seed_type}")

    # If asking for multiple envs, use facetgrid and adjust height
    height = 3 if len(envs) > 3 else 5

    # If multiple lr and arch, set hue and style accordingly
    if len(lr) > 1 and len(arch) > 1:
        hue = 'lr'
        style = 'arch'
    elif len(lr) > 1 and len(arch) <= 1:
        hue = 'lr'
        style = None
    elif len(arch) > 1 and len(lr) <= 1:
        hue = 'arch'
        style = None
    else:
        hue = None
        style = None

    if seed_type == 'average':
        g = sns.relplot(x='frame',
                        y='average_reward',
                        data=data,
                        hue=hue,
                        style=style,
                        kind='line',
                        legend='full',
                        height=height,
                        aspect=1.5,
                        col='env',
                        facet_kws={'sharey': False})

    elif seed_type == 'all':
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
                        facet_kws={'sharey': False})
    else:
        raise ValueError(f"{seed_type} not a recognized choice")

    if savepath is not None:
        g.savefig(savepath)

    if show:
        plt.show()


def parse_args():
    # Parse input arguments
    # Use --help to see a pretty description of the arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # yapf: disable
    parser.add_argument('--results-dir', help='Directory for results', default='results',
            required=False, type=str)
    parser.add_argument('--create-csv', help='Create csv, overwrites if exists',
            action='store_true')
    parser.add_argument('--envs', help='Env to plot, empty means all', type=str, nargs='*')
    parser.add_argument('--lr', help='LRs to plot, empty means all', type=str, nargs='*')
    parser.add_argument('--arch', help='Arch to plot, empty means all', type=str, nargs='*')
    parser.add_argument('--seed', help='How to handle seeds', type=str, choices=['average', 'all'],
            default='average')
    parser.add_argument('--save-path', help='Save the plot here', type=str)
    parser.add_argument('--no-plot', help='No plots', action='store_true')
    parser.add_argument('--no-show', help='Does not show plots', action='store_true')
    # yapf: enable

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.create_csv:
        print("Recreating results directory")
        df = collate_results(args.results_dir)
        df.to_csv(os.path.join(args.results_dir, 'combined.csv'))

    if not args.no_plot:
        if args.save_path:
            os.makedirs(os.path.split(args.save_path)[0], exist_ok=True)
        df = pd.read_csv(os.path.join(args.results_dir, 'combined.csv'))
        plot(df, args.envs, args.lr, args.arch, args.seed, args.save_path, not args.no_show)