import argparse
import glob
import os
import sys

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
sns.set(style="darkgrid")

pd.read_csv('results/combined.csv')

default_lr = 0.0001

if __name__ == '__main__' and 'ipykernel' in sys.argv[0]:
    sys.argv[1:] = ['--env', 'QbertNoFrameskip-v4']


filtered = df.query("env=='QbertNoFrameskip-v4' and arch!='ari'")
sns.relplot(data=filtered, x='frame', y='average_reward', col='arch', hue='markov_coef', kind='line')
