import argparse
import glob
import os
import sys

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
sns.set(style="darkgrid")

default_lr = 0.0001

def load_split_loss_data(filename='split_loss_online.csv', method='online', n=1000.0):
    df = pd.read_csv(filename)
    df = pd.DataFrame(np.einsum('ijk->ik',df.values[:-401].reshape(-1,int(n),df[:-401].shape[1]))/n, columns=df.columns)
    df = df.reset_index()
    df_rainbow = pd.DataFrame(df['rainbow']).rename({'rainbow':'loss'}, axis=1)
    df_markov = pd.DataFrame(df['markov']).rename({'markov':'loss'}, axis=1)
    df_rainbow['loss_type']='rainbow'
    df_markov['loss_type']='markov'
    df = pd.concat((df_rainbow, df_markov), axis=0)
    df['method']=method
    df = df.reset_index()
    return df
df_online = load_split_loss_data('split_loss_online.csv', 'online')
df_rainbow = load_split_loss_data('split_loss_de.csv', 'rainbow')

data = pd.concat((df_online,df_rainbow),axis=0).reset_index(drop=True)
g = sns.relplot(data=data, x='index', y='loss', hue='method', hue_order=['rainbow','online'], kind='line', col='loss_type')
# g
for a in g.axes.flatten():
    a.set_xlabel('steps (in thousands)')
plt.savefig('split_loss_results.png')
plt.show()
