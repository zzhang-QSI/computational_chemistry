import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from mmlib import  fileio


def parse_energy_dat_file(dat_file):
    df= pd.DataFrame( fileio.GetProperties(dat_file))
    return df.set_index('time')


def plot_energy_trajactory(energy_df):
    fig,ax=plt.subplots(figsize=(18,40))
    energy_df.plot.line(subplots=True,ax=ax)


if __name__ == '__main__':
    df=parse_energy_dat_file("../../../data/md/he2.dat")
    plot_energy_trajactory(df)
    plt.show()
