import matplotlib

matplotlib.use('Agg')

import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np


# def permutation(statistic, error):


def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation
        http://stackoverflow.com/questions/8930370/where-can-i-find-mad-mean-absolute-deviation-in-scipy
    """
    arr = np.ma.array(arr).compressed()  # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))


if __name__ == "__main__":
    df = pd.read_csv('./vehicles.csv')
    print((df.columns))

    data0 = df.values.T[0]
    data1 = df.values.T[1]

    # Cleaning the data
    data_clean1 = data1[~np.isnan(data1)]
    data_clean0 = data0[0:len(data_clean1)]

    # Printing levels of measurement
    print("This are the figures for the New Fleet")
    print((("Mean: %f") % (np.mean(data0))))
    print((("Median: %f") % (np.median(data0))))
    print((("Var: %f") % (np.var(data0))))
    print((("std: %f") % (np.std(data0))))
    print((("MAD: %f") % (mad(data0))))

    print("This are the figures for Current Fleet")
    print((("Mean: %f") % (np.mean(data_clean1))))
    print((("Median: %f") % (np.median(data_clean1))))
    print((("Var: %f") % (np.var(data_clean1))))
    print((("std: %f") % (np.std(data_clean1))))
    print((("MAD: %f") % (mad(data_clean1))))


    # PLOTS
    # ScatterPlot
    sns_plot = sns.lmplot(df.columns[0], df.columns[1], data=df, fit_reg=False)
    sns_plot.axes[0, 0].set_ylim(0, )
    sns_plot.axes[0, 0].set_xlim(0, )
    sns_plot.savefig("vehicles_scaterplot.png", bbox_inches='tight')
    sns_plot.savefig("vehicles_scaterplot.pdf", bbox_inches='tight')
#
#
    plt.clf()
    # Histogram
    # Histogram for the current Fleet
    sns_plot2 = sns.distplot(data_clean0, bins=20, kde=False, rug=True).get_figure()

    axes = plt.gca()
    axes.set_xlabel('MPG')
    axes.set_ylabel('Number of cars')
    sns_plot2.savefig("vehiclesCurrent_histogram.png", bbox_inches='tight')
    sns_plot2.savefig("vehiclesCurrent_histogram.pdf", bbox_inches='tight')

    # Histogram for the New Fleet
    sns_plot3 = sns.distplot(data_clean1, bins=20, kde=False, rug=True).get_figure()

    axes = plt.gca()
    axes.set_xlabel('MPG')
    axes.set_ylabel('Number of Cars')
    sns_plot3.savefig("vehiclesNew_histogram.png", bbox_inches='tight')
    sns_plot3.savefig("vehiclesNew_histogram.pdf", bbox_inches='tight')
