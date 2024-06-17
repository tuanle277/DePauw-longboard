import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from IPython.display import display

def plot_classification_report(classification_report, title='Classification report', cmap='RdBu'):
    # Your plotting logic
    pass

def plot_heatmap(data, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, cmap='RdBu'):
    fig, ax = plt.subplots()
    c = ax.pcolor(data, edgecolors='k', linestyle='dashed', linewidths=0.2, cmap=cmap)
    ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(c)
    plt.show()


# Plot a table of all the training/test results using the classifiers
def display_dict_models(dict_models, sort_by='test_score'):
    cls = [key for key in dict_models.keys()]
    test_s = [dict_models[key]['test_score'] for key in cls]
    training_s = [dict_models[key]['train_score'] for key in cls]
    training_t = [dict_models[key]['train_time'] for key in cls]

    df_ = pd.DataFrame(data=np.zeros(shape=(len(cls),4)), columns = ['classifier', 'train_score', 'test_score', 'train_time'])
    for ii in range(0,len(cls)):
        df_.loc[ii, 'classifier'] = cls[ii]
        df_.loc[ii, 'train_score'] = training_s[ii]
        df_.loc[ii, 'test_score'] = test_s[ii]
        df_.loc[ii, 'train_time'] = training_t[ii]

    display(df_.sort_values(by=sort_by, ascending=False))