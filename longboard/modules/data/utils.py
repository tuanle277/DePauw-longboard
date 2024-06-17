from collections import Counter 
from imblearn.over_sampling import SMOTEN 
from sklearn.model_selection import StratifiedKFold

import numpy as np 
import scipy 


def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entropy=scipy.stats.entropy(probabilities)

    # Shannon entropy/relative entropy of given distribution(s): How much information do we need to decode the outcome
    # The more uncertain you are, the more information you need to decode in the product
    return entropy

def calculate_statistics(list_values):
    rms = np.nanmean(np.sqrt(list_values**2))
    return [rms]

def calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_mean_crossings]

def get_features(list_values):
    entropy = calculate_entropy(list_values)
    crossings = calculate_crossings(list_values)
    statistics = calculate_statistics(list_values)
    return entropy, crossings, statistics

# Stratified sampling
# input: array of raw sentences and its elements' labels
# output: generator of n numbers of partitions of splits
def stratefy_sample(n_splits):
  # Number of partitions the dataset is split into

  skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
  return skf

def generate_more_data(x, y):
    """
    Generates additional data by applying transformations to the existing data.

    Args:
        x (numpy.ndarray): The input features.
        y (numpy.ndarray): The input labels.

    Returns:
        tuple: A tuple containing the following elements:
            x (numpy.ndarray): The augmented features.
            y (numpy.ndarray): The augmented labels.
    """
    # Your data augmentation logic
    print(f"Original class counts: {Counter(y)}")

    # Instantiate SMOTENC algorithm
    sm = SMOTEN(random_state=123)
    x_res, y_res = sm.fit_resample(x, y)

    print(f"Class counts after resampling {Counter(y_res)}")
    return x_res, y_res

# input: the time column
# output: an array of values indicating the starting point(secs) of each new data collection window
def get_times(time_data, window_length):
	times = []
	for i in range(int(min(time_data)), int(max(time_data)) + window_length - 1, window_length):
		times.append(i)

	print(times)
	return times