import pandas as pd
import numpy as np

from sklearn import preprocessing

from utils import get_times, generate_more_data
from feature_extraction import get_features

def get_features_train_data(window_length):
    """
    Reads data from multiple CSV files, processes the data, and returns features and labels for training a machine learning model.

    Args:
        window_length (int): The length of the sliding window used for feature extraction.

    Returns:
        x_train (numpy.ndarray): Features for training the model.
        y_train (numpy.ndarray): Labels corresponding to the features.
    """
    file_names = [
        "pumping_.csv",
        "pumping_ - goofy, phone in left hand.csv",
        "pumping_ - goofy, phone in right hand.csv",
        "pumping_, goofy, chest pocket.csv",
        "pumping_ - standard, low deck, right hip pocket.csv",
        "pumping_ - goofy, low deck, right hip pocket.csv",
        "pumping_ - right hip, goofy, topmount.csv",
        "pushing_ - standard, low board, right hip.csv",
        "pushing_2.csv",
        "pushing_ - standard.csv",
        "pushing_ - goofy.csv",
        "pushing_ - standard, low deck, right hip pocket.csv",
        "pushing_ - goofy, low deck, right hip pocket.csv",
        "coasting_.csv",
        "coasting_ - right hip pocket, top mount, goofy.csv",
        "coasting_ - low board, right hip, goofy, crouch.csv"
    ]

    path = "" ## <- change this to data path

    x_train = []
    y_train = []
    positional_x, positional_y, positional_z, labels, pred_holder = [], [], [], [], []
    total_data_points = 0

    for file_name in file_names:
        print(f"Processing file: {file_name}")
        print("__________________")

        columns = ["az", "ay", "ax", "time", "Azimuth", "Pitch", "Roll"]
        df = pd.read_csv(path + file_name, usecols=columns)

        if file_name == "pumping_.csv":
            df = df.loc[(df["time"] > 10) & (df["time"] < df.iloc[-1]["time"] - 3)]
        else:
            df = df.loc[(df["time"] > 3) & (df["time"] < df.iloc[-1]["time"] - 3)]

        rotated_df = rotate_phone(df)
        df = rotate_phone(df)

        df, positional_data = get_displacement_data(df, file_name.split("_")[0])
        positional_x.extend(positional_data[0])
        positional_y.extend(positional_data[1])
        positional_z.extend(positional_data[2])
        labels.extend(positional_data[3])

        features, num_data_points = get_features_by_window(df, window_length, file_name.split("_")[0])
        total_data_points += num_data_points
        x_train.extend(features)
        y_train.extend([file_name.split("_")[0]] * features.shape[0])

    print(f"Total number of data points: {total_data_points}")

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # Remove any invalid data points
    valid_indices = ~(np.any(np.isnan(x_train), axis=1) | np.any(x_train == 0, axis=1))
    x_train = x_train[valid_indices]
    y_train = y_train[valid_indices]

    positional_x = np.stack(positional_x)
    positional_y = np.stack(positional_y)
    positional_z = np.stack(positional_z)

    with open(path + 'positional_x.npy', 'wb') as f:
        np.save(f, positional_x)

    with open(path + 'positional_y.npy', 'wb') as f:
        np.save(f, positional_y)

    with open(path + 'positional_z.npy', 'wb') as f:
        np.save(f, positional_z)

    with open(path + 'pred_holder.npy', 'wb') as f:
        np.save(f, pred_holder)

    x_train, y_train = generate_more_data(x_train, y_train)

    return x_train, y_train


# rotate around x-axis
def rotateX(rad):
	return ([1, 0, 0],
	 				[0, np.cos(rad), np.sin(rad)],
	  			[0, -np.sin(rad), np.cos(rad)])

# rotate around y-axis
def rotateY(rad):
	return ([np.cos(rad), 0, -np.sin(rad)],
	 				[0, 1, 0],
	  			[np.sin(rad), 0, np.cos(rad)])

# rotate around z-axis
def rotateZ(rad):
	return ([np.cos(rad), np.sin(rad), 0],
	 				[-np.sin(rad), np.cos(rad), 0],
	  			[0, 0, 1])

# rotate around all three axis
def rotation3D(matrix, x, y, z):
	alpha, beta, theta = np.deg2rad(x), np.deg2rad(y), np.deg2rad(z)

	rotation_matrix = np.matmul(np.matmul(rotateZ(theta), rotateY(beta)), rotateX(alpha))
	return np.matmul(rotation_matrix, matrix)

# rotate the data into a universal coordinate system by reversing the phone's rotation at the time of data collection
def rotate_phone(df):
    """
    Rotates the phone data based on the provided logic.

    Args:
        df (pandas.DataFrame): The input data frame containing the phone data.

    Returns:
        pandas.DataFrame: The rotated data frame.
    """
    # Your rotation logic
    rotated_pos = []
    for index in range(len(df)):
        azimuth, pitch, roll = df['Azimuth'].iat[index], df['Pitch'].iat[index], df['Roll'].iat[index]
        pos_matrix = (df['ax'].iat[index], df['ay'].iat[index], df['az'].iat[index])
        rotated_pos.append(rotation3D(pos_matrix, -pitch, -roll, -azimuth))

    df['rotated_ax'] = [x[0] for x in rotated_pos]
    df['rotated_ay'] = [x[1] for x in rotated_pos]
    df['rotated_az'] = [x[2] for x in rotated_pos]

def get_displacement_data(df, label):
    """
    Calculates displacement data from the input data frame.

    Args:
        df (pandas.DataFrame): The input data frame containing the sensor data.
        label (str): The label associated with the data frame.

    Returns:
        tuple: A tuple containing the following elements:
            positional_x (list): List of x-coordinate displacements.
            positional_y (list): List of y-coordinate displacements.
            positional_z (list): List of z-coordinate displacements.
            labels (list): List of labels corresponding to the displacements.
    """
    delta_v, xt, delta_t, prev_pos, pos = 0, [[0, 0, 0]], 0, [0] * 3, [0] * 3
    positional_x, positional_y, positional_z = [], [], []

    for i in range(len(df) - 1):
        delta_t = df.iloc[i + 1]['time'] - df.iloc[i]['time']
        # delta_v = [df.iloc[i]['rotated_ax'], df.iloc[i]['rotated_ay'], df.iloc[i]['rotated_az']]
        delta_v = [df.iloc[i]['ax'], df.iloc[i]['ay'], df.iloc[i]['az']]
        pos = [x + y for (x, y) in zip([(2 * y) - z for (y, z) in zip(pos, prev_pos)], [delta_v[0] * (delta_t ** 2), delta_v[1] * (delta_t ** 2), delta_v[2] * (delta_t ** 2)])]
        xt.append(pos)
        prev_pos = pos

        positional_x.append(pos[0])
        positional_y.append(pos[1])
        positional_z.append(pos[2])

    labels = [label] * len(positional_x)
    positional_data = [positional_x, positional_y, positional_z, labels]

    df['rotated_xx'] = [x[0] for x in xt]
    df['rotated_xy'] = [x[1] for x in xt]
    df['rotated_xz'] = [x[2] for x in xt]

    return df, positional_data


def get_features_by_window(dataset, window_length, label):
    """
    Extracts features from the input data frame using a sliding window approach.

    Args:
        df (pandas.DataFrame): The input data frame containing the sensor data.
        window_length (int): The length of the sliding window.
        label (str): The label associated with the data frame.

    Returns:
        tuple: A tuple containing the following elements:
            features (numpy.ndarray): The extracted features.
            num_data_points (int): The number of data points in the extracted features.
    """
    # Your feature extraction logic
    featuresPerWindow = []
    times = get_times(dataset['time'], window_length)
    j = 0
    currentTime = times[j]
    total = 0
    # Create a dictionary of all the different data within 1 time window (period window)
    dict = {"time": [], "WindowStart": [],"xx": [], "xy": [], "xz": [], "az": [], "ax": [], "ay": [], "Azimuth": [], "Pitch": [], "Roll": []}
    for i in range(len(dataset)):

        # Get data from csv
        time = dataset.iloc[i]['time']
        xx = dataset.iloc[i]['rotated_xx']
        xy = dataset.iloc[i]['rotated_xy']
        xz = dataset.iloc[i]['rotated_xz']
        az = dataset.iloc[i]['rotated_az']
        ax = dataset.iloc[i]['rotated_ax']
        ay = dataset.iloc[i]['rotated_ay']
        # az = dataset.iloc[i]['az']
        # ax = dataset.iloc[i]['ax']
        # ay = dataset.iloc[i]['ay']
        Azimuth = dataset.iloc[i]['Azimuth']
        Pitch = dataset.iloc[i]['Pitch']
        Roll = dataset.iloc[i]['Roll']

    # Insert to dict
        dict['time'].append(time)
        dict['WindowStart'].append(currentTime)
        dict['xx'].append(xx)
        dict['xy'].append(xy)
        dict['xz'].append(xz)
        dict['ax'].append(ax)
        dict['ay'].append(ay)
        dict['az'].append(az)
        dict['Azimuth'].append(Azimuth)
        dict['Pitch'].append(Pitch)
        dict['Roll'].append(Roll)

    # when this is the last window
        if (i + 1 >= len(dataset) or j == len(times) - 1):
        # Process the features of the final window
            if len(dict['xx']) > 125:
                currentWindowFeatures = get_features(dict, label)
                if ~np.isnan(currentWindowFeatures).any():
                    featuresPerWindow.append(currentWindowFeatures)
            break

    # when the next points are in a new window
        if (i + 1 != len(dataset)) and (dataset.iloc[i + 1]['time'] >= times[j + 1]):
        # Process the features of the current window
            total += len(dict['xx'])
            print(f"There are {len(dict['xx'])} data points")
            if len(dict['xx']) > 125:

                currentWindowFeatures = get_features(dict, label)
                if ~np.isnan(currentWindowFeatures).any():
                    featuresPerWindow.append(currentWindowFeatures)

            dict = {"time": [], "WindowStart": [], "xx": [], "xy": [], "xz": [], "az": [], "ax": [], "ay": [], "Azimuth": [], "Pitch": [], "Roll": []}
            j += 1
            currentTime = times[j]

    # normalize all of the features we processed
    normalizedFPW = preprocessing.normalize(np.array(featuresPerWindow))

    print(f"in the dataset: there are {normalizedFPW.shape[0]} instances")
    # return an array where each element contains information about a single window
    return normalizedFPW, total

