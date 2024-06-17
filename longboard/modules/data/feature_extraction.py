import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.fft import fft, rfft
from scipy.fft import fftfreq, rfftfreq

from sklearn import preprocessing
from statistics import mean

from utils import get_times

# input: arrays of displacement vectors in 3 dimensions after 2 seconds
# get angle of the vector from each dimension, trigonometry
def normalize_vector(xx, xy, xz):
	lx, ly, lz = xx[-1] - xx[0], xy[-1] - xy[0], xz[-1] - xz[0]

	hypo = (lx ** 2 + ly ** 2 + lz ** 2) ** 0.5
	side1 = (lx ** 2 + ly ** 2) ** 0.5
	side2 = ly

	newPitch = ((np.arctan(side1 / hypo) * 180) / np.pi)
	newAzimuth = ((np.arccos(side2 / hypo) * 180) / np.pi)

	return newPitch, newAzimuth

# Method to Verlet integrate the raw acceleration data
def get_displacement_data(df, label):
	transformed_acceleration = []

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
def rotate_device(df):
	rotated_pos = []
	for index in range(len(df)):
		azimuth, pitch, roll = df['Azimuth'].iat[index], df['Pitch'].iat[index], df['Roll'].iat[index]
		pos_matrix = (df['ax'].iat[index], df['ay'].iat[index], df['az'].iat[index])
		rotated_pos.append(rotation3D(pos_matrix, -pitch, -roll, -azimuth))

	df['rotated_ax'] = [x[0] for x in rotated_pos]
	df['rotated_ay'] = [x[1] for x in rotated_pos]
	df['rotated_az'] = [x[2] for x in rotated_pos]

	# df['ax'] = [x[0] for x in rotated_pos]
	# df['ay'] = [x[1] for x in rotated_pos]
	# df['az'] = [x[2] for x in rotated_pos]

	return df

def get_features(dict):
	# rotate the position data so that the y-axis is in line with the direction of motion
	newRoll, newAzimuth = normalize_vector(dict['xx'], dict['xy'], dict['xz'])
	b = 1
	for i in range(len(dict['xx'])):
		pos_matrix = (dict['xx'][i], dict['xy'][i], dict['xz'][i])
		last = rotation3D(pos_matrix, -newRoll, 0, -newAzimuth)
		dict['xx'][i] = last[0]
		dict['xy'][i] = last[1]
		dict['xz'][i] = last[2]

		acc_matrix = (dict['ax'][i], dict['ay'][i], dict['az'][i])
		last = rotation3D(acc_matrix, -newRoll, 0, -newAzimuth)
		dict['ax'][i] = last[0]
		dict['ay'][i] = last[1]
		dict['az'][i] = last[2]

	features = []

	# y dimension -> acceleration, x, z dimensions -> position displacements
	a = np.array(dict["ay"])
	x1 = np.array(dict["xx"])
	x2 = np.array(dict["xz"])
	x3 = np.array(dict["xy"])

	'''
	Returns
		[cA_n, cD_n, cD_n-1, …, cD2, cD1] -> list
		Ordered list of coefficients arrays where n denotes the level of decomposition. The first element (cA_n) of the result is approximation coefficients array and the following elements (cD_n - cD_1) are details coefficients arrays.
	'''

	# list_coeff_xx = pywt.wavedec(x1, family)
	# list_coeff_xz = pywt.wavedec(x2, family)

	# freq_x, freq_z = get_freq(pos, list_coeff_xx, list_coeff_xz)

	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection = '3d')
	# ax.set_xlabel('x axis')
	# ax.set_ylabel('y axis')
	# ax.set_zlabel('z axis')
	# ax.set_xlim([x1.min(), x1.min() + 0.025])
	# ax.set_zlim([x2.min(), x2.min() + 0.025])
	# ax.set_ylim([x3.min(), x3.min() + 0.025])
	# plot = ax.scatter(x1, x3, x2, color = 'green')
	# plt.show()

	# # ground level view
	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection = '3d')
	# ax.view_init(elev=0, azim = 0)
	# ax.set_xlabel('x axis')
	# ax.set_ylabel('y axis')
	# ax.set_zlabel('z axis')
	# ax.set_xlim([x1.min(), x1.min() + 0.025])
	# ax.set_zlim([x2.min(), x2.min() + 0.025])
	# ax.set_ylim([x3.min(), x3.min() + 0.025])

	# plot = ax.scatter(x1, x3, x2, color = 'green')
	# plt.show()

	# Basically get the fourier transform plot of the position displacement data
	yf1 = rfft(np.array([x - x1.mean() for x in x1]))
	yf2 = rfft(np.array([x - x2.mean() for x in x2]))

	N = len(x1)

	# N = # of points 2 = # of secs per window/ window duration
	samplingRate = N / 3

	freqx1 = rfftfreq(N, 1 / samplingRate)
	freqx2 = rfftfreq(N, 1 / samplingRate)

	peakx = np.max(yf1)
	peakz = np.max(yf2)

	maxfreqx = freqx1[np.argmax(np.abs(yf1))]
	maxfreqz = freqx2[np.argmax(np.abs(yf2))]

	# if action == "pumping" and maxfreqx >= 2 or action != "pumping" and maxfreqx < 1:
	# 	plt.xticks(np.arange(0, 15, step=2.5))
	# 	plt.plot(freqx1[:N//10], np.abs(yf1[:N//10]))
	# 	plt.title('Fast Fourier Transform Displacement along x-axis')
	# 	plt.suptitle(f'Longboard action: {action}', fontsize=14, fontweight='bold')  # Main title
	# 	plt.xlabel('Frequency[Hz]')
	# 	plt.ylabel('Amplitude')
	# 	plt.axvline(x=maxfreqx, color='r', linestyle='--', alpha=0.4)  # Draw vertical line at peak
	# 	plt.text(maxfreqx, peakx, f'{maxfreqx} Hz', horizontalalignment='left', verticalalignment='bottom')
	# 	plt.grid()
	# 	plt.show()
	# 	b -= 1

	# plt.plot(freqx2[:30], np.abs(yf1[:30]))
	# plt.title('Fast Fourier Transform Displacement along z-axis')
	# plt.xlabel('Frequency[Hz]')
	# plt.ylabel('Amplitude')
	# plt.show()


	# # This is the difference between the maximum and minimum accelerations in the y dimension
	features.append(a.max()-a.min())

  # x
	features.append((x1.max() - x1.min()) / 2)
	# features.append(freq_x)
	features.append(maxfreqx)

  # z
	features.append((x2.max() - x2.min()) / 2)
	features.append(maxfreqz)

	features += get_features(a) + get_features(x1) + get_features(x2)

	return np.array(features)

