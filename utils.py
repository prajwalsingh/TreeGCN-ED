from glob import glob
from natsort import natsorted
import os
import numpy as np
import tensorflow as tf
import functools
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import h5py
import math
import random

batch_size = 16

def aug_pc(pointcloud):
        assert len(pointcloud.shape)==2

        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),    0],
                               [ math.sin(theta),  math.cos(theta),    0],
                               [0,                             0,      1]])
        
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return  rot_pointcloud

def fetch_data(path):
	# print(path.numpy().decode('utf-8'))
	pc = np.load(path.numpy().decode('utf-8'), allow_pickle=True)
	pc = aug_pc(pc)
	# with h5py.File(path.numpy().decode('utf-8'), 'r') as file:
	# 	pc    = file['pc'][()]
	# 	feat  = file['feat'][()]
	# print(path)
	# print(pc.shape)
	return pc, pc

def load_batch(path, batch_size=16):
	return tf.data.Dataset.from_tensor_slices((path)).shuffle(len(path)).batch(batch_size, drop_remainder=True)

def load_data(path):
	X, Y = list(zip(*map(fetch_data, path)))
	X    = tf.convert_to_tensor(X, dtype='float32')
	Y    = tf.convert_to_tensor(Y, dtype='float32')
	return X, Y

def view_data(X, Y, cat_dict, rev_cat_dict):
	if not os.path.isdir('save_fig'):
		os.makedirs('save_fig')

	for idx, (shape, cat) in enumerate(zip(X, Y)):
		shape = shape.numpy()
		cat   = cat.numpy()
		ax    = plt.axes(projection='3d')
		ax.scatter3D(shape[:, 0], shape[:, 1], shape[:, 2])
		plt.title('Category: {0}'.format(rev_cat_dict[cat]))
		plt.savefig('save_fig/{0}.png'.format(idx))

def view_results(tree_ED, X, Y, batch_size, categ, fileno):
	if not os.path.isdir('results'):
		os.makedirs('results/train/')
		os.makedirs('results/test/')
	idx   = np.random.randint(0, batch_size-1)
	tree  = [tf.expand_dims(X[idx], axis=0)]
	_, Y_cap = tree_ED(tree, training=False)
	Y_cap = Y_cap[0]
	Y     = Y[idx]
	ax    = plt.axes(projection='3d')
	ax.scatter3D(Y[:, 0], Y[:, 1], Y[:, 2])
	plt.savefig('results/{0}/{1}_gt.png'.format(categ, fileno))
	plt.clf()
	ax    = plt.axes(projection='3d')
	ax.scatter3D(Y_cap[:, 0], Y_cap[:, 1], Y_cap[:, 2])
	plt.savefig('results/{0}/{1}_pred.png'.format(categ, fileno))
