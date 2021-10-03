import tensorflow as tf
import os
from natsort import natsorted
from glob import glob
from utils import load_data, load_batch, view_results
from tree_model import TreeED, train_step, test_step
from tqdm import tqdm
import h5py
from losses import chamfer_loss
import numpy as np

os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '2'

def test_unit(model, model_opt, X, Y):
	print('Train test unit ....')
	print(train_step(tree_net, tree_opt, X, Y, N))
	print('Test  test unit ....')
	print(test_step(tree_net, X, Y, N))

if __name__ == '__main__':
	batch_size   = 16
	latent_dim   = 512
	K            = 10 # Support
	N            = 2048
	train_path   = natsorted(glob('new_ShapeNetCorev2/*/train/*.npy'))
	val_path     = natsorted(glob('new_ShapeNetCorev2/*/val/*.npy'))
	test_path    = natsorted(glob('new_ShapeNetCorev2/*/test/*.npy'))
	train_batch  = load_batch(train_path, batch_size=batch_size)
	val_batch    = load_batch(val_path, batch_size=batch_size)
	# test_batch    = load_batch(test_path, batch_size=batch_size)
	# # X_train, Y_train = load_data(next(iter(train_batch)))
	# # print(X_train.shape, Y_train.shape)
	# # X_test, Y_test  = load_data(next(iter(val_batch)))
	# # print(X_test.shape, Y_test.shape)
	tree_ED      = TreeED(N, K, latent_dim, batch_size)
	# feat, tree   = tree_ED([X_train], training=True)
	# print(feat.shape, tree.shape)
	tree_ED_opt      = tf.keras.optimizers.Adam(lr=1e-4)
	# # test_unit(tree_net, tree_opt, X_train, Y_train)
	# # test_unit(tree_net, tree_opt, X_test, Y_test)
	tree_ED_ckpt     = tf.train.Checkpoint(step=tf.Variable(1), model=tree_ED, gopt=tree_ED_opt)
	tree_ED_man      = tf.train.CheckpointManager(tree_ED_ckpt, directory='treeED_ckpt', max_to_keep=10)
	tree_ED_eman     = tf.train.CheckpointManager(tree_ED_ckpt, directory='treeED_eckpt', max_to_keep=10)
	tree_ED_ckpt.restore(tree_ED_man.latest_checkpoint).expect_partial()
	EPOCHS      = 5000
	START       = 1#int(tree_ED_ckpt.step) // len(train_batch) + 1
	save_freq   = 500
	tvis_freq   = 500
	vvis_freq   = 120
	if tree_ED_man.latest_checkpoint:
		print('Restored from last checkpoint, epoch : {0}'.format(START))

	if not os.path.isdir('test_results'):
		os.makedirs('test_results')

	for idx, path in enumerate(tqdm(test_path), start=1):
		object_class = path.split(os.path.sep)[1]
		file_name = os.path.splitext(os.path.split(path)[-1])[0]

		with open('test_results/{}.txt'.format(object_class), 'a') as file:
			X = tf.expand_dims(tf.convert_to_tensor(np.load(path, allow_pickle=True), dtype=tf.float32), axis=0)
			tree  = [X]
			_, Y_cap = tree_ED(tree, training=False)
			# Y_cap = Y_cap[0]
			loss  = chamfer_loss(X, Y_cap, N)
			file.write('{}:{}\n'.format(file_name, loss))

		# if not os.path.isdir('benchmark/all/'):
		# 	os.makedirs('benchmark/all/')
		# with h5py.File('benchmark/all/{}'.format(os.path.basename(path)), 'w') as file:
		# 	file.create_dataset('data', data=Y_cap)