import tensorflow as tf
import os
from natsort import natsorted
from glob import glob
from utils import load_data, load_batch, view_results
from tree_model import TreeED, train_step, test_step
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '1'

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
	tree_ED_ckpt.restore(tree_ED_man.latest_checkpoint)
	EPOCHS      = 5000
	START       = int(tree_ED_ckpt.step) // len(train_batch) + 1
	save_freq   = 500
	tvis_freq   = 500
	vvis_freq   = 120
	if tree_ED_man.latest_checkpoint:
		print('Restored from last checkpoint, epoch : {0}'.format(START))

	for epoch in range(START, EPOCHS):
		train_loss = tf.keras.metrics.Mean()
		test_loss  = tf.keras.metrics.Mean()

		for idx, path in enumerate(tqdm(train_batch), start=1):
			X, Y = load_data(path)
			loss = train_step(tree_ED, tree_ED_opt, X, Y, N)
			# tree_ED.summary()
			train_loss.update_state(loss)
			tree_ED_ckpt.step.assign_add(1)
			if (idx%save_freq) == 0:
				tree_ED_man.save()
			if (idx%tvis_freq) == 0:
				view_results(tree_ED, X, Y, batch_size, 'train', int(tree_ED_ckpt.step))
			print('Train_Loss: {0}'.format(loss))


		for idx, path in enumerate(tqdm(val_batch), start=1):
			X, Y = load_data(path)
			loss = test_step(tree_ED, X, Y, N)
			test_loss.update_state(loss)
			if (idx%vvis_freq) == 0:
				view_results(tree_ED, X, Y, batch_size, 'test', int(tree_ED_ckpt.step)+idx)
			print('Test_Loss: {0}'.format(loss))

		tree_ED_eman.save()
		with open('log.txt', 'a') as file:
			file.write('Epoch: {0}\tTrain_Loss: {1}\t Test_Loss: {2}\n'.format(epoch, train_loss.result(), test_loss.result()))

		print('Epoch: {0}\tTrain_Loss: {1}\t Test_Loss: {2}'.format(epoch, train_loss.result(), test_loss.result()))
