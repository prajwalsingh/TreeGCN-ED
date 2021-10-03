import tensorflow as tf
from tensorflow.keras import Model, layers
from losses import chamfer_loss

class TreeDecoder(Model):
	def __init__(self, K, upsample, last_layer, batch_size, depth, curr_node, degree, in_feat, out_feat):
		super(TreeDecoder, self).__init__()
		self.in_feat    = in_feat
		self.out_feat   = out_feat
		self.curr_node  = curr_node
		self.degree     = degree
		self.depth      = depth
		self.batch_size = batch_size
		self.last_layer = last_layer
		self.upsample   = upsample
		self.F_K  = [layers.Dense(units=K*self.in_feat), layers.Dense(units=self.out_feat)] # Loop Term
		self.U    = [layers.Dense(units=self.out_feat) for i in range(self.depth+1)] # Accumulate information from Ancestor
		self.W_up = tf.Variable(tf.initializers.GlorotUniform()(shape=[self.curr_node, self.in_feat, self.degree*self.in_feat]), name='WeightMat') # Xavier Initalization
		if not self.last_layer:
			self.act  = layers.LeakyReLU(alpha=0.2)
			self.b    = tf.Variable(tf.initializers.GlorotUniform()(shape=[1, self.degree*self.curr_node, self.out_feat]), name='Bias') # Xavier Initalization, for next level

	def call(self, tree):
		Anc_info = 0
		# Gathering information from ancestor
		for depth in range(self.depth+1):
			anc_node  = tree[depth].shape[1]
			rep_node  = self.curr_node // anc_node
			Q         = self.U[depth](tree[depth])
			Anc_info  = Anc_info + tf.reshape(tf.tile(Q, [1, 1, rep_node]), [-1, self.curr_node, self.out_feat])

		#  Upsampling the nodes
		if self.upsample:
			next_level = tf.expand_dims(tree[-1], axis=2) @ self.W_up
			next_level = tf.reshape(next_level, [-1, self.curr_node * self.degree, self.in_feat])
			next_level = self.F_K[1]( self.F_K[0]( next_level ) )
			next_level = next_level + tf.reshape(tf.tile(Anc_info, [1, 1, self.degree]), [-1, self.curr_node*self.degree, self.out_feat])
		else:
			next_level = self.F_K[1]( self.F_K[0]( tree[-1] ) )
			next_level = next_level + Anc_info
		# Adding bias and passing through non linearity function
		if not self.last_layer:
			next_level = self.act(next_level + self.b)

		tree.append(next_level)
		return tree

class TreeEncoder(Model):
	def __init__(self, K, downsample, last_layer, batch_size, depth, curr_node, in_feat, out_feat):
		super(TreeEncoder, self).__init__()
		self.in_feat    = in_feat
		self.out_feat   = out_feat
		self.curr_node  = curr_node
		self.depth      = depth
		self.batch_size = batch_size
		self.last_layer = last_layer
		self.downsample = downsample
		self.F_K    = [layers.Dense(units=K*self.in_feat), layers.Dense(units=self.out_feat)] # Loop Term
		self.U      = [layers.Dense(units=self.out_feat) for i in range(self.depth+1)] # Accumulate information from Ancestor
		self.W_down = tf.Variable(tf.initializers.GlorotUniform()(shape=[self.curr_node, self.in_feat, self.in_feat]), name='WeightMat') # Xavier Initalization
		if not self.last_layer:
			self.act  = layers.LeakyReLU(alpha=0.2)
			self.b    = tf.Variable(tf.initializers.GlorotUniform()(shape=[1, self.out_feat]), name='Bias') # Xavier Initalization, for next level

	def call(self, tree):
		Anc_info = None
		# Gathering information from ancestor
		for depth in range(self.depth+1):
			anc_node   = tree[depth].shape[1]
			red_node   = anc_node // self.curr_node # reduce node
			Q          = self.U[depth](tree[depth])
			gath_feat  = None
			start_idx  = 0
			group_size = anc_node // red_node # Total number of item in one group
			stop_idx   = group_size
			for _ in range(red_node):
				#gath_feat  = gath_feat + Q[:, start_idx:stop_idx]
				if gath_feat is not None:
					gath_feat  = tf.maximum(gath_feat, Q[:, start_idx:stop_idx])
				else:
					gath_feat  = Q[:, start_idx:stop_idx]
				start_idx  = stop_idx
				stop_idx   = stop_idx  + group_size
			
			if Anc_info is not None:
					Anc_info  = tf.maximum(Anc_info, gath_feat)
			else:
					Anc_info  = gath_feat

		#  Upsampling the nodes
		if self.downsample>0:
			N          = self.curr_node
			next_level = tf.expand_dims(tree[-1], axis=2) @ self.W_down
			next_level = tf.reshape(next_level, [-1, N, self.in_feat])
			if self.downsample == 2:
				upper_half, lower_half = next_level[:, :N//2], next_level[:, (N//2):]
				#next_level = upper_half + lower_half
				next_level  = tf.maximum(upper_half, lower_half)
			else:
				#next_level = tf.reduce_sum(next_level, axis=1, keepdims=True)
				next_level = tf.reduce_max(next_level, axis=1, keepdims=True)
				
			next_level = self.F_K[1]( self.F_K[0]( next_level ) )
			# next_level = next_level + tf.reshape(tf.tile(Anc_info, [1, 1, self.degree]), [-1, self.curr_node*self.degree, self.out_feat])
		# else:
		# 	next_level = self.F_K[1]( self.F_K[0]( tree[-1] ) )
			red_node   = Anc_info.shape[1] // next_level.shape[1] # Total number groups
			gath_feat  = None
			start_idx  = 0
			group_size = Anc_info.shape[1] // red_node # Total number of item in one group
			stop_idx   = group_size
			for _ in range(red_node):
				#gath_feat  = gath_feat + Anc_info[:, start_idx:stop_idx]
				if gath_feat is not None:
					gath_feat  = tf.maximum(gath_feat, Anc_info[:, start_idx:stop_idx])
				else:
					gath_feat  = Anc_info[:, start_idx:stop_idx]
				start_idx  = stop_idx
				stop_idx   = stop_idx  + group_size
			
			next_level = tf.maximum(next_level, gath_feat)

		# Adding bias and passing through non linearity function
		if not self.last_layer:
			next_level = self.act(next_level + self.b)

		tree.append(next_level)
		return tree


class TreeED(Model):
	def __init__(self, N=2048, K=10, latent_dim=512, batch_size=16):
		super(TreeED, self).__init__()

		filters_enc         = [3, 32, 64, 128, 128, 256, 512]
		downsample_enc      = [2,  2,  2,   2,   2,  64]
		self.depth_enc      = len(filters_enc) - 1
		self.tree_layer_enc = []
		curr_nodes_enc      = N
		for layer_no in range(self.depth_enc):
			if layer_no == self.depth_enc-1:
				self.tree_layer_enc.append(TreeEncoder(K, downsample_enc[layer_no], True, batch_size, layer_no, curr_nodes_enc, filters_enc[layer_no], filters_enc[layer_no+1]))
			else:
				self.tree_layer_enc.append(TreeEncoder(K, downsample_enc[layer_no], False, batch_size, layer_no, curr_nodes_enc, filters_enc[layer_no], filters_enc[layer_no+1]))
			curr_nodes_enc = curr_nodes_enc // downsample_enc[layer_no]


		filters_dec         = [latent_dim, 512, 256, 256, 128, 128, 128, 3]
		degrees_dec         = [         1,   2,   2,   2,   2,   2,  64]
		self.depth_dec      = len(filters_dec) - 1
		self.tree_layer_dec = []
		curr_nodes_dec      = 1
		for layer_no in range(self.depth_dec):
			if layer_no == self.depth_dec-1:
				self.tree_layer_dec.append(TreeDecoder(K, True, True, batch_size, layer_no, curr_nodes_dec, degrees_dec[layer_no], filters_dec[layer_no], filters_dec[layer_no+1]))
			else:
				self.tree_layer_dec.append(TreeDecoder(K, True, False, batch_size, layer_no, curr_nodes_dec, degrees_dec[layer_no], filters_dec[layer_no], filters_dec[layer_no+1]))
			curr_nodes_dec = curr_nodes_dec * degrees_dec[layer_no]

	
	def call(self, tree):
		for layer_no in range(self.depth_enc):
			tree = self.tree_layer_enc[layer_no]( tree )

		enc_tree = tree[-1]
		enc_tree = tf.reshape(enc_tree, [enc_tree.shape[0], -1])
		tree     = [tree[-1]]

		for layer_no in range(self.depth_dec):
			tree = self.tree_layer_dec[layer_no]( tree )

		return enc_tree, tree[-1]

def train_step(treeED, opt, X, Y, N):
	with tf.GradientTape() as tape:
		X = [X]
		_, Y_cap = treeED(X, training=True)
		loss  = chamfer_loss(Y, Y_cap, N)
	variables = treeED.trainable_variables
	gradients = tape.gradient(loss, variables)
	opt.apply_gradients(zip(gradients, variables))
	return loss

def test_step(treeED, X, Y, N):
	X = [X]
	_, Y_cap = treeED(X, training=False)
	loss     = chamfer_loss(Y, Y_cap, N)
	return loss