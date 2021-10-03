import tensorflow as tf

def chamfer_distance(S1, S2, N):
	S1 = tf.tile(tf.expand_dims(S1, axis=-2), [1, 1, N, 1])
	S2 = tf.tile(tf.expand_dims(S2, axis=1),  [1, N, 1, 1])
	return tf.reduce_sum(tf.reduce_min(tf.reduce_sum(tf.square((tf.subtract(S1, S2))), axis=-1), axis=-1))

def chamfer_loss(Y, Y_cap, N):
	xy_dis = chamfer_distance(Y, Y_cap, N)
	yx_dis = chamfer_distance(Y_cap, Y, N)
	return (xy_dis + yx_dis)

# def chamfer_distance(S1, S2, N1, N2):
# 	S1 = tf.tile(tf.expand_dims(S1, axis=-2), [1, 1, N2, 1])
# 	S2 = tf.tile(tf.expand_dims(S2, axis=1),  [1, N1, 1, 1])
# 	return tf.reduce_sum(tf.reduce_min(tf.reduce_sum(tf.square((tf.subtract(S1, S2))), axis=-1), axis=-1))

# def chamfer_loss(Y, Y_cap, N1, N2):
# 	xy_dis = chamfer_distance(Y, Y_cap, N1, N2)
# 	yx_dis = chamfer_distance(Y_cap, Y, N2, N1)
# 	return (xy_dis + yx_dis)