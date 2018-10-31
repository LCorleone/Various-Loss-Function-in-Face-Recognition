from keras import backend as K
import tensorflow as tf
import transformations as tm
import numpy as np


# y_true = tf.constant([[0], [2], [2]])
# y_pred = tf.constant([[1, 2, 0], [1, 1, 1], [2, 2, 2]])
# batch_idxs = K.arange(0, K.shape(y_true)[0])
# batch_idxs = K.expand_dims(batch_idxs, 1)
# # idxs = K.concatenate([batch_idxs, y_true], 1)
# idxs = tf.constant([[0, 0], [1, 0], [1,1]])
# updates = K.ones((3)) * (-0.2)
# t = K.tf.scatter_nd(idxs, updates, tf.constant([3, 3]))

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     sess.run(tf.local_variables_initializer())
#     y_true, y_pred, idxs, updates, t = sess.run([y_true, y_pred, idxs, updates,t ])
#     print(y_true)
#     print(y_pred)
#     print(idxs)
#     print(updates)
#     print(t)


# indices = tf.constant([[0], [2]])
# updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
#                         [7, 7, 7, 7], [8, 8, 8, 8]],
#                        [[5, 5, 5, 5], [6, 6, 6, 6],
#                         [7, 7, 7, 7], [8, 8, 8, 8]]])
# shape = tf.constant([4, 4, 4])
# scatter = tf.scatter_nd(indices, updates, shape)


# in_shape = K.shape(indices)
# up_shape = K.shape(updates)
# shape_shape = K.shape(shape)

# with tf.Session() as sess:
#     print(sess.run([indices, updates, scatter, in_shape, up_shape, shape_shape]))

y_true = tf.constant([1,2,3,4,5])
y_true = K.cast(y_true, 'int32')
# y_norm = y_true
y_norm = K.l2_normalize(y_true)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    y_true, y_norm = sess.run([y_true, y_norm])
    print(y_true)
    print(y_norm)
