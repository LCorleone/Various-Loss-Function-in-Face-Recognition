import keras
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pylab as plt
import warnings
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Activation, add, Dense, Flatten, Dropout, Multiply, Embedding, Lambda
from keras.layers import Conv2D, MaxPooling2D, PReLU, Layer, AveragePooling2D
from keras import backend as K
from keras.optimizers import SGD, Adam
from sklearn.metrics import roc_auc_score
from keras.constraints import unit_norm
import os
from keras.callbacks import LearningRateScheduler
import transformations as tm


class Histories(keras.callbacks.Callback):
    def __init__(self, loss, monitor):
        self.loss = loss
        self.path = os.path.join('images', self.loss)
        self.monitor = monitor
        if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            self.monitor_op = np.less
            self.best = np.Inf
        return

    def on_train_begin(self, logs=None):
        return

    def on_train_end(self, logs=None):
        return

    def on_epoch_begin(self, epoch, logs=None):
        return

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Can save best model only with %s available, '
                          'skipping.' % (self.monitor), RuntimeWarning)
        else:
            if self.monitor_op(current, self.best):
                print('\nEpoch %05d: %s improved from %0.5f to %0.5f'
                      % (epoch + 1, self.monitor, self.best,
                         current))
                self.best = current
                self.model.save('best_model_with_' + self.loss + '.h5', overwrite=True)

        print('\n======================================')
        print('using loss type: {}'.format(self.loss))
        print(len(self.validation_data))  # be careful of the dimenstion of the self.validation_data, somehow some extra dim will be included
        print(self.validation_data[0].shape)
        print(self.validation_data[1].shape)
        print('========================================')
        # (IMPORTANT) Only use one input: "inputs=self.model.input[0]"
        nn_input = self.model.input  # this can be a list or a matrix.

        labels = self.validation_data[1].flatten()
        feature_layer_model = Model(nn_input, outputs=self.model.get_layer('feature_embedding').output)
        feature_embedding = feature_layer_model.predict(self.validation_data[0])
        # if self.loss == 'AM-softmax':
        #     feature_embedding = tm.unit_vector(feature_embedding, axis=1)
        visualize(feature_embedding, labels, epoch, self.loss, self.path)

        return

    def on_batch_begin(self, batch, logs=None):
        return

    def on_batch_end(self, batch, logs=None):
        return


def visualize(feat, labels, epoch, loss, path):

    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()
    for i in range(10):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    XMax = np.max(feat[:, 0])
    XMin = np.min(feat[:, 1])
    YMax = np.max(feat[:, 0])
    YMin = np.min(feat[:, 1])

    plt.xlim(xmin=XMin, xmax=XMax)
    plt.ylim(ymin=YMin, ymax=YMax)
    plt.text(XMin, YMax, "epoch=%d" % epoch)
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + '/' + loss + '_epoch=%d.jpg' % epoch)


class Dense_with_Center_loss(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Dense_with_Center_loss, self).__init__(**kwargs)

    def build(self, input_shape):
        # 添加可训练参数
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='glorot_normal',
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.output_dim,),
                                    initializer='zeros',
                                    trainable=True)
        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer='glorot_normal',
                                       trainable=True)

    def call(self, inputs):
        # different from the original paper, here center is trainable
        # the original operation is to update by the features of the minibatch
        self.inputs = inputs
        return K.dot(inputs, self.kernel) + self.bias

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def loss(self, y_true, y_pred, lamb=0.5):
        y_true = K.cast(y_true, 'int32')
        crossentropy = K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
        centers = K.gather(self.centers, y_true[:, 0])  # get the center
        center_loss = K.sum(K.square(centers - self.inputs), axis=1)  # center loss
        return crossentropy + lamb * center_loss

    def get_config(self):
        config = {'output_dim': self.output_dim}
        base_config = super(Dense_with_Center_loss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def sparse_amsoftmax_loss(y_true, y_pred, scale=24, margin=0.2):
    y_true = K.expand_dims(y_true[:, 0], 1)  # shape=(None, 1)
    y_true = K.cast(y_true, 'int32')  # dtype=int32
    batch_idxs = K.arange(0, K.shape(y_true)[0])
    batch_idxs = K.expand_dims(batch_idxs, 1)
    ordinal_y = K.concatenate([batch_idxs, y_true], 1)
    sel_logits = K.tf.gather_nd(y_pred, ordinal_y)
    t = K.tf.scatter_nd(ordinal_y, sel_logits * 0 + (-margin), K.tf.shape(y_pred))
    comb_logits_diff = K.tf.add(y_pred, t)
    return K.sparse_categorical_crossentropy(y_true, scale * comb_logits_diff, from_logits=True)


class Dense_with_Asoftmax_loss(Layer):

    def __init__(self, output_dim, m, **kwargs):
        self.output_dim = output_dim
        self.m = m
        super(Dense_with_Asoftmax_loss, self).__init__(**kwargs)

    def build(self, input_shape):
        # 添加可训练参数
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='glorot_normal',
                                      trainable=True)

    def call(self, inputs):
        self.inputs = inputs
        self.xw = K.dot(inputs, self.kernel)
        self.w_norm = K.tf.norm(self.kernel, axis=0) + 1e-8
        self.x_norm = K.tf.norm(inputs, axis=1) + 1e-8
        self.logits = self.xw / self.w_norm
        return self.logits

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def loss(self, y_true, y_pred):
        y_true = K.expand_dims(y_true[:, 0], 1)
        y_true = K.cast(y_true, 'int32')
        batch_idxs = K.arange(0, K.shape(y_true)[0])
        batch_idxs = K.expand_dims(batch_idxs, 1)
        ordinal_y = K.concatenate([batch_idxs, y_true], 1)
        sel_logits = K.tf.gather_nd(self.logits, ordinal_y)
        cos_th = K.tf.div(sel_logits, self.x_norm)
        if self.m == 1:
            return K.sparse_categorical_crossentropy(y_true, self.logits, from_logits=True)
        else:
            if self.m == 2:
                cos_sign = K.tf.sign(cos_th)
                res = 2 * K.tf.multiply(K.tf.sign(cos_th), K.tf.square(cos_th)) - 1
            elif self.m == 4:
                cos_th2 = K.tf.square(cos_th)
                cos_th4 = K.tf.pow(cos_th, 4)
                sign0 = K.tf.sign(cos_th)
                sign3 = K.tf.multiply(K.tf.sign(2 * cos_th2 - 1), sign0)
                sign4 = 2 * sign0 + sign3 - 3
                res = sign3 * (8 * cos_th4 - 8 * cos_th2 + 1) + sign4
            else:
                raise ValueError('unsupported value of m')
            scaled_logits = K.tf.multiply(res, self.x_norm)
            comb_logits_diff = K.tf.add(self.logits, K.tf.scatter_nd(ordinal_y, K.tf.subtract(scaled_logits, sel_logits), K.tf.to_int32(K.tf.shape(self.logits))))
            return K.sparse_categorical_crossentropy(y_true, comb_logits_diff, from_logits=True)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'm': self.m}
        base_config = super(Dense_with_Asoftmax_loss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Dense_with_AMsoftmax_loss(Layer):

    def __init__(self, output_dim, m, scale, **kwargs):
        self.output_dim = output_dim
        self.m = m
        self.scale = scale
        super(Dense_with_AMsoftmax_loss, self).__init__(**kwargs)

    def build(self, input_shape):
        # 添加可训练参数
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='glorot_normal',
                                      trainable=True)

    def call(self, inputs):
        self.inputs = inputs
        # self.xw = K.dot(inputs, self.kernel)
        # self.w_norm = K.tf.norm(self.kernel, axis=0) + 1e-8
        # self.x_norm = K.tf.norm(inputs, axis=1) + 1e-8
        # self.x_norm = K.expand_dims(self.x_norm, 1)
        # self.logits = self.xw / self.w_norm / self.x_norm
        self.w_norm = K.tf.nn.l2_normalize(self.kernel, 0, 1e-10)
        self.x_norm = K.tf.nn.l2_normalize(self.inputs, 1, 1e-10)
        self.logits = K.tf.matmul(self.x_norm, self.w_norm)

        return self.logits

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def loss(self, y_true, y_pred):
        y_true = K.expand_dims(y_true[:, 0], 1)
        y_true = K.cast(y_true, 'int32')
        batch_idxs = K.arange(0, K.shape(y_true)[0])
        batch_idxs = K.expand_dims(batch_idxs, 1)
        ordinal_y = K.concatenate([batch_idxs, y_true], 1)
        sel_logits = K.tf.gather_nd(self.logits, ordinal_y)
        comb_logits_diff = K.tf.add(self.logits, K.tf.scatter_nd(ordinal_y, sel_logits - self.m - sel_logits, K.tf.to_int32(K.tf.shape(self.logits))))
        return K.sparse_categorical_crossentropy(y_true, self.scale * comb_logits_diff, from_logits=True)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'm': self.m,
                  'scale': self.scale}
        base_config = super(Dense_with_AMsoftmax_loss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def build_net(nn_input_shape=(28, 28, 1), num_classes=10, loss='softmax'):
    nn_inputs = Input(shape=nn_input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(nn_inputs)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    # for visualization
    # x = Dense(2)(x)
    # feature_embedding = PReLU(name='feature_embedding')(x)
    feature_embedding = Dense(2, name='feature_embedding')(x)
    if loss == 'softmax':
        out = Dense(num_classes, activation='softmax')(feature_embedding)
        model = Model(inputs=nn_inputs, outputs=out)
        model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        return model
    elif loss == 'center-loss':
        center_logits = Dense_with_Center_loss(num_classes)
        out = center_logits(feature_embedding)
        model = Model(inputs=nn_inputs, outputs=out)
        model.compile(loss=center_logits.loss, optimizer='Adam', metrics=['sparse_categorical_crossentropy'])
        return model
    elif loss == 'A-softmax':
        # different from am-softmax is no norm feature
        A_softmax_logits = Dense_with_Asoftmax_loss(num_classes, m=4)
        out = A_softmax_logits(feature_embedding)
        model = Model(inputs=nn_inputs, outputs=out)
        model.compile(loss=A_softmax_logits.loss, optimizer='Adam', metrics=['sparse_categorical_crossentropy'])
        return model
    elif loss == 'AM-softmax':
        AM_softmax_logits = Dense_with_AMsoftmax_loss(num_classes, m=0.2, scale=24)
        out = AM_softmax_logits(feature_embedding)
        model = Model(inputs=nn_inputs, outputs=out)
        model.compile(loss=AM_softmax_logits.loss, optimizer='Adam', metrics=['sparse_categorical_crossentropy'])
        return model
