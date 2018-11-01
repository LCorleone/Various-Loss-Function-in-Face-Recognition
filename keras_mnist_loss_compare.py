import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import keras
from keras.datasets import mnist
from keras import backend as K
from keras.callbacks import LearningRateScheduler
import sys
sys.path.append('./utils')
from utils_func import build_net, Histories, callback_annealing

# set GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)



# set training params
batch_size = 128
num_classes = 10
epochs = 50

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# adjust the channels and input shape
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# normalize the data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0

# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

print('x_train shape: {}'.format(x_train.shape))
print('train samples: {}'.format(x_train.shape[0]))
print('test samples: {}'.format(x_test.shape[0]))
print(y_train.shape)


def step_decay(epoch):
    if epoch % 5 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * .5)
        print("lr changed to {}".format(lr * .5))
    return K.get_value(model.optimizer.lr)


loss_name = 'A-softmax'
if loss_name == 'A-softmax':
    model, annealing_lambda = build_net(loss=loss_name)
    model.summary()
    histories = Histories(loss=loss_name, monitor='val_loss')
    lrate = LearningRateScheduler(step_decay)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test), callbacks=[histories, callback_annealing(annealing_lambda)])
else:
    model = build_net(loss=loss_name)
    model.summary()
    # when not using softmax loss, the acc returned by keras is inaccurate, so using val_loss
    histories = Histories(loss=loss_name, monitor='val_loss')
    lrate = LearningRateScheduler(step_decay)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test), callbacks=[histories])
