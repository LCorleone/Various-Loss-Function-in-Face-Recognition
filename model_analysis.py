# model analysis
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.models import load_model
import keras
from keras.layers import Conv2D, MaxPooling2D, PReLU, Layer, AveragePooling2D
from keras.datasets import mnist
from keras import backend as K
from keras.callbacks import LearningRateScheduler
import sys
sys.path.append('./utils')
from utils_func import build_net, Histories, Dense_with_Center_loss


# set GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)

# loss_name = 'AM-softmax'
# model = build_net(loss=loss_name)
# model.summary()
model_path = 'best_model_with_center-loss.h5'
model = load_model(model_path, custom_objects={'Dense_with_Center_loss': Dense_with_Center_loss})
model.summary()