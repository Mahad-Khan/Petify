import tensorflow as tf
from tensorflow.keras.models import load_model
tf.keras.backend.clear_session()
model=None
pretrained_inception_model=None
pretrained_xception_model=None
import os

path=os.path.abspath(__file__)
str_path=path.split("/")[0:-1]
str_path="/".join(str_path)
model_path=str_path+"/"+"model_92_84.h5"
pretrained_inception_model_path=str_path+"/"+"pretrained_inception.h5"
pretrained_xception_model_path=str_path+"/"+"pretrained_xception.h5"

def global_model():
    global model
    model=load_model(model_path,compile=False)
    #tf.keras.backend.clear_session()

def get_model():
    return model

def global_pre_trained_inception_model():
    global pretrained_inception_model
    pretrained_inception_model=load_model(pretrained_inception_model_path,compile=False)
    #tf.keras.backend.clear_session()

def get_inception_model():
    return pretrained_inception_model

def global_pre_trained_xception_model():
    global pretrained_xception_model
    pretrained_xception_model=load_model(pretrained_xception_model_path,compile=False)
    #tf.keras.backend.clear_session()

def get_xception_model():
    return pretrained_xception_model