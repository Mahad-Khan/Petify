from blue import app
from models import global_model,global_pre_trained_inception_model,global_pre_trained_xception_model
import tensorflow as tf
import threading


global_model()
global_pre_trained_xception_model()
global_pre_trained_inception_model()

if __name__ == "__main__":

    #tf.keras.backend.clear_session()
    app.run(host="0.0.0.0",port=5000)
    # Repeat()