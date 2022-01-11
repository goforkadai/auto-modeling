import os
import sys

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import autokeras as ak
print(tf.__version__)


model = tf.keras.models.load_model('../../data_directory/tmp/tmp-model_data/model',custom_objects=ak.CUSTOM_OBJECTS)

model.summary()

