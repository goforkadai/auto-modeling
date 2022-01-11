import os
import sys

import autokeras as ak
import tensorflow as tf
from tensorflow import keras
import pandas as pd

model_path = sys.argv[1]
model = tf.keras.models.load_model(model_path,custom_objects=ak.CUSTOM_OBJECTS)

model.summary()
for i, layer in enumerate (model.layers):
    print (i, layer)
    try:
        print ("    ",layer.activation)
    except AttributeError:
        print('   no activation attribute')


#
#df_dataset = pd.read_csv(dataset_path, index_col=0)
#df_dataset = df_dataset.sample(frac=0.001,random_state=0)
#df_input = df_dataset.filter(like='x_')
#df_target = df_dataset.filter(like='y_')
#columns_input = df_input.columns
#columns_target = df_target.columns
#
#df_predict = pd.DataFrame(model.predict(df_input.values).flatten())
#
#df_target = pd.DataFrame(df_target.values.flatten())
#df_compare = pd.concat([df_target,df_predict],axis=1)
#df_compare.columns = ['target', 'predict']
#df_compare.to_csv(output_path+'compare.csv')
