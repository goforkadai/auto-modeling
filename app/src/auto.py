import sys
import tensorflow as tf
from tensorflow import keras
import autokeras as ak
import pandas as pd
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

dataset_path = sys.argv[1]
model_path = sys.argv[2]
graph_path = sys.argv[3]
#dataset_path = '../../data_directory/tmp/tmp-dataset/tmp.csv'
df_dataset = pd.read_csv(dataset_path, index_col=0)


df_train = df_dataset.sample(frac=0.8,random_state=0)
df_test = df_dataset.drop(df_train.index)

df_train_x = df_train.filter(like='x')
df_train_y = df_train.drop(columns=df_train_x.columns)
df_test_x = df_test.filter(like='x')
df_test_y = df_test.drop(columns=df_test_x.columns)


clf = ak.StructuredDataRegressor(max_trials=100)

callbacks = [tf.keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True)]
clf.fit(df_train_x,df_train_y,callbacks = callbacks)
model = clf.export_model()
model.save(model_path + 'my_model', save_format='tf')
#plot_model(model, to_file='../../data_directory/pictures/my_model.png', show_layer_names=True, show_shapes=True)


results = clf.predict(df_test_x)
df_results = pd.DataFrame(results)
df_results.columns = ['predicted_y_' + str(num) for num in range(len(df_results.columns))]
df_output = pd.concat([df_test_y.reset_index(),df_results.reset_index(drop=True)], axis=1)
#print(df_output)
df_output.to_csv(graph_path + 'graph.csv', index=False)

#df_output = pd.DataFrame(df_test_x,results)
#print(df_output)

#test_predictions = model.predict(df_test_x).flatten()
#plt.scatter(df_test_y, test_predictions)
#plt.xlabel('True Values')
#plt.ylabel('Predictions')
#plt.axis('equal')
#plt.axis('square')
#plt.xlim([0,plt.xlim()[1]])
#plt.ylim([0,plt.ylim()[1]])
#_ = plt.plot()
#plt.savefig('../../data_directory/pictures/img.png')
