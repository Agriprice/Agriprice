#!/usr/bin/env python
# coding: utf-8

# ### #Library dan beberapa package yang digunakan untuk membangun model prediksi harga beras

# In[1]:


import math
import pandas_datareader as web
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from tensorflow.keras import optimizers, Model

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping

# from google.colab import drive
# drive.mount('/gdrive')
# %cd /gdrive/MyDrive/'Colab Notebooks'/
# %ls


# ## #Input dataset harga beras IR dalam format CSV

# In[4]:


df = pd.read_csv("Dataset1PM.csv")


# In[5]:


df.info()


# ### #Melihat seluruh nilai dan isi dataset

# In[10]:


df.drop(df.columns[[2]], axis=1, inplace=True)
#df


# ### #Pengecekan nilai data yang hilang untuk seluruh atribut di dalam dataset

# In[11]:


df.isnull().sum()


# ### #Menghapus atribut komoditas dan provinsi karena tidak digunakan dalam pembentukan model

# In[12]:


df['Tanggal'] = df['Tanggal'].astype('datetime64')

#print(df.dtypes)


# ### #Melihat statistik deskriptif dataset harga komoditas cabai merah besar di pasar 

# In[13]:


df.describe()


# ### #Menampilkan plot visualisasi grafik data deret waktu harian harga beras di pasar tradisional

# In[14]:


# plt.figure(figsize=(15,8))
# sns.lineplot(data=df,x='Tanggal',y='IRMedPM')


# 
# ### #Normalisasi data menggunakan normalisasi min-max

# In[ ]:





# ### #Membagi data latih dan data uji dengan nilai persentase sebesar 70% dan 30%

# In[15]:


# split data
train_size = int(len(df) * 0.8) # Menentukan banyaknya data train yaitu sebesar 80% data
train = df[:train_size]
test =df[train_size:].reset_index(drop=True)


# In[16]:


scaler = MinMaxScaler()
scaler.fit(train[['IRMedPM']])

train['scaled'] = scaler.transform(train[['IRMedPM']])
test['scaled'] = scaler.transform(test[['IRMedPM']])


# In[17]:


# train.head()


# ### #Arsitektur jaringan LSTM dan percobaan beberapa parameter

# In[18]:


def sliding_window(data, window_size):
    sub_seq, next_values = [], []
    for i in range(len(data)-window_size):
        sub_seq.append(data[i:i+window_size])
        next_values.append(data[i+window_size])
    X = np.stack(sub_seq)
    y = np.array(next_values)
    return X,y


# In[196]:


window_size = 28

X_train, y_train = sliding_window(train[['scaled']].values, window_size)
X_test, y_test = sliding_window(test[['scaled']].values, window_size)


# In[157]:


# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)


# ### #Fit the first model.

# In[158]:


def create_model(LSTM_unit=75, dropout=0.2):
    # create model
    model = Sequential()
    model.add(LSTM(units=LSTM_unit, activation='tanh', input_shape=(window_size, 1)))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    # Compile model
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model


# In[159]:


# Early Stopping
es = EarlyStopping(monitor = 'val_loss', mode = "min", patience = 5, verbose = 0)

# create model
model = KerasRegressor(build_fn=create_model, epochs=50, validation_split=0.1, batch_size=16, callbacks=[es], verbose=0)

# define the grid search parameters
LSTM_unit = [75]
dropout=[0.05]
param_grid = dict(LSTM_unit=LSTM_unit, dropout=dropout)


# In[197]:


grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)


# In[198]:


grid_result = grid.fit(X_train, y_train)


# In[199]:


# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
# Mengambil model terbaik
best_model = grid_result.best_estimator_.model


# In[200]:


# best_model.summary()


# In[201]:


history = best_model.history


# In[202]:


# # grafik loss function MSE

# plt.plot(history.history['loss'], label='Training loss')
# plt.plot(history.history['val_loss'], label='Validation loss')
# plt.title('loss function MSE')
# plt.ylabel('MSE')
# plt.xlabel('Epoch')
# plt.legend()


# In[203]:


# # grafik metric MAE

# plt.plot(history.history['mae'], label='Training MAE')
# plt.plot(history.history['val_mae'], label='Validation MAE')
# plt.title('metric MAE')
# plt.ylabel('MAE')
# plt.xlabel('Epoch')
# plt.legend()


# In[204]:


# Prediksi data train
predict_train = scaler.inverse_transform(best_model.predict(X_train))
true_train = scaler.inverse_transform(y_train)

# Prediksi data test
predict_test = scaler.inverse_transform(best_model.predict(X_test))
true_test = scaler.inverse_transform(y_test)


# In[205]:


# Mean Absolute Error (MAE) data train
# mae_train = np.mean(np.abs(true_train-predict_train))
# print('MAE data train sebesar:', mae_train)

# Mean Absolute Error (MAE) test data
# mae_test = np.mean(np.abs(true_test-predict_test))
# print('MAE data test sebesar:', mae_test)


# In[206]:


# abs_error_train = np.abs(true_train-predict_train)
# sns.boxplot(y=abs_error_train)


# In[207]:


# abs_error_test = np.abs(true_test-predict_test)
# sns.boxplot(y=abs_error_test)


# In[208]:


# print('range data train', true_train.max()-true_train.min())
# print('range data test', true_test.max()-true_test.min())


# ### plot prediksi data train

# In[209]:


# train['predict'] = np.nan
# train['predict'][-len(predict_train):] = predict_train[:,0]

# plt.figure(figsize=(15,8))
# sns.lineplot(data=train, x='Tanggal', y='IRMedPM', label = 'train')
# sns.lineplot(data=train, x='Tanggal', y='predict', label = 'predict')


# ### plot prediksi data test

# In[210]:


# test['predict'] = np.nan
# test['predict'][-len(predict_test):] = predict_test[:,0]

# plt.figure(figsize=(15,8))
# sns.lineplot(data=test, x='Tanggal', y='IRMedPM', label = 'test')
# sns.lineplot(data=test, x='Tanggal', y='predict', label = 'predict')


# ### plot prediksi data test 1 bulan terakhir 

# In[211]:


# plt.figure(figsize=(15,8))
# sns.lineplot(data=test, x='Tanggal', y='IRMedPM', label = 'test')
# sns.lineplot(data=test, x='Tanggal', y='predict', label = 'predict')


# In[212]:


# forecasting data selanjutnya
y_test = scaler.transform(test[['IRMedPM']])
n_future = 31*4
future = [[y_test[-1,0]]]
X_new = y_test[-window_size:,0].tolist()

for i in range(n_future):
    y_future = best_model.predict(np.array([X_new]).reshape(1,window_size,1))
    future.append([y_future[0,0]])
    X_new = X_new[1:]
    X_new.append(y_future[0,0])

future = scaler.inverse_transform(np.array(future))
date_future = pd.date_range(start=test['Tanggal'].values[-1], periods=n_future+1, freq='D')


# In[213]:


# Plot Data sebulan terakhir dan 4 bulan ke depan
plt.figure(figsize=(20,8))
sns.lineplot(data=test[-365:], x='Tanggal', y='IRMedPM', label = 'Aktual')
sns.lineplot(data=test[-365:], x='Tanggal', y='predict', label = 'Model')
sns.lineplot(x=date_future, y=future[:,0], label = 'Predict',)
plt.ylabel('IRMedPM');

