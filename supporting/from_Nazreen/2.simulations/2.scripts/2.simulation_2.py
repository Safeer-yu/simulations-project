#simulation 2 id+pos+allog -> orth
#SIMULATION 2- RUN THIS SCRIPT
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from tensorflow import keras
from tensorflow.keras import Model
from keras.layers import concatenate, Dense
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import load_model
from tensorflow.keras import regularizers
from keras import Input

data = pd.read_excel('Targets2.xlsx',index_col=0,sheet_name="id+pos+allog")
df = pd.DataFrame(data)  
dataset = df.values
X = dataset[0:477, 0:1150]
X= np.asarray(X).astype(np.float32)
inputA = Input(shape=(1150,))

data = pd.read_excel('orthograph_sim1_2.xlsx')
df = pd.DataFrame(data)  
dataset = df.values
Y = dataset[0:477,0:350]
Y = np.asarray(Y).astype(np.float32)
Output = Input(shape=(350,))
x = Dense(1000, activation="sigmoid")(inputA)
y1 = Dense(500, activation="sigmoid")(x)
z = Dense(350, activation="sigmoid")(y1)

model = Model(inputA, outputs=z)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
model.compile(optimizer="adam", loss=["mse"])
model.fit(X,Y, batch_size=25, epochs=2000, callbacks=[callback])

#simulation 2
f1 = np.array(Y).flatten().tolist()
data = pd.read_excel('TL-Allogs2.xlsx',index_col=0, sheet_name="id+pos+allog")
df = pd.DataFrame(data)
dataset = df.values
X = dataset[0:477, 0:1150]
X= np.asarray(X).astype(np.float32)
pred = model.predict(X)
f2 = np.array(pred).flatten().tolist()
valz = []
for i in range(477):
  r = np.corrcoef(pred[i], Y[i])
  valz.append(round(1-r[0][1],3))
df_new = pd.DataFrame()
df_new['target-TL-allog']= valz

data = pd.read_excel('TL+Allogs2.xlsx', index_col=0,sheet_name="id+pos+allog")
df = pd.DataFrame(data)
dataset = df.values
X = dataset[0:477, 0:1150]
X= np.asarray(X).astype(np.float32)
pred = model.predict(X)
f3 = np.array(pred).flatten().tolist()
valz = []
for i in range(477):
  r = np.corrcoef(pred[i], Y[i])
  valz.append(round(1-r[0][1],3))
df_new['target-TL+allog']= valz

data = pd.read_excel('SubL-Allogs2.xlsx', index_col=0,sheet_name="id+pos+allog")
df = pd.DataFrame(data)
dataset = df.values
X = dataset[0:477, 0:1150]
X= np.asarray(X).astype(np.float32)
pred = model.predict(X)
f4 = np.array(pred).flatten().tolist() #sublnoallog
valz = []
for i in range(477):
  r = np.corrcoef(pred[i], Y[i])
  valz.append(round(1-r[0][1],3))
df_new['target-SubL-allog']= valz

data = pd.read_excel('SubL+Allogs2.xlsx', index_col=0,sheet_name="id+pos+allog")
df = pd.DataFrame(data)
dataset = df.values
X = dataset[0:477, 0:1150]
X= np.asarray(X).astype(np.float32)
pred = model.predict(X)
f5 = np.array(pred).flatten().tolist() #sublallog
valz = []
for i in range(477):
  r = np.corrcoef(pred[i], Y[i])
  valz.append(round(1-r[0][1],3))
df_new['target-SubL+allog']= valz

r = np.corrcoef(f1,f5)
print("correlation distance between identity and SublAllog\n", 1-r)
r = np.corrcoef(f1, f4)
print("correlation distance between identity and SublnoAllog\n", 1-r)
r = np.corrcoef(f1, f3)
print("correlation distance between identity and TLAllog\n", 1-r)
r = np.corrcoef(f1, f2)
print("correlation distance between identity and TLnoAllog\n", 1-r)
r = np.corrcoef(f3, f5)
print("CORRELATION distance BETWEEN TL+Allog and SUBL+Allog is",1-r)  
r = np.corrcoef(f2, f4)
print("CORRELATION distance BETWEEN TL-Allog and SUBL-Allog is",1-r) 
df_new.to_excel('Ttest_data_s2.xlsx',encoding = 'utf-8-sig')

#output
correlation distance between identity and SublAllog
 [[0.         0.21388306]
 [0.21388306 0.        ]]
correlation distance between identity and SublnoAllog
 [[0.       0.239424]
 [0.239424 0.      ]]
correlation distance between identity and TLAllog
 [[0.         0.24177913]
 [0.24177913 0.        ]]
correlation distance between identity and TLnoAllog
 [[0.         0.25724328]
 [0.25724328 0.        ]]
CORRELATION distance BETWEEN TL+Allog and SUBL+Allog is [[0.         0.14562588]
 [0.14562588 0.        ]]
CORRELATION distance BETWEEN TL-Allog and SUBL-Allog is [[0.         0.18164503]
 [0.18164503 0.        ]]