#NN ARCHITECTURE- simulation 3
#id,pos -> orth
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
from sklearn.preprocessing import MinMaxScaler

data = pd.read_excel('Target1.xlsx', index_col=0, sheet_name="identity")
df = pd.DataFrame(data)   
dataset = df.values
X = dataset[0:477, 0:35]
data = pd.read_excel('Target1.xlsx', index_col=0, sheet_name="position")
df1 = pd.DataFrame(data)
dataset = df1.values 
X1 = dataset[0:477, 0:100] 
inputA = Input(shape=(35,)) 
inputB = Input(shape=(100,))
data = pd.read_excel('orthograph_sim1_2.xlsx')
df = pd.DataFrame(data)  
dataset = df.values
Y = dataset[0:477,0:350]
Y = np.asarray(Y).astype(np.float32)
X= np.asarray(X).astype(np.float32)
X1 = np.asarray(X1).astype(np.float32)
Output = Input(shape=(350,))

x = Dense(700, activation="sigmoid")(inputA)
x = Model(inputs=inputA, outputs=x)
y = Dense(700, activation="sigmoid")(inputB)
y = Model(inputs=inputB, outputs=y)
combined = concatenate([x.output, y.output])
z = Dense(500, activation="sigmoid")(combined)
z1 = Dense(350, activation="sigmoid")(z)

model = Model(inputs=[x.input, y.input], outputs=z1)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
model.compile(optimizer="adam", loss=["mse"])
model.fit([X, X1],Y,batch_size=25, epochs=700, callbacks=[callback])
#700,700,500 - words
#simulation 1
f1 = np.array(Y).flatten().tolist()
data = pd.read_excel('TL+Allog1.xlsx', index_col=0, sheet_name="identity")
df = pd.DataFrame(data)
dataset = df.values
X = dataset[0:477, 0:35]
data = pd.read_excel('TL+Allog1.xlsx', index_col=0, sheet_name="position")
df = pd.DataFrame(data)
dataset = df.values
X1 = dataset[0:477, 0:100]
X= np.asarray(X).astype(np.float32)
X1 = np.asarray(X1).astype(np.float32)
pred = model.predict([X, X1])
f2 = np.array(pred).flatten().tolist()

valz = []
for i in range(477):
  r = np.corrcoef(pred[i], Y[i])
  valz.append(round(1-r[0][1],3))
df_new = pd.DataFrame()
df_new['target-TL+allog']= valz

data = pd.read_excel('TL-Allog1.xlsx', index_col=0, sheet_name="identity")
df = pd.DataFrame(data)
dataset = df.values
X = dataset[0:477, 0:35]
data = pd.read_excel('TL-Allog1.xlsx', index_col=0, sheet_name="position")
df = pd.DataFrame(data)
dataset = df.values
X1 = dataset[0:477, 0:100]
X= np.asarray(X).astype(np.float32)
X1 = np.asarray(X1).astype(np.float32)
pred1 = model.predict([X, X1])
valz = []
for i in range(477):
  r = np.corrcoef(pred1[i], Y[i])
  valz.append(round(1-r[0][1],3))
df_new['target-TL-allog']= valz
f3 = np.array(pred1).flatten().tolist()

data = pd.read_excel('SubL+Allog1.xlsx', index_col=0, sheet_name="identity")
df = pd.DataFrame(data)
dataset = df.values
X = dataset[0:477, 0:35]
data = pd.read_excel('SubL+Allog1.xlsx', index_col=0, sheet_name="position")
df = pd.DataFrame(data)
dataset = df.values 
X1 = dataset[0:477, 0:100]
X= np.asarray(X).astype(np.float32)
X1 = np.asarray(X1).astype(np.float32)
pred2 = model.predict([X, X1])

valz = []
for i in range(477):
  r = np.corrcoef(pred2[i], Y[i])
  valz.append(round(1-r[0][1],4))
df_new['target-subL+allog']= valz
f4 = np.array(pred2).flatten().tolist()

data = pd.read_excel('SubL-Allog1.xlsx', index_col=0, sheet_name="identity")
df = pd.DataFrame(data)
dataset = df.values
X = dataset[0:477, 0:35]
data = pd.read_excel('SubL-Allog1.xlsx', index_col=0, sheet_name="position")
df = pd.DataFrame(data)
dataset = df.values 
X1 = dataset[0:477, 0:100]
X= np.asarray(X).astype(np.float32)
X1 = np.asarray(X1).astype(np.float32)
pred3 = model.predict([X, X1])

valz = []
for i in range(477):
  r = np.corrcoef(pred3[i], Y[i])
  valz.append(round(1-r[0][1],4))
df_new['target-subL-allog']= valz
f5 = np.array(pred3).flatten().tolist()

valz = []
for i in range(477):
  r = np.corrcoef(pred1[i], pred3[i])
  valz.append(round(1-r[0][1],4))
df_new['tl-allog-subL-allog']= valz

valz = []
for i in range(477):
  r = np.corrcoef(pred[i], pred2[i])
  valz.append(round(1-r[0][1],4))
df_new['tl+allog-subL+allog']= valz

r = np.corrcoef(f1, f4)
print("correlation distance between identity and Subl+Allog\n", 1-r)
r = np.corrcoef(f1,f5)
print("correlation distance between identity and Subl-Allog\n", 1-r)
r = np.corrcoef(f1, f2)
print("correlation distance between identity and TL+Allog\n", 1-r)
r = np.corrcoef(f1, f3)
print("correlation distance between identity and TL-Allog\n", 1-r)
r = np.corrcoef(f3, f5)
print("correlation distance BETWEEN TL-Allog and SUBL-Allog is",1-r)  
r = np.corrcoef(f2, f4)
print("correlation distance BETWEEN TL+Allog and SUBL+Allog is",1-r) 
df_new.to_excel('Ttest_data_s3.xlsx',encoding = 'utf-8-sig')

