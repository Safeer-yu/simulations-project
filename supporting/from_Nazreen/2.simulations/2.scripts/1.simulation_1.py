
#SIMULATION 1 
#id, pos, allog -> orth
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from keras.layers import concatenate, Dense
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import load_model
from tensorflow.keras import regularizers
from keras import Input
from keras.engine import input_spec

#id,pos,allog ->orth
#NN ARCHITECTURE- simulation 1

data = pd.read_excel('Target1.xlsx', index_col=0, sheet_name="identity")
df = pd.DataFrame(data)  
dataset = df.values
X = dataset[0:477, 0:35]

data = pd.read_excel('Target1.xlsx', index_col=0, sheet_name="position")
df1 = pd.DataFrame(data)
dataset = df1.values 
X1 = dataset[0:477, 0:100] 

data = pd.read_excel('Targetx.xlsx', index_col=0,sheet_name="allog")
df1 = pd.DataFrame(data)
dataset = df1.values 
X2 = dataset[0:477, 0:116] 

inputA = Input(shape=(35,)) 
inputB = Input(shape=(100,))
inputC = Input(shape=(116,))

data = pd.read_excel('orthograph_sim1_2.xlsx')
df = pd.DataFrame(data)  
dataset = df.values
Y = dataset[0:477,0:350]
Y = np.asarray(Y).astype(np.float32)
X= np.asarray(X).astype(np.float32)
X1 = np.asarray(X1).astype(np.float32)
X2 = np.asarray(X2).astype(np.float32)
Output = Input(shape=(350,))

x = Dense(700, activation="sigmoid")(inputA)
x = Model(inputs=inputA, outputs=x)

y = Dense(700, activation="sigmoid")(inputB)
y = Model(inputs=inputB, outputs=y)

q = Dense(700, activation="sigmoid")(inputC)
q = Model(inputs=inputC, outputs=q)

combined = concatenate([x.output, y.output, q.output])

z = Dense(500, activation="sigmoid")(combined)
z1 = Dense(350, activation="sigmoid")(z)
model = Model(inputs=[x.input, y.input,q.input], outputs=z1)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
model.compile(optimizer="adam", loss=["mse"])
model.fit([X, X1,X2],Y,batch_size=25, epochs=700, callbacks=[callback])

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
data = pd.read_excel('TL+Allogx.xlsx', index_col=0,sheet_name="allog")
df1 = pd.DataFrame(data)
dataset = df1.values 
X2 = dataset[0:477, 0:116] 
X= np.asarray(X).astype(np.float32)
X1 = np.asarray(X1).astype(np.float32)
X2 = np.asarray(X2).astype(np.float32)

pred = model.predict([X, X1, X2])
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
data = pd.read_excel('TL-Allogx.xlsx', index_col=0,sheet_name="allog")
df1 = pd.DataFrame(data)
dataset = df1.values 
X2 = dataset[0:477, 0:116] 
X= np.asarray(X).astype(np.float32)
X1 = np.asarray(X1).astype(np.float32)
X2 = np.asarray(X2).astype(np.float32)
pred1 = model.predict([X, X1,X2])
f3 = np.array(pred1).flatten().tolist()

valz = []
for i in range(477):
  r = np.corrcoef(pred1[i], Y[i])
  valz.append(round(1-r[0][1],3))
df_new['target-TL-allog']= valz


data = pd.read_excel('SubL+Allog1.xlsx', index_col=0, sheet_name="identity")
df = pd.DataFrame(data)
dataset = df.values
X = dataset[0:477, 0:35]
data = pd.read_excel('SubL+Allog1.xlsx', index_col=0, sheet_name="position")
df = pd.DataFrame(data)
dataset = df.values 
X1 = dataset[0:477, 0:100]
data = pd.read_excel('SubL+Allogx.xlsx', index_col=0,sheet_name="allog")
df1 = pd.DataFrame(data)
dataset = df1.values 
X2 = dataset[0:477, 0:116] 
X= np.asarray(X).astype(np.float32)
X1 = np.asarray(X1).astype(np.float32)
X2 = np.asarray(X2).astype(np.float32)
pred2 = model.predict([X, X1, X2])
f4 = np.array(pred2).flatten().tolist()

valz = []
for i in range(477):
  r = np.corrcoef(pred2[i], Y[i])
  valz.append(round(1-r[0][1],4))
df_new['target-subL+allog']= valz

data = pd.read_excel('SubL-Allog1.xlsx', index_col=0, sheet_name="identity")
df = pd.DataFrame(data)
dataset = df.values
X = dataset[0:477, 0:35]
data = pd.read_excel('SubL-Allog1.xlsx', index_col=0, sheet_name="position")
df = pd.DataFrame(data)
dataset = df.values 
X1 = dataset[0:477, 0:100]
data = pd.read_excel('SubL-Allogx.xlsx', index_col=0,sheet_name="allog")
df1 = pd.DataFrame(data)
dataset = df1.values 
X2 = dataset[0:477, 0:116] 
X= np.asarray(X).astype(np.float32)
X1 = np.asarray(X1).astype(np.float32)
X2 = np.asarray(X2).astype(np.float32)
pred3 = model.predict([X, X1, X2])
f5 = np.array(pred3).flatten().tolist()

valz = []
for i in range(477):
  r = np.corrcoef(pred3[i], Y[i])
  valz.append(round(1-r[0][1],4))
df_new['target-subL-allog']= valz

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
print("CORRELATION distance BETWEEN TL-Allog and SUBL-Allog is",1-r)  
r = np.corrcoef(f2, f4)
print("CORRELATION distance BETWEEN TL+Allog and SUBL+Allog is",1-r) 
df_new.to_excel('Ttest_data_s1.xlsx',encoding = 'utf-8-sig')

df_new.to_excel('Ttest_data_s1.xlsx',encoding = 'utf-8-sig')

#RESULTS
correlation distance between identity and Subl+Allog
 [[0.         0.26807292]
 [0.26807292 0.        ]]
correlation distance between identity and Subl-Allog
 [[0.        0.2327267]
 [0.2327267 0.       ]]
correlation distance between identity and TL+Allog
 [[0.         0.13759767]
 [0.13759767 0.        ]]
correlation distance between identity and TL-Allog
 [[0.         0.07125341]
 [0.07125341 0.        ]]
CORRELATION distance BETWEEN TL-Allog and SUBL-Allog is [[0.         0.18972973]
 [0.18972973 0.        ]]
CORRELATION distance BETWEEN TL+Allog and SUBL+Allog is [[0.         0.19941005]
 [0.19941005 0.        ]]








