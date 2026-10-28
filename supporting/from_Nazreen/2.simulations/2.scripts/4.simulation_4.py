#simulation 4 -id+pos, allog- mapped to orthography 
#SIMULATION 4- RUN THIS SCRIPT
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

data = pd.read_excel('Target2.xlsx', index_col=0, sheet_name="id+pos")
df = pd.DataFrame(data)  
dataset = df.values
X = dataset[0:477, 0:350]
X= np.asarray(X).astype(np.float32)
inputA = Input(shape=(350,))

data = pd.read_excel('Targetx.xlsx', index_col =0,sheet_name="allog")
df1 = pd.DataFrame(data)
dataset = df1.values 
X1 = dataset[0:477, 0:116] 
X1= np.asarray(X1).astype(np.float32)
inputB = Input(shape=(116,))

data = pd.read_excel('orthograph_sim1_2.xlsx')
df = pd.DataFrame(data)  
dataset = df.values
Y = dataset[0:477, 0:350]
Y = np.asarray(Y).astype(np.float32)

x = Dense(700, activation="sigmoid")(inputA)
x = Model(inputs=inputA, outputs=x)

y = Dense(700, activation="sigmoid")(inputB)
y = Model(inputs=inputB, outputs=y)

combined = concatenate([x.output, y.output])

z = Dense(700, activation="sigmoid")(combined)
z = Dense(500, activation="sigmoid")(z)
z1 = Dense(350, activation="sigmoid")(z)
model = Model(inputs=[x.input, y.input], outputs=z1)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
model.compile(optimizer="adam", loss=["mse"])
model.fit([X, X1],Y,batch_size=25, epochs=2000, callbacks=[callback])

#simulation 4
f1 = np.array(Y).flatten().tolist()
data = pd.read_excel('TL-Allog.xlsx',  index_col =0,sheet_name="id+pos")
df = pd.DataFrame(data)
dataset = df.values
X = dataset[0:477, 0:350]
X= np.asarray(X).astype(np.float32)

data = pd.read_excel('TL-Allogx.xlsx',  index_col =0,sheet_name="allog")
df1 = pd.DataFrame(data)
dataset = df1.values 
X1 = dataset[0:477, 0:116] 
X1= np.asarray(X1).astype(np.float32)

pred = model.predict([X,X1])
f2 = np.array(pred).flatten().tolist()  #tlnoallog

valz = []
for i in range(477):
  r = np.corrcoef(pred[i], Y[i])
  valz.append(round(1-r[0][1],3))
df_new = pd.DataFrame()
df_new['target-TL-allog']= valz

data = pd.read_excel('TL+Allog.xlsx',  index_col =0,sheet_name="id+pos")
df = pd.DataFrame(data)
dataset = df.values
X = dataset[0:477, 0:350]
X= np.asarray(X).astype(np.float32)

data = pd.read_excel('TL+Allogx.xlsx',  index_col =0,sheet_name="allog")
df1 = pd.DataFrame(data)
dataset = df1.values 
X1 = dataset[0:477, 0:116] 
X1= np.asarray(X1).astype(np.float32)

pred1 = model.predict([X,X1])
f3 = np.array(pred1).flatten().tolist() #tlallog
valz = []
for i in range(477):
  r = np.corrcoef(pred1[i], Y[i])
  valz.append(round(1-r[0][1],3))
df_new['target-TL+allog']= valz

data = pd.read_excel('SubL-Allog.xlsx',   index_col =0,sheet_name="id+pos")
df = pd.DataFrame(data)
dataset = df.values
X = dataset[0:477, 0:350]
X= np.asarray(X).astype(np.float32)

data = pd.read_excel('SubL-Allogx.xlsx',  index_col =0,sheet_name="allog")
df1 = pd.DataFrame(data)
dataset = df1.values 
X1 = dataset[0:477, 0:116] 
X1= np.asarray(X1).astype(np.float32)

pred2 = model.predict([X,X1])
f4 = np.array(pred2).flatten().tolist() #sublnoallog

valz = []
for i in range(477):
  r = np.corrcoef(pred2[i], Y[i])
  valz.append(round(1-r[0][1],4))
df_new['target-subL-allog']= valz

data = pd.read_excel('SubL+Allog.xlsx',   index_col =0,sheet_name="id+pos")
df = pd.DataFrame(data)
dataset = df.values
X = dataset[0:477, 0:350]
X= np.asarray(X).astype(np.float32)

data = pd.read_excel('SubL+Allogx.xlsx',  index_col =0,sheet_name="allog")
df1 = pd.DataFrame(data)
dataset = df1.values 
X1 = dataset[0:477, 0:116] 
X1= np.asarray(X1).astype(np.float32)

pred3 = model.predict([X,X1])
f5 = np.array(pred3).flatten().tolist() #sublallog
valz = []
for i in range(477):
  r = np.corrcoef(pred3[i], Y[i])
  valz.append(round(1-r[0][1],4))
df_new['target-subL+allog']= valz

r = np.corrcoef(f1,f5)
print(" correlation distance between identity and SublAllog\n", 1-r)
r = np.corrcoef(f1, f4)
print(" correlation distance between identity and SublnoAllog\n", 1-r)
r = np.corrcoef(f1, f3)
print(" correlation distance between identity and TLAllog\n", 1-r)
r = np.corrcoef(f1, f2)
print("correlation distance between identity and TLnoAllog\n", 1-r)
r = np.corrcoef(f3, f5)
print("correlation distance BETWEEN TL+Allog and SUBL+Allog is",1-r)  
r = np.corrcoef(f2, f4)
print("correlation distance BETWEEN TL-Allog and SUBL-Allog is",1-r) 

df_new.to_excel('Ttest_data_s4.xlsx',encoding = 'utf-8-sig')

#output
correlation  between identity and SublAllog
 [[1.         0.74273484]
 [0.74273484 1.        ]]
 correlation  between identity and SublnoAllog
 [[1.         0.74119774]
 [0.74119774 1.        ]]
 correlation  between identity and TLAllog
 [[1.         0.76166368]
 [0.76166368 1.        ]]
correlation  between identity and TLnoAllog
 [[1.         0.83338781]
 [0.83338781 1.        ]]
correlation  BETWEEN TL+Allog and SUBL+Allog is [[1.         0.80651849]
 [0.80651849 1.        ]]
correlation  BETWEEN TL-Allog and SUBL-Allog is [[1.         0.79992116]
 [0.79992116 1.        ]]
