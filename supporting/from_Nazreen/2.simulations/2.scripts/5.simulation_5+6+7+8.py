

#simulation 5 -id,pos,allog->semantics

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
data = pd.read_excel('Targetx.xlsx', index_col=0, sheet_name="allog")
df1 = pd.DataFrame(data)
dataset = df1.values 
X2 = dataset[0:477, 0:116] 

inputA = Input(shape=(35,)) 
inputB = Input(shape=(100,))
inputC = Input(shape=(116,))

data = pd.read_excel('semantic_vec.xlsx')
df = pd.DataFrame(data)  

dataset = df.values
Y = dataset[0:477,0:300]
Y = np.asarray(Y).astype(np.float32)
X= np.asarray(X).astype(np.float32)
X1 = np.asarray(X1).astype(np.float32)
X2 = np.asarray(X2).astype(np.float32)

Output = Input(shape=(300,))

x = Dense(700, activation="sigmoid")(inputA)
x = Model(inputs=inputA, outputs=x)

y = Dense(700, activation="sigmoid")(inputB)
y = Model(inputs=inputB, outputs=y)

q = Dense(700, activation="sigmoid")(inputC)
q = Model(inputs=inputC, outputs=q)

combined = concatenate([x.output, y.output, q.output])

z = Dense(500, activation="sigmoid")(combined)
z1 = Dense(300, activation="sigmoid")(z)

model = Model(inputs=[x.input, y.input, q.input], outputs=z1)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
model.compile(optimizer="adam", loss=["mse"])

model.fit([X, X1, X2],Y,batch_size=25, epochs=700, callbacks=[callback])
#700,700,500 - words

#simulation 5
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

pred = model.predict([X, X1, X2])
f3 = np.array(pred).flatten().tolist()
valz = []
for i in range(477):
  r = np.corrcoef(pred[i], Y[i])
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

pred = model.predict([X, X1, X2])
f4 = np.array(pred).flatten().tolist()
valz = []
for i in range(477):
  r = np.corrcoef(pred[i], Y[i])
  valz.append(round(1-r[0][1],3))
df_new['target-SubL+allog']= valz

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

pred = model.predict([X, X1,X2])
f5 = np.array(pred).flatten().tolist()
valz = []
for i in range(477):
  r = np.corrcoef(pred[i], Y[i])
  valz.append(round(1-r[0][1],3))
df_new['target-SubL-allog']= valz

r = np.corrcoef(f1, f4)
print("corelation between identity and Subl+Allog\n", r)
r = np.corrcoef(f1,f5)
print(" corelation between identity and Subl-Allog\n", r)
r = np.corrcoef(f1, f2)
print("corelation between identity and TL+Allog\n", r)
r = np.corrcoef(f1, f3)
print("corelation between identity and TL-Allog\n", r)

r = np.corrcoef(f3, f5)
print("CORRELATION BETWEEN TL-Allog and SUBL-Allog is",r)  
 
r = np.corrcoef(f2, f4)
print("CORRELATION BETWEEN TL+Allog and SUBL+Allog is",r) 
df_new.to_excel('Ttest_data_s5.xlsx',encoding = 'utf-8-sig')

between identity and Subl+Allog
 [[1.         0.30093265]
 [0.30093265 1.        ]]
between identity and Subl-Allog
 [[1.         0.30090628]
 [0.30090628 1.        ]]
between identity and TL+Allog
 [[1.         0.30092645]
 [0.30092645 1.        ]]
between identity and TL-Allog
 [[1.         0.30093613]
 [0.30093613 1.        ]]
CORRELATION BETWEEN TL-Allog and SUBL-Allog is [[1.         0.99999756]
 [0.99999756 1.        ]]
CORRELATION BETWEEN TL+Allog and SUBL+Allog is [[1.         0.99999689]
 [0.99999689 1.        ]]

#simulation 6- id+pos+allog->semantics

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

data = pd.read_excel('semantic_vec.xlsx')
df = pd.DataFrame(data)  
dataset = df.values
Y = dataset[0:477,0:300]
Y = np.asarray(Y).astype(np.float32)
Output = Input(shape=(300,))
x = Dense(1200, activation="sigmoid")(inputA)
y = Dense(700, activation="sigmoid")(x)
z = Dense(300, activation="sigmoid")(y)

model = Model(inputA, outputs=z)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
model.compile(optimizer="adam", loss=["mse"])
model.fit(X,Y, batch_size=25, epochs=700, callbacks=[callback])

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
print("correlation distance between identity and TLnoAllog\n",1- r)
r = np.corrcoef(f3, f5)
print("CORRELATION distance BETWEEN TL+Allog and SUBL+Allog is",1-r)  
r = np.corrcoef(f2, f4)
print("CORRELATION distance BETWEEN TL-Allog and SUBL-Allog is",1-r) 
df_new.to_excel('Ttest_data_s6.xlsx',encoding = 'utf-8-sig')
#output
correlation distance between identity and SublAllog
 [[1.        0.3434266]
 [0.3434266 1.       ]]
correlation distance between identity and SublnoAllog
 [[1.         0.34741388]
 [0.34741388 1.        ]]
correlation distance between identity and TLAllog
 [[1.         0.34137076]
 [0.34137076 1.        ]]
correlation distance between identity and TLnoAllog
 [[1.         0.34733574]
 [0.34733574 1.        ]]
CORRELATION distance BETWEEN TL+Allog and SUBL+Allog is [[1.         0.98697891]
 [0.98697891 1.        ]]
CORRELATION distance BETWEEN TL-Allog and SUBL-Allog is [[1.         0.99077856]
 [0.99077856 1.        ]]

#simulation 7 : id,pos -> semantics
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
data = pd.read_excel('/content/semantic_vec.xlsx')
df = pd.DataFrame(data)  
dataset = df.values
Y = dataset[0:477,0:300]
Y = np.asarray(Y).astype(np.float32)
X= np.asarray(X).astype(np.float32)
X1 = np.asarray(X1).astype(np.float32)
Output = Input(shape=(300,))

x = Dense(700, activation="sigmoid")(inputA)
x = Model(inputs=inputA, outputs=x)
y = Dense(700, activation="sigmoid")(inputB)
y = Model(inputs=inputB, outputs=y)
combined = concatenate([x.output, y.output])
z = Dense(500, activation="sigmoid")(combined)
z1 = Dense(300, activation="sigmoid")(z)

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
df_new.to_excel('Ttest_data_s7.xlsx',encoding = 'utf-8-sig')

#output
correlation distance between identity and Subl+Allog
 [[1.         0.34664907]
 [0.34664907 1.        ]]
correlation distance between identity and Subl-Allog
 [[1.         0.34590229]
 [0.34590229 1.        ]]
correlation distance between identity and TL+Allog
 [[1.         0.34788667]
 [0.34788667 1.        ]]
correlation distance between identity and TL-Allog
 [[1.         0.34837461]
 [0.34837461 1.        ]]
correlation distance BETWEEN TL-Allog and SUBL-Allog is [[1.         0.98952646]
 [0.98952646 1.        ]]
correlation distance BETWEEN TL+Allog and SUBL+Allog is [[1.         0.99034645]
 [0.99034645 1.        ]]

#simulation 8 -id+pos,allog->semantics
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

data = pd.read_excel('/content/semantic_vec.xlsx')
df = pd.DataFrame(data)  
dataset = df.values
Y = dataset[0:477, 0:300]
Y = np.asarray(Y).astype(np.float32)

x = Dense(700, activation="sigmoid")(inputA)
x = Model(inputs=inputA, outputs=x)

y = Dense(700, activation="sigmoid")(inputB)
y = Model(inputs=inputB, outputs=y)

combined = concatenate([x.output, y.output])

z = Dense(700, activation="sigmoid")(combined)
z = Dense(500, activation="sigmoid")(z)
z1 = Dense(300, activation="sigmoid")(z)
model = Model(inputs=[x.input, y.input], outputs=z1)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
model.compile(optimizer="adam", loss=["mse"])
model.fit([X, X1],Y,batch_size=25, epochs=700, callbacks=[callback])

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
  valz.append(round(1-r[0][1],3))
df_new['target-SubL-allog']= valz

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
  valz.append(round(1-r[0][1],3))
df_new['target-SubL+allog']= valz

r = np.corrcoef(f1,f5)
print("correlation between identity and SublAllog\n", r)
r = np.corrcoef(f1, f3)
print(" correlation between identity and TLAllog\n", r)
r = np.corrcoef(f1, f4)
print("correlation between identity and SublnoAllog\n", r)
r = np.corrcoef(f1, f2)
print("correlation between identity and TLnoAllog\n", r)
r = np.corrcoef(f3, f5)
print("CORRELATION BETWEEN TL+Allog and SUBL+Allog is",r)  
r = np.corrcoef(f2, f4)
print("CORRELATION BETWEEN TL-Allog and SUBL-Allog is",r) 

valz = []
for i in range(477):
  r = np.corrcoef(pred[i], pred2[i])
  valz.append(round(1-r[0][1],4))
df_new['tl-allog-subL-allog']= valz

valz = []
for i in range(477):
  r = np.corrcoef(pred1[i], pred3[i])
  valz.append(round(1-r[0][1],4))
df_new['tl+allog-subL+allog']= valz
df_new.to_excel('Ttest_data_s8.xlsx',encoding = 'utf-8-sig')

#output

15/15 [==============================] - 0s 8ms/step
15/15 [==============================] - 0s 7ms/step
15/15 [==============================] - 0s 7ms/step
15/15 [==============================] - 0s 7ms/step
between identity and SublAllog
 [[1.         0.30233306]
 [0.30233306 1.        ]]
between identity and TLAllog
 [[1.         0.30233298]
 [0.30233298 1.        ]]
between identity and SublnoAllog
 [[1.         0.30233186]
 [0.30233186 1.        ]]
between identity and TLnoAllog
 [[1.        0.3023323]
 [0.3023323 1.       ]]
CORRELATION BETWEEN TL+Allog and SUBL+Allog is [[1. 1.]
 [1. 1.]]
CORRELATION BETWEEN TL-Allog and SUBL-Allog is [[1. 1.]
 [1. 1.]]

#simulation 5-8 - output layer modification 
from gensim.models import KeyedVectors
model_arabic = KeyedVectors.load_word2vec_format('drive/MyDrive/cc.ar.300 2.vec')

import pandas as pd
import numpy as np
df = pd.read_excel('/content/Words_That_Have_Semantic_Vecs.xlsx',sheet_name='For_Simulation_TL_1')
wordlist = df['Target'].to_list()
vecs = []
for word in wordlist:
  vecs.append(model_arabic[word])

dfx= pd.DataFrame(vecs)
dfx.columns = ['vec_'+str(i) for i in dfx.columns]
dfx['Target']=wordlist
dfx.to_excel('semantic_vec.xlsx',encoding = 'utf-8-sig')
print(dfx.head(5))
