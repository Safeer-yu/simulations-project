
#plot 1
import numpy as np
import matplotlib.pyplot as plt

data = {'+ Allog':0.02, '- Allog':0.02}
courses = list(data.keys())
values = list(data.values())
fig, ax = plt.subplots(figsize=(4.5, 4.5))
plt.bar(courses, values, color ='skyblue',
        width = 0.3)
 
plt.xlabel("TL Priming")
plt.ylabel("Correlation")
plt.title("TL Priming modulated by allography")
plt.show()

#plot 2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
fig, ax = plt.subplots(figsize=(4.5, 4.5))
df = pd.DataFrame({
      'x_axis': ['TL-Allog','TL+Allog','SubL-Allog','SubL+Allog'],
      'y_axis': [0.25,0.24,0.23,0.21] })
plt.plot('x_axis','y_axis', data=df,linestyle='-',marker='o')
plt.show()

#plot3
fig, ax = plt.subplots(figsize=(4.0, 5.0))
sns.set(style="whitegrid", color_codes=True)
df = pd.DataFrame({
      'data'   : ['-Allog','+Allog'],
      'TL-SubL': [0.1575,0.1783]})
pal = sns.color_palette("Blues_d", 2)
rank = df['TL-SubL'].argsort().argsort()

sns.barplot(x=df['data'],y=df['TL-SubL'], palette=np.array(pal[::-1])[rank],n_boot=20).set(title='TL-Priming')
plt.show()
