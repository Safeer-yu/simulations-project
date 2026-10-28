
#PREREQUISITES: pip install arabic-reshaper
#               pip install python-bidi
#INPUT AND OUTPUT FOR S2,S4,S6,S8 
#add multiple df to excel sheet
import os
import pandas as pd
def append_df_to_excel(filename, df, sheet_name='Sheet1', startrow=None,
                       truncate_sheet=False,
                      **to_excel_kwargs): 
    if not os.path.isfile(filename):
        df.to_excel(
            filename,
            sheet_name=sheet_name, 
            startrow=startrow if startrow is not None else 0, 
            **to_excel_kwargs)
        return
    if 'engine' in to_excel_kwargs:
        to_excel_kwargs.pop('engine')
    writer = pd.ExcelWriter(filename, engine='openpyxl', mode='a')
    writer.book = load_workbook(filename) 
    if startrow is None and sheet_name in writer.book.sheetnames:
        startrow = writer.book[sheet_name].max_row
    if truncate_sheet and sheet_name in writer.book.sheetnames:
        idx = writer.book.sheetnames.index(sheet_name)
        writer.book.remove(writer.book.worksheets[idx])
        writer.book.create_sheet(sheet_name, idx)
    writer.sheets = {ws.title:ws for ws in writer.book.worksheets}
    if startrow is None:
        startrow = 0
    df.to_excel(writer, sheet_name, startrow=startrow, **to_excel_kwargs)
    writer.save()

#simulation 2 INPUT LAYER : id+pos+allog 
import pandas as pd
import numpy as np
from openpyxl import load_workbook
from keras.preprocessing.text import Tokenizer
import nltk
import os
import arabic_reshaper
from bidi.algorithm import get_display

def indexes(my_list, desired_element):
    return [index+1 for index, element in enumerate(my_list) if element == desired_element]

def pos_id_allog(word):
  word = arabic_reshaper.reshape(word)
  id_pos = []
  i = 0
  letters = "ء أ ؤ إ ئ ا ب ت ة ث ج ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ك ل م ن ه و ى ي"
  letters = "ء أ ـأ ؤ ـؤ إ ـإ ئ ـئ ـئـ ئـ ا ـا ب ـب ـبـ بـ ت ـت ـتـ تـ ة ـة ث ـث ـثـ ثـ ج ـج ـجـ جـ ح ـح ـحـ حـ خ ـخ ـخـ خـ د ـد ذ ـذ ر ـر ز ـز س ـس ـسـ سـ ش ـش ـشـ شـ ص ـص ـصـ صـ ض ـض ـضـ ضـ ط ـط ـطـ طـ ظ ـظ ـظـ ظـ ع ـع ـعـ عـ غ ـغ ـغـ غـ ف ـف ـفـ فـ ق ـق ـقـ قـ ك ـك ـكـ كـ ل ـل ـلـ لـ م ـم ـمـ مـ ن ـن ـنـ نـ ه ـه ـهـ هـ و ـو ى ـى ي ـي ـيـ يـ"
  letters = arabic_reshaper.reshape(letters) 
  for i in range(len(letters)):
    position = indexes(word,letters[i])
    value = [0,0,0,0,0,0,0,0,0,0]
    if position != []:
      for j in range(len(position)):
        value[position[j]-1] = 1
      print(position,value, letters[i])
      id_pos.append(value)
    elif letters[i]not in [' ','-','ـ']:
      id_pos.append(value)
  print(id_pos,word)
  idpos = [item for sublist in id_pos for item in sublist]
  return idpos

if __name__ == '__main__':
  #df = pd.read_excel('/content/Words_That_Have_Semantic_Vecs.xlsx',sheet_name='For_Simulation_TL_1')
  #l1 = df['Target'].to_list()
  #l2 = df['TL-Allog'].to_list()
  #l3 = df['TL+Allog'].to_list()
  #l4 = df['SubL-Allog'].to_list()
  #l5 = df['SubL+Allog'].to_list()
  l1=['منبوذا']
  final_list = l1
  d = pd.DataFrame()
  df = pd.DataFrame()
  for i in range(477):
    wrd = final_list[i]
    print(wrd)
    identity_position = pos_id_allog(wrd)
    #print(identity_position)
    data = pd.DataFrame([wrd])
    d = d.append(data)
    data1 = pd.DataFrame([identity_position])
    df = df.append(data1)
    df.replace(np.nan, '0', regex=True, inplace= True)                 
  start = 1
  d.columns = range(start, start + len(d.columns))
  d.index = np.arange(1, len(d)+1)
  d.to_excel('Targets2.xlsx', sheet_name = 'word')
  df.columns = range(start, start + len(df.columns))
  df.index = np.arange(1, len(df)+1)
  append_df_to_excel('Targets2.xlsx',df, sheet_name='id+pos+allog')


#INPUT LAYER id+pos for simulation 4
def indexes(my_list, desired_element):
    return [index+1 for index, element in enumerate(my_list) if element == desired_element]

def pos_id(word):
  id_pos = []
  i = 0
  #words = "أ ا ب ت ث ج ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ك ل م ن ه و ي" 
  words = "ء أ ؤ إ ئ ا ب ت ة ث ج ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ك ل م ن ه و ى ي"
  for i in range(len(words)):
    position = indexes(word,words[i])
    value = [0,0,0,0,0,0,0,0,0,0]
    if position != []:
      for j in range(len(position)):
        value[position[j]-1] = 1
      #print(position,value, words[i])
      id_pos.append(value)
    elif words[i]not in [' ','-','ـ']:
      id_pos.append(value)
  #print(id_pos,word)
  idpos = [item for sublist in id_pos for item in sublist]
  return idpos
if __name__ == '__main__':

  df = pd.read_excel('/content/Words_That_Have_Semantic_Vecs.xlsx',sheet_name='For_Simulation_TL_1')
  l1 = df['Target'].to_list()
  #l2 = df['TL-Allog'].to_list()
  #l3 = df['TL+Allog'].to_list()
  l4 = df['SubL-Allog'].to_list()
  #l5 = df['SubL+Allog'].to_list()
  final_list = l4
  d = pd.DataFrame()
  df = pd.DataFrame()
  for i in range(477):
    wrd = final_list[i]
    identity_position = pos_id(wrd)
    #print(identity_position)
    data = pd.DataFrame([wrd])
    d = d.append(data)
    data1 = pd.DataFrame([identity_position])
    df = df.append(data1)
    df.replace(np.nan, '0', regex=True, inplace= True)                 
  start = 1
  d.columns = range(start, start + len(d.columns))
  d.index = np.arange(1, len(d)+1)
  d.to_excel('SubL-Allog.xlsx', sheet_name = 'word')
  df.columns = range(start, start + len(df.columns))
  df.index = np.arange(1, len(df)+1)
  append_df_to_excel('SubL-Allog.xlsx',df, sheet_name='id+pos')

#output layer for simulation 1,2,3,4,X
import numpy as np
from keras.utils import np_utils
import pandas as pd
import re
def onehotencod(word):
  letters = "ءأؤإئابتةثجحخدذرزسشصضطظعغفقكلمنهوىي"
  encoded_word = []
  for i in word:
    position = letters.find(i)
    #print(y_train_label_onehot[position],i)
    encoded_word.append(y_train_label_onehot[position].tolist())
  one_hot_word = [value for sublist in encoded_word for value in sublist]
  #print(flattened_word)
  while(len(one_hot_word)!=350): #35units * 10 letter words = 350
    one_hot_word.append(0)
  return(one_hot_word)

y_train_label = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]
y_train_label.reverse()
y_train_label_onehot = np_utils.to_categorical(y_train_label)
#sample = 'ارتجلت'
#print(onehotencod(sample))
df = pd.read_excel('/content/Words_That_Have_Semantic_Vecs.xlsx',sheet_name='For_Simulation_TL_1')
dfx = pd.DataFrame()
dfx['onehotenc'] = df.Target.apply(onehotencod)
dfx['Target'] = df['Target']
dfx.to_excel('orthograph_sim1_2.xlsx')

#simulation 2 :id+pos+allog, INPUT LAYER
import pandas as pd
import numpy as np
from openpyxl import load_workbook
from keras.preprocessing.text import Tokenizer
import nltk
import os
import arabic_reshaper
from bidi.algorithm import get_display

def indexes(my_list, desired_element):
    return [index+1 for index, element in enumerate(my_list) if element == desired_element]

def pos_id_allog(word):
  word = arabic_reshaper.reshape(word)
  id_pos = []
  i = 0
  letters = "ء أ ؤ إ ئ ا ب ت ة ث ج ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ك ل م ن ه و ى ي"
  letters = "ء أ ـأ ؤ ـؤ إ ـإ ئ ـئ ـئـ ئـ ا ـا ب ـب ـبـ بـ ت ـت ـتـ تـ ة ـة ث ـث ـثـ ثـ ج ـج ـجـ جـ ح ـح ـحـ حـ خ ـخ ـخـ خـ د ـد ذ ـذ ر ـر ز ـز س ـس ـسـ سـ ش ـش ـشـ شـ ص ـص ـصـ صـ ض ـض ـضـ ضـ ط ـط ـطـ طـ ظ ـظ ـظـ ظـ ع ـع ـعـ عـ غ ـغ ـغـ غـ ف ـف ـفـ فـ ق ـق ـقـ قـ ك ـك ـكـ كـ ل ـل ـلـ لـ م ـم ـمـ مـ ن ـن ـنـ نـ ه ـه ـهـ هـ و ـو ى ـى ي ـي ـيـ يـ"
  letters = arabic_reshaper.reshape(letters) 
  for i in range(len(letters)):
    position = indexes(word,letters[i])
    value = [0,0,0,0,0,0,0,0,0,0]
    if position != []:
      for j in range(len(position)):
        value[position[j]-1] = 1
      #print(position,value, letters[i])
      id_pos.append(value)
    elif letters[i]not in [' ','-','ـ']:
      id_pos.append(value)
  #print(id_pos,word)
  idpos = [item for sublist in id_pos for item in sublist]
  return idpos

if __name__ == '__main__':
  df = pd.read_excel('/content/Words_That_Have_Semantic_Vecs.xlsx',sheet_name='For_Simulation_TL_1')
  l1 = df['Target'].to_list()
  #l2 = df['TL-Allog'].to_list()
  #l3 = df['TL+Allog'].to_list()
  #l4 = df['SubL-Allog'].to_list()
  #l5 = df['SubL+Allog'].to_list()
  final_list = l1
  d = pd.DataFrame()
  df = pd.DataFrame()
  for i in range(477):
    wrd = final_list[i]
    identity_position = pos_id_allog(wrd)
    #print(identity_position)
    data = pd.DataFrame([wrd])
    d = d.append(data)
    data1 = pd.DataFrame([identity_position])
    df = df.append(data1)
    df.replace(np.nan, '0', regex=True, inplace= True)                 
  start = 1
  d.columns = range(start, start + len(d.columns))
  d.index = np.arange(1, len(d)+1)
  d.to_excel('Targets2.xlsx', sheet_name = 'word')
  df.columns = range(start, start + len(df.columns))
  df.index = np.arange(1, len(df)+1)
  append_df_to_excel('Targets2.xlsx',df, sheet_name='id+pos+allog')

#Allog input layer for Simulation 4
import os
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from keras.preprocessing.text import Tokenizer
import nltk
import arabic_reshaper
from bidi.algorithm import get_display

df = pd.read_excel('/content/Words_That_Have_Semantic_Vecs.xlsx',sheet_name='For_Simulation_TL_1')

def indexes(my_list, desired_element):
    return [index+1 for index, element in enumerate(my_list) if element == desired_element]

def input_sim1(input_word):
    input_word = arabic_reshaper.reshape(input_word)
    t = Tokenizer()
    #fit_text = "أ إ ـا ا ـآ آ ب بـ ـبـ ـب ت تـ ـتـ ـت ث ثـ ـثـ ـث ج جـ ـجـ ـج ح حـ ـحـ ـح خ خـ ـخـ ـخ د ـد ذ ـذ ر ـر ز ـز س سـ ـسـ ـس ش شـ ـشـ ـش ص صـ ـصـ ـص ض ضـ ـضـ ـض ط طـ ـطـ ـط ظ ظـ ـظـ ـظ ع عـ ـعـ ـع غ غـ ـغـ ـغ ف فـ ـفـ ـف ق قـ ـقـ ـق ك كـ ـكـ ـك ل لـ ـلـ ـل م مـ ـمـ ـم ن نـ ـنـ ـن ه هـ ـهـ ـه و ـو ي يـ ـيـ ـي"
    fit_text = "ء أ ـأ ؤ ـؤ إ ـإ ئ ـئ ـئـ ئـ ا ـا ب ـب ـبـ بـ ت ـت ـتـ تـ ة ـة ث ـث ـثـ ثـ ج ـج ـجـ جـ ح ـح ـحـ حـ خ ـخ ـخـ خـ د ـد ذ ـذ ر ـر ز ـز س ـس ـسـ سـ ش ـش ـشـ شـ ص ـص ـصـ صـ ض ـض ـضـ ضـ ط ـط ـطـ طـ ظ ـظ ـظـ ظـ ع ـع ـعـ عـ غ ـغ ـغـ غـ ف ـف ـفـ فـ ق ـق ـقـ قـ ك ـك ـكـ كـ ل ـل ـلـ لـ م ـم ـمـ مـ ن ـن ـنـ نـ ه ـه ـهـ هـ و ـو ى ـى ي ـي ـيـ يـ"
    fit_text = arabic_reshaper.reshape(fit_text) 
    t.fit_on_texts(fit_text)
    sequences = t.texts_to_sequences(input_word)
    #print("word_index : ", t.word_index)
    identity = []                                                                                                                                                                                                                                                                   
    flattened = [val for sublist in sequences for val in sublist]
    #print("word", flattened)
    for key in t.word_index.values():
        if key in flattened:
            identity.append('1')
        else:
            identity.append('0')
    #print("letter identity", identity)  
    return identity,flattened
if __name__ == '__main__':   
    #l1 = df['Target'].to_list()   
    #l2 = df['TL-Allog'].to_list()
    #l3 = df['TL+Allog'].to_list()
    #l4 = df['SubL-Allog'].to_list()
    l5 = df['SubL+Allog'].to_list()
    input_list = l5
    df2 = pd.DataFrame()
    df = pd.DataFrame()
    df1= pd.DataFrame()
    for i in range(477): 
        wrd = input_list[i]
        identity,position= input_sim1(wrd)
        word_data = pd.DataFrame([wrd])
        df2 = df2.append(word_data)
        identity_data = pd.DataFrame([identity])
        df = df.append(identity_data)       
        position_data = pd.DataFrame([position])
        df1 = df1.append(position_data)  
        df1.replace(np.nan, '0', regex=True, inplace= True)
    start = 1
    df2.columns = range(start, start + len(df2.columns))
    df2.index = np.arange(1, len(df2)+1)
    df2.to_excel('SubL+Allogx.xlsx', sheet_name = 'word')

    df.columns = range(start, start + len(df.columns))
    df.index = np.arange(1, len(df)+1)
    append_df_to_excel('SubL+Allogx.xlsx',df, sheet_name='allog')  
    
    df1.columns = range(start, start + len(df1.columns))
    df1.index = np.arange(1, len(df1)+1)
    append_df_to_excel('SubL+Allogx.xlsx',df1, sheet_name='position')
