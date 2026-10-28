import arabic_reshaper
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np

def input_sim1(input_word):
    input_word = arabic_reshaper.reshape(input_word)
    t = Tokenizer()
    fit_text = "ء أ ـأ ؤ ـؤ إ ـإ ئ ـئ ـئـ ئـ ا ـا ب ـب ـبـ بـ ت ـت ـتـ تـ ة ـة ث ـث ـثـ ثـ ج ـج ـجـ جـ ح ـح ـحـ حـ خ ـخ ـخـ خـ د ـد ذ ـذ ر ـر ز ـز س ـس ـسـ سـ ش ـش ـشـ شـ ص ـص ـصـ صـ ض ـض ـضـ ضـ ط ـط ـطـ طـ ظ ـظ ـظـ ظـ ع ـع ـعـ عـ غ ـغ ـغـ غـ ف ـف ـفـ فـ ق ـق ـقـ قـ ك ـك ـكـ كـ ل ـل ـلـ لـ م ـم ـمـ مـ ن ـن ـنـ نـ ه ـه ـهـ هـ و ـو ى ـى ي ـي ـيـ يـ"
    fit_text = arabic_reshaper.reshape(fit_text)
    t.fit_on_texts(fit_text)
    sequences = t.texts_to_sequences(input_word)
    identity = []
    flattened = [val for sublist in sequences for val in sublist]
    for key in t.word_index.values():
        if key in flattened:
            identity.append('1')
        else:
            identity.append('0')
    return identity, flattened

if __name__ == '__main__':

    word_list = ['الافنصالية', 'الإلقيمية', 'البرحينية', 'التعميلات', 'التفريسية', 'ينذفها', 'ينذقون', 'ينركها', 'يسدعك', 'يشرعك']





    df_word     = pd.DataFrame({'word': word_list})
    df_allog    = pd.DataFrame()
    df_position = pd.DataFrame()

    for wrd in word_list:
        identity, position = input_sim1(wrd)
        df_allog    = pd.concat([df_allog,    pd.DataFrame([identity])], ignore_index=True)
        df_position = pd.concat([df_position, pd.DataFrame([position])], ignore_index=True)

    df_allog.replace(np.nan, '0', regex=True, inplace=True)

    start = 1
    for df in [df_allog, df_position]:
        df.columns = range(start, start + len(df.columns))
        df.index   = np.arange(1, len(df) + 1)

    output_file = 'tests/test_allograph_encoding.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_word.to_excel(    writer, sheet_name='word',     index=True)
        df_allog.to_excel(   writer, sheet_name='allog',    index=True)
        df_position.to_excel(writer, sheet_name='position', index=True)

    print(f"Saved to {output_file}")