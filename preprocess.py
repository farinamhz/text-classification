import codecs
import glob
from pandas import *
# from hazm import Normalizer, Stemmer
from parsivar import Normalizer

SOURCES = [
    ('Final_Dataset\Train\اجتماعی\*.txt', 'اجتماعی'),
    ('Final_Dataset\Train\اديان\*.txt', 'اديان'),
    ('Final_Dataset\Train\اقتصادی\*.txt', 'اقتصادی'),
    ('Final_Dataset\Train\سیاسی\*.txt', 'سیاسی'),
    ('Final_Dataset\Train\فناوري\*.txt', 'فناوري'),
    ('Final_Dataset\Train\مسائل راهبردي ايران\*.txt', 'مسائل راهبردي ايران'),
    ('Final_Dataset\Train\ورزشی\*.txt', 'ورزشی'),
]
SOURCES_test = [
    ('Final_Dataset\Test\اجتماعی\*.txt', 'اجتماعی'),
    ('Final_Dataset\Test\اديان\*.txt', 'اديان'),
    ('Final_Dataset\Test\اقتصادی\*.txt', 'اقتصادی'),
    ('Final_Dataset\Test\سیاسی\*.txt', 'سیاسی'),
    ('Final_Dataset\Test\فناوري\*.txt', 'فناوري'),
    ('Final_Dataset\Test\مسائل راهبردي ايران\*.txt', 'مسائل راهبردي ايران'),
    ('Final_Dataset\Test\ورزشی\*.txt', 'ورزشی'),
]

'''''
def fetch_stop_words():
    nmz = Normalizer()
    f = open('./persian', encoding='utf-8')
    words = f.read()
    return sorted(set([nmz.normalize(text = w) for w in words.split('\n') if w]))
'''''

def fetch_stop_words():
    nmz = Normalizer()
    stops1 = "\n".join(
        sorted(list(set([nmz.normalize(w) for w in codecs.open('persian', encoding='utf-8').read().split('\n') if w]))))
    stops2 = "\n".join(
        sorted(list(set([nmz.normalize(w) for w in codecs.open('verbal', encoding='utf-8').read().split('\n') if w]))))
    stops3 = "\n".join(
        sorted(list(set([nmz.normalize(w) for w in codecs.open('nonverbal', encoding='utf-8').read().split('\n') if w]))))

    return stops1+stops2+stops3


def remove_mystopwords(sentence):
    #remove punc
    for ch in "1234567890&;#$()*,-./:[]«»؛،؟۰۱۲۳۴۵۶۷۸۹!":
        sentence = sentence.replace(ch, "")
    #remove stopwords
    stopwords = fetch_stop_words()
    tokens = sentence.split(" ")
    tokens_filtered = [word for word in tokens if not word in stopwords]
    # for word in tokens_filtered:
    #     stemmer = Stemmer()
    #     stemmer.stem(word)
    return " ".join(tokens_filtered)

def read_files(path):
    files = glob.glob(path)
    for file in files:
        with codecs.open(file, "r", encoding='utf-8',
                         errors='ignore') as f:
            text = f.read()
            # print("1")
            # print(text)
            normalizer = Normalizer()
            normalizer.normalize(text)
            text = text.replace("nbsp", " ")
            text = text.replace("amp", " ")
            text = text.replace("ي", "ی")
            text = text.replace('ك', 'ک')
            text = text.replace("‌هایی ", " ")
            text = text.replace("‌ها ", " ")
            text = remove_mystopwords(text)
            text = text.replace('آن', ' ')
            text = text.replace(' این ', ' ')
            text = text.replace(' ریال ', ' ')
            text = text.replace('مى', ' ')
            text = text.replace('که', ' ')
            # print(2)
            # print(text)
            # print("FINIIIIIIISH")
            text = text.replace('\n', '')
            yield file, text


def build_df(path, label):
    rows = []
    index = []
    for file_name, text in read_files(path):
        rows.append({'text': text, 'class': label})
        index.append(file_name)
    df = DataFrame(rows, index=index)
    return df


def make_test_and_train():
    train_data = DataFrame({'text': [], 'class': []})
    for path, label in SOURCES:
        train_data = train_data.append(build_df(path, label))

    # train_data = train_data.reindex(numpy.random.permutation(train_data.index))

    test_data = DataFrame({'text': [], 'class': []})
    for path, label in SOURCES_test:
        test_data = test_data.append(build_df(path, label))

    '''''
    # save to excel
    writer = pandas.ExcelWriter('train.xlsx')
    train_data.to_excel(writer)
    writer.save()

    writer = pandas.ExcelWriter('test.xlsx')
    test_data.to_excel(writer)
    writer.save()
    '''''

    return train_data, test_data


# train_data, test_data = make_test_and_train()