import codecs
import glob
import string
from pandas import *
from hazm import Normalizer
import re

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

def clean_sentence(sentence):
    sentence = arToPersianChar(sentence)
    return sentence


def arToPersianChar(userInput):
    dic = {
        'ك': 'ک',
        'ي': 'ی'
    }
    return multiple_replace(dic, userInput)


def multiple_replace(dic, text):
    pattern = "|".join(map(re.escape, dic.keys()))
    return re.sub(pattern, lambda m: dic[m.group()], str(text))

'''''
def fetch_stop_words():
    nmz = Normalizer()
    f = open('./persian', encoding='utf-8')
    words = f.read()
    return sorted(set([nmz.normalize(text = w) for w in words.split('\n') if w]))
'''''

def fetch_stop_words():
    nmz = Normalizer()
    stops = "\n".join(
        sorted(
            list(
                set(
                    [
                        nmz.normalize(w) for w in codecs.open('persian', encoding='utf-8').read().split('\n') if w]))))
    return stops


def remove_mystopwords(sentence):
    #remove punc
    for ch in "1234567890&;#$()*,-./:[]«»؛،؟۰۱۲۳۴۵۶۷۸۹!":
        sentence = sentence.replace(ch, "")
    #remove stopwords
    stopwords = fetch_stop_words()
    tokens = sentence.split(" ")
    tokens_filtered = [word for word in tokens if not word in stopwords]
    return " ".join(tokens_filtered)

def read_files(path):
    files = glob.glob(path)
    for file in files:
        with codecs.open(file, "r", encoding='utf-8',
                         errors='ignore') as f:
            text = f.read()
            # print("1")
            # print(text)
            # text = clean_sentence(text)
            text = text.replace("nbsp", " ")
            text = text.replace("amp", " ")
            normalizer = Normalizer()
            normalizer.normalize(text=text)
            text = remove_mystopwords(text)

            # text = text.replace("ها", " ")
            # text = text.replace("های", " ")
            # text = text.replace("هایی", " ")

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