import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, accuracy_score
from preprocess import *
from sklearn.metrics import classification_report

categories = ['اجتماعی', 'اديان', 'اقتصادی', 'سیاسی', 'فناوري', 'مسائل راهبردي ايران', 'ورزشی']
categories_number = ['1', '2', '3', '4', '5', '6', '7']

train_data, test_data = make_test_and_train()

# pipeline = Pipeline([
#     ('tfidf', TfidfTransformer(use_idf=True, smooth_idf=True)),
#     ('clf', MultinomialNB(alpha=1))
# ])



# count_vect = CountVectorizer(max_features=500)
# X_train_counts = count_vect.fit_transform(train_data['text'])
# print(count_vect.get_feature_names())

# model1 = TfidfVectorizer(max_features=500, use_idf=True, smooth_idf=True)
# x_train_tfidf = model1.fit(result_combine)
# model2 = MultinomialNB(alpha=1).fit(x_train_tfidf, train_data['class'])
# pred = model2.predict(test_data['text'])

# naive_bayes_train_prediction
result_combine = pandas.concat([train_data, test_data])

count_vect = CountVectorizer(max_features=500)
# X_train_counts = count_vect.fit_transform(result_combine['text'])
X_train_counts = count_vect.fit_transform(train_data['text'])
print(count_vect.get_feature_names())

tfidf_transformer = TfidfTransformer(use_idf=True, smooth_idf=True)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB(alpha=1).fit(X_train_tfidf, train_data['class'])

test = count_vect.transform(test_data['text'])
test = tfidf_transformer.transform(test)

pred = clf.predict(test)
# model = make_pipeline(TfidfVectorizer(max_features=500, use_idf=True, smooth_idf=True),
#                       MultinomialNB(alpha=1))
# model.fit(train_data['text'], train_data['class'])
# pred = model.predict(test_data['text'])


# report
def print_report():
    global test_data, pred, categories_number
    print('Naive Bayes Confusion Matrix:\n', confusion_matrix(test_data['class'], pred), end='\n\n')
    print('Naive Bayes Classification Report:\n',
          classification_report(test_data['class'], pred), end='\n\n')
    print('Naive Bayes accuracy:\n',
          accuracy_score(test_data['class'], pred), end='\n\n')
    # print(list(test_data['class']))
    # print(list(pred))

    # plot the confusion matrix
    mat = confusion_matrix(test_data['class'], pred)
    sns.heatmap(mat.T, square=True, annot=True, fmt="d", xticklabels=categories_number, yticklabels=categories_number)
    plt.xlabel("true labels")
    plt.ylabel("predicted label")
    plt.show()


print_report()
