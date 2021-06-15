import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, accuracy_score
from preprocess import *
from sklearn.metrics import classification_report

categories = ['اجتماعی', 'اديان', 'اقتصادی', 'سیاسی', 'فناوري', 'مسائل راهبردي ايران', 'ورزشی']
categories_number = ['1', '2', '3', '4', '5', '6', '7']

train_data, test_data = make_test_and_train()


# naive_bayes_train_prediction
result_combine = pandas.concat([train_data, test_data])
# X_train, X_test, y_train, y_test = train_test_split(train_data['text'], train_data['class'])

count_vect = CountVectorizer(max_features=500)
count_vect.fit(result_combine['text'])
X_train_counts = count_vect.transform(train_data['text'])

tfidf_transformer = TfidfTransformer(use_idf=True, smooth_idf=True)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB(alpha=1).fit(X_train_tfidf, train_data['class'])

test = count_vect.transform(test_data['text'])
test = tfidf_transformer.transform(test)

pred = clf.predict(test)

print(count_vect.get_feature_names())


# report
def print_report():
    global test_data, pred, categories_number
    print('Naive Bayes Confusion Matrix:\n', confusion_matrix(test_data['class'], pred), end='\n\n')
    print('Naive Bayes Classification Report:\n',
          classification_report(test_data['class'], pred), end='\n\n')
    print('Naive Bayes accuracy:\n',
          accuracy_score(test_data['class'], pred), end='\n\n')

    # plot the confusion matrix
    mat = confusion_matrix(test_data['class'], pred)
    sns.heatmap(mat.T, square=True, annot=True, fmt="d", xticklabels=categories_number, yticklabels=categories_number)
    plt.xlabel("true labels")
    plt.ylabel("predicted label")
    plt.show()


print_report()
