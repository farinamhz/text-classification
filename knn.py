import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, accuracy_score
from preprocess import *

categories_number = ['1', '2', '3', '4', '5', '6', '7']

train_data, test_data = make_test_and_train()

# KNN_train_prediction

count_vect = CountVectorizer(max_features=500)
X_train_counts = count_vect.fit_transform(train_data['text'])

tfidf_transformer = TfidfTransformer(use_idf=True, smooth_idf=True)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf1 = KNeighborsClassifier(n_neighbors=1, metric='minkowski', p=2).fit(X_train_tfidf, train_data['class'])
clf2 = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2).fit(X_train_tfidf, train_data['class'])
clf3 = KNeighborsClassifier(n_neighbors=15, metric='minkowski', p=2).fit(X_train_tfidf, train_data['class'])

test = count_vect.transform(test_data['text'])
test = tfidf_transformer.transform(test)

pred1 = clf1.predict(test)
pred2 = clf2.predict(test)
pred3 = clf3.predict(test)


# report
def print_report(name, pred):
    global test_data, categories_number
    print(name)
    print('KNN Confusion Matrix:\n', confusion_matrix(test_data['class'], pred), end='\n\n')
    print('KNN accuracy:\n',
          accuracy_score(test_data['class'], pred), end='\n\n')

    # plot the confusion matrix
    mat = confusion_matrix(test_data['class'], pred)
    sns.heatmap(mat.T, square=True, annot=True, fmt="d", xticklabels=categories_number, yticklabels=categories_number)
    plt.xlabel("true labels" + name)
    plt.ylabel("predicted label" + name)
    plt.show()


print_report("With k = 1", pred1)
print_report("With k = 5", pred2)
print_report("With k = 15", pred3)
