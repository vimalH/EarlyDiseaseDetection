from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn import tree,svm
from sklearn.metrics import accuracy_score,classification_report
from sklearn.linear_model import Ridge
# from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.preprocessing import StandardScaler



tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
					
import os
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_confusion_matrix(cm, classes=['Anthracnose','Downey Mildew','Early Blight','Blight','Rust','Spot','Normal','Powdery Mildew'],
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



my_data = np.genfromtxt('datafile.csv',delimiter=',')
X_train,X_test,y_train,y_test = train_test_split(my_data[1:,:my_data.shape[1]-1],my_data[1:,my_data.shape[1]-1],test_size=0.33,random_state=1) #change test_size to increase/decrease test sample size
#normalize the dataset
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

clf_dtree = tree.DecisionTreeClassifier(max_depth=25,random_state=True)
clf_dtree.fit(X_train,y_train)
prediction = clf_dtree.predict(X_test)
print("Accuracy of decision tree is:",accuracy_score(y_test, prediction))
print("Classification report for decision tree:\n",classification_report(y_test, prediction))
cnf_matrix = confusion_matrix(y_test, prediction)
print(cnf_matrix)
plot_confusion_matrix(cnf_matrix,title="Decision tree confusion matrix")
plt.show()

clf = svm.LinearSVC(C=0.5,random_state=True,multi_class='crammer_singer')
# clf = svm.SVC(gamma=2,C=100,random_state=True)
clf.fit(X_train,y_train)
prediction = clf.predict(X_test)
print("LinearSVC accuracy: ",accuracy_score(y_test, prediction))
print("Classification report for svm-linear svc:\n",classification_report(y_test, prediction))
cnf_matrix = confusion_matrix(y_test, prediction)
print(cnf_matrix)
plot_confusion_matrix(cnf_matrix,title="Linear SVC confusion matrix")
plt.show()


clf = svm.SVC(gamma=2, C=1)
clf.fit(X_train,y_train)
# clf = svm.SVC(gamma=2,C=100,random_state=True)
# clf.fit(X_train,y_train)
prediction = clf.predict(X_test)
print("SVM accuracy: ",accuracy_score(y_test, prediction))
print("Classification report for SVM:\n",classification_report(y_test, prediction))
cnf_matrix = confusion_matrix(y_test, prediction)
print(cnf_matrix)
plot_confusion_matrix(cnf_matrix,title="SVM confusion matrix")
plt.show()
