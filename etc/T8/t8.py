from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from sklearn import datasets

iris = datasets.load_iris()
iris_data = iris.data
iris_target = iris.target

x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_target, test_size=0.20)

classifer = KNeighborsClassifier(n_neighbors=5)
classifer.fit(x_train, y_train)

y_pred = classifer.predict(x_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))