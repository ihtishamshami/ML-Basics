from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


iris = datasets.load_iris()
X,y = iris.data, iris.target

X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size=0.2, random_state=42)

para_grid = {'C': [0.1, 1, 10, 100],'gamma': [0.01, 0.1, 1, 10],'kernel': ['linear', 'rbf', 'poly']}

svm_classifier = SVC()
grid_search = GridSearchCV(svm_classifier, para_grid, cv=5, scoring='accuracy')

grid_search.fit(X_train, Y_train)
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
Y_pred = best_model.predict(X_test)

best_score = grid_search.best_score_
print(f"Best Parameters: {best_params}")
print("Accuracy on test set", accuracy_score(Y_test, Y_pred))