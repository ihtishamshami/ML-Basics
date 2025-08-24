from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()
X , Y = iris.data ,iris.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, Y_train) 

Y_pred = model.predict(X_test)
  
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy of the Decision Tree Classifier: {accuracy:.2f}")
