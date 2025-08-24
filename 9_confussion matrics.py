import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score,recall_score, f1_score
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

X ,Y =  load_digits(return_X_y=True)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)

model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

cm = confusion_matrix(Y_test, Y_pred)
sns.heatmap(cm, 
            annot=True, 
            fmt='g')

plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()


