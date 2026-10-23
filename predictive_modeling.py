
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Step 1: Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Step 2: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Train Decision Tree
model1 = DecisionTreeClassifier(random_state=42)
model1.fit(X_train, y_train)
pred1 = model1.predict(X_test)

# Step 4: Train Random Forest
model2 = RandomForestClassifier(n_estimators=100, random_state=42)
model2.fit(X_train, y_train)
pred2 = model2.predict(X_test)

# Step 5: Display accuracy
print("Decision Tree Accuracy:",
      accuracy_score(y_test, pred1) * 100, "%")

print("Random Forest Accuracy:",
      accuracy_score(y_test, pred2) * 100, "%")

# Step 6: Display confusion matrix
cm = confusion_matrix(y_test, pred2)
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=data.target_names
)
display.plot()
plt.title("Random Forest Confusion Matrix")
plt.show()
