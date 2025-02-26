# model.py

import joblib
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_and_save_model():
    # Load the iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Decision Tree Classifier
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate the model
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Model accuracy:", acc)

    # Save the trained model to a file
    joblib.dump(clf, 'iris_model.pkl')
    print("Model saved as iris_model.pkl")

if __name__ == "__main__":
    train_and_save_model()
