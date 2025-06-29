import os
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ========== Custom load_data from iris.csv/xlsx ==========
def load_data(file_path="iris.xlsx"):  # or "irisdata.xlsx"
    if file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)

    X = df.iloc[:, :-1]  # all features
    y = df.iloc[:, -1]   # last column is label
    return X, y

# ========== Tests on Your Actual Data ==========
def test_data_shape():
    X, _ = load_data()
    print("Data shape:", X.shape)
    assert X.shape[1] == 4, "Expected 4 features"

def test_model_accuracy():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy on test data:", acc)
    assert acc > 0.8, "Model accuracy should be above 80%"

# ========== Check Output of train.py ==========
def test_model_artifact():
    assert os.path.exists("model.joblib"), "model.joblib file not found."

def test_metrics_accuracy():
    assert os.path.exists("metrics.csv"), "metrics.csv file not found."
    df = pd.read_csv("metrics.csv")
    acc = df.loc[df['metric'] == 'accuracy', 'value'].values[0]
    assert float(acc) > 0.8, "Reported accuracy must be > 80%"
