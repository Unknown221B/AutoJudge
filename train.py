import json
import pandas as pd
import joblib
from scipy.sparse import hstack

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from features import handcrafted_features


def load_data(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)


df = load_data("data/problems_data.jsonl")

df["combined_text"] = (
    df["title"].fillna("") + " " +
    df["description"].fillna("") + " " +
    df["input_description"].fillna("") + " " +
    df["output_description"].fillna("")
)

y_class = df["problem_class"]
y_score = df["problem_score"]

X = df["combined_text"]

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_text = tfidf.fit_transform(X)
X_hand = handcrafted_features(X)
X_final = hstack([X_text, X_hand])

X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test = train_test_split(
    X_final, y_class, y_score, test_size=0.2, random_state=42
)

print("\n--- Classification Models ---")

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

for name, model in classifiers.items():
    model.fit(X_train, y_class_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_class_test, preds)
    print(f"{name} Accuracy: {acc:.4f}")

final_clf = RandomForestClassifier(n_estimators=200, random_state=42)
final_clf.fit(X_train, y_class_train)

best_preds = final_clf.predict(X_test)
print("\nConfusion Matrix (Best Classifier):")
print(confusion_matrix(y_class_test, best_preds))

print("\n--- Regression Models ---")

regressors = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

for name, model in regressors.items():
    model.fit(X_train, y_score_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_score_test, preds)
    rmse = mean_squared_error(y_score_test, preds) ** 0.5
    print(f"{name} | MAE: {mae:.3f} | RMSE: {rmse:.3f}")

final_reg = RandomForestRegressor(n_estimators=200, random_state=42)
final_reg.fit(X_train, y_score_train)

joblib.dump(tfidf, "tfidf.pkl")
joblib.dump(final_clf, "classifier.pkl")
joblib.dump(final_reg, "regressor.pkl")

print("\nFinal models saved successfully.")

final_class_preds = final_clf.predict(X_test)
print("Final Classification Accuracy:",
      accuracy_score(y_class_test, final_class_preds))
print("Final Confusion Matrix:")
print(confusion_matrix(y_class_test, final_class_preds))

final_score_preds = final_reg.predict(X_test)
print("Final Regression MAE:",
      mean_absolute_error(y_score_test, final_score_preds))
print("Final Regression RMSE:",
      mean_squared_error(y_score_test, final_score_preds) ** 0.5)