from pathlib import Path
import json
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = Path("data")

with open(DATA_DIR / "train_core_vs_neg.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open(DATA_DIR / "test_core_vs_neg.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

X_train_texts = [t for (t, y) in train_data]
y_train = [y for (t, y) in train_data]

X_test_texts = [t for (t, y) in test_data]
y_test = [y for (t, y) in test_data]

vectorizer = TfidfVectorizer(lowercase=True, min_df=5, max_df=0.9)

X_train = vectorizer.fit_transform(X_train_texts)
X_test = vectorizer.transform(X_test_texts)
clf = LogisticRegression(l1_ratio=1, solver="liblinear", max_iter=2000)

clf.fit(X_train, y_train)

# test set predictions
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)
print("\nClassification report:")
print(classification_report(y_test, y_pred))

###ROC AUC
auc = roc_auc_score(y_test, y_prob)
print("ROC AUC:", round(auc, 3))


# Model sparsity diagnostic
coef = clf.coef_[0]
nonzero = np.count_nonzero(coef)

print("\nModel sparsity:")
print("Total coefficients:", len(coef))
print("Non-zero coefficients:", nonzero)

# Get feature names
feature_names = vectorizer.get_feature_names_out()


# Top 15 positive-weight words (CORE = 1)
top_positive = np.argsort(coef)[-15:][::-1]

print("\nTop 15 positive-weight words (CORE):")
for idx in top_positive:
    print(f"{feature_names[idx]:20s} {coef[idx]:.4f}")


# Top 15 negative-weight words (NEG = 0)
# Exclude coefficients that L1 regularization set to zero
negative_indices = np.where(coef < 0)[0]

top_negative = negative_indices[np.argsort(coef[negative_indices])[:15]]

print("\nTop 15 negative-weight words (NEG):")
for idx in top_negative:
    print(f"{feature_names[idx]:20s} {coef[idx]:.4f}")

print(f"Total negative-weight words: {len(negative_indices)}")


# Inspect misclassified test examples
false_positives = [
    i
    for i, (actual, predicted) in enumerate(zip(y_test, y_pred))
    if actual == 0 and predicted == 1
]

false_negatives = [
    i
    for i, (actual, predicted) in enumerate(zip(y_test, y_pred))
    if actual == 1 and predicted == 0
]

print("\nFalse positives:", len(false_positives))
print("False negatives:", len(false_negatives))

# Select up to 10 misclassified examples
selected = false_positives[:5] + false_negatives[:5]

remaining = false_positives[5:] + false_negatives[5:]
selected += remaining[: max(0, 10 - len(selected))]

print("\n--- Misclassified chunks for review ---")

for i in selected:
    print("\n" + "=" * 60)
    print(f"Example index: {i}")
    print(f"True label: {y_test[i]}")
    print(f"Predicted label: {y_pred[i]}")
    print(f"Text:\n{X_test_texts[i]}")
