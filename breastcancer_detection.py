from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Load dataset
# ---------------------------
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Base Logistic Regression
# ---------------------------
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print("Base Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))

# ---------------------------
# Regularized Logistic Regression
# ---------------------------
lr_reg = LogisticRegression(penalty='l2', C=0.1, max_iter=1000)
cv_scores = cross_val_score(lr_reg, X, y, cv=5)
print("Regularized LR Cross-validation Accuracy:", cv_scores.mean())

# ---------------------------
# PCA (optional)
# ---------------------------
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)
print("PCA shape:", X_pca.shape)

# ---------------------------
# Decision Tree
# ---------------------------
tree = DecisionTreeClassifier(max_depth=4, min_samples_split=5, random_state=42)
tree.fit(X_train, y_train)
print("Decision Tree Accuracy:", tree.score(X_test, y_test))

# ---------------------------
# Random Forest
# ---------------------------
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
print("Random Forest Accuracy:", rf.score(X_test, y_test))

# ---------------------------
# XGBoost
# ---------------------------
xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4)
xgb.fit(X_train, y_train)
print("XGBoost Accuracy:", xgb.score(X_test, y_test))

# ---------------------------
# SMOTE + Logistic Regression
# ---------------------------
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

lr_smote = LogisticRegression(class_weight='balanced', max_iter=1000)
lr_smote.fit(X_res, y_res)

smote_pred = lr_smote.predict(X_test)
print("SMOTE LR Test Accuracy:", accuracy_score(y_test, smote_pred))
print(classification_report(y_test, smote_pred))

# ---------------------------
# Learning Curve
# ---------------------------
train_sizes, train_scores, test_scores = learning_curve(
    lr_smote, X, y, cv=5
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_mean, label='Train Accuracy')
plt.plot(train_sizes, test_mean, label='Validation Accuracy')
plt.legend()
plt.title("Learning Curve")
plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.show()
