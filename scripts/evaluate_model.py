from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.parse import hstack, csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def prepare_features_with_extras(X_train, X_test, extra_train_df, extra_test_df, extra_cols):
    
    # TF_DIF
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    extras_train = []
    extras_test = []

    for col in extra_cols:
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(extra_train_df[[col]])
        test_scaled = scaler.transform(extra_test_df[[col]])


        extras_train.append(csr_matrix(train_scaled))
        extras_test.append(csr_matrix(test_scaled))
    
    X_train_combined = hstack([X_train_tfidf] + extras_train)
    X_test_combined = hstack([X_test_tfidf] + extras_test)

    return X_train_combined, X_test_combined




def evaluate_model(y_true, y_pred, labels=None, show_confusion=True, return_dict=False):
    
    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {acc:.4f}\n")

    report = classification_report(y_true, y_pred, labels=labels, output_dict=return_dict, zero_division=0)
    print("Classification Report:\n")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    if show_confusion:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    if return_dict:
        return {"accuracy": acc, "report": report}
 