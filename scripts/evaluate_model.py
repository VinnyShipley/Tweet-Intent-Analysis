#from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix
import numpy as np
import pandas as pd



def prepare_features_with_extras(X_train, X_test, extra_features_dict=None):
    
    # TF_DIF
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    extras_train = []
    extras_test = []

    if extra_features_dict:
        for feature_name, full_series in extra_features_dict.items():
            train_series = full_series.loc[X_train.index].values.reshape(-1, 1)
            test_series = full_series.loc[X_test.index].values.reshape(-1, 1)

            scaler = MinMaxScaler()
            train_scaled = scaler.fit_transform(train_series)
            test_scaled = scaler.transform(test_series)

            extras_train.append(csr_matrix(train_scaled))
            extras_test.append(csr_matrix(test_scaled))

    X_train_combined = hstack([X_train_tfidf] + extras_train)
    X_test_combined = hstack([X_test_tfidf] + extras_test)

    return X_train_combined, X_test_combined



def evaluate_model_fixed_split(df, X_train, X_test, y_train, y_test, extra_feature_names=[], model=None):

    if model is None:
        model = LogisticRegression(max_iter=1000)

    # Prepare extra features
    feature_dict = {name: df[name] for name in extra_feature_names}
    X_train_vec, X_test_vec = prepare_features_with_extras(
        X_train, X_test, extra_features_dict=feature_dict
    )

    # Fit model
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    # Get confidence score if supported
    confidence_score = (
        np.max(model.predict_proba(X_test_vec), axis=1)
        if hasattr(model, "predict_proba")
        else [None] * len(y_pred)
    )

    # Build evaluation dataframe
    eval_df = pd.DataFrame({
        "tweet_text": X_test,
        "true_intent": y_test,
        "predicted_intent": y_pred,
        "confidence_score": confidence_score,
    })

    eval_df["correct"] = eval_df["true_intent"] == eval_df["predicted_intent"]

    # Include all features; if not used, fill with NaN
    all_possible_features = ['tweet_length', 'is_question', 'sentiment_score']
    for feature in all_possible_features:
        if feature in df.columns:
            if feature in extra_feature_names:
                eval_df[feature] = df.loc[X_test.index, feature].values
            else:
                eval_df[feature] = [np.nan] * len(X_test)
        else:
            eval_df[feature] = [np.nan] * len(X_test)
    
    print("Extra features in test set:")
    for feature in extra_feature_names:
        print(f"{feature} sample values:", df.loc[X_test.index, feature].head())



    return eval_df











 