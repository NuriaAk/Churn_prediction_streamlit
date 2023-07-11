from pickle import dump, load
import xgboost as xgb
from sklearn.metrics import accuracy_score
import pandas as pd

def split_data(df: pd.DataFrame):
    y = df['Churn Value']
    X = df.drop(['Churn Value'], axis = 1)

    return X, y

def open_data(path="data/Churn  prediction project_final_df.csv"):
    df = pd.read_csv(path)
    df = df.drop(labels = ["Premium Tech Support_Yes", "Paperless Billing_Yes",
        "Payment Method_Bank Withdrawal"], axis = 1)
    return df

def preprocess_data(df: pd.DataFrame, test=True):
    df.dropna(inplace=True)

    if test:
        X_df, y_df = split_data(df)
    else:
        X_df = df

    if test:
        return X_df, y_df
    else:
        return X_df

def fit_and_save_model(X_df, y_df, path="model_weights.mw"):
    model =xgb.XGBClassifier(max_depth = 2)
    model.fit(X_df, y_df)

    test_prediction = model.predict(X_df)
    accuracy = accuracy_score(test_prediction, y_df)
    print(f"Model accuracy is {accuracy}")

    with open(path, "wb") as file:
        dump(model, file)

    print(f"Model was saved to {path}")

def load_model_and_predict(df, path="model_weights.mw"):
    with open(path, "rb") as file:
        model = load(file)

    prediction = model.predict(df)[0]
    # prediction = np.squeeze(prediction)

    prediction_proba = model.predict_proba(df)[0]
    # prediction_proba = np.squeeze(prediction_proba)

    encode_prediction_proba = {
        0: "Not churn with a probability:",
        1: "Churn with a probability:"
    }

    encode_prediction = {
        0: "The customer will not churn!",
        1: "The customer will churn ... oops"
    }


    prediction_data = {}
    for key, value in encode_prediction_proba.items():
        prediction_data.update({value: prediction_proba[key]})

    prediction_df = pd.DataFrame(prediction_data, index=[0])
    prediction = encode_prediction[prediction]

    return prediction, prediction_df


if __name__ == "__main__":
    df = open_data()
    X_df, y_df = preprocess_data(df)
    fit_and_save_model(X_df, y_df)
