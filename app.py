import pandas as pd
import streamlit as st
from PIL import Image
from model import open_data, preprocess_data, split_data, load_model_and_predict

def process_main_page():
    show_main_page()
    process_side_bar_inputs()


def show_main_page():
    image = Image.open('data/churn.png')

    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Churn Prediction (Month to Month contract)",
        page_icon=image,

    )
    st.write(
        """
        ## Churn value classification for customers with a Month to Month contract type.
        Let's check the probability of the customer's churn value.
        """
    )

    st.image(image)


def write_user_data(df):
    st.write("## Customer's information")
    st.write(df)


def write_prediction(prediction, prediction_probas):
    st.write("## Prediction")
    st.write(prediction)

    st.write("## Prediction probability")
    st.write(prediction_probas)

def process_side_bar_inputs():
    st.sidebar.header("Customer's parameters")
    user_input_df = sidebar_input_features()

    train_df = open_data()
    train_X_df, _ = split_data(train_df)
    full_X_df = pd.concat((user_input_df, train_X_df), axis=0)
    preprocessed_X_df = preprocess_data(full_X_df, test=False)

    user_X_df = preprocessed_X_df[:1]
    write_user_data(user_X_df)

    prediction, prediction_probas = load_model_and_predict(user_X_df)
    write_prediction(prediction, prediction_probas)

def sidebar_input_features():
    monthly_charge = st.sidebar.slider("Choose a Monthly Charge of a customer", min_value=15, max_value=120, value=20,
                            step=5)
    total_charges = st.sidebar.slider("Choose Total Charges of a customer", min_value=15, max_value=8000, value=20,
                            step=5)
    senior = st.sidebar.selectbox("Is the customer 65 or older?", ("Yes", "No"))
    dependents = st.sidebar.selectbox("Does the customer live with any dependents? (children, parents, grandparents, etc.)", ("Yes", "No"))
    married = st.sidebar.selectbox("Is the customer married?", ("Yes", "No"))
    online_security = st.sidebar.selectbox("Does the customer use Online Security service?", ("Yes", "No"))


    translatetion = {
        "Yes": 1,
        "No": 0
    }

    data = {
        "Monthly Charge": monthly_charge,
        "Total Charges": total_charges,
        "Senior Citizen_Yes": translatetion[senior],
        "Married_Yes": translatetion[married],
        "Dependents_Yes": translatetion[dependents],
        "Online Security_Yes": translatetion[online_security]}

    df = pd.DataFrame(data, index=[0])

    return df


if __name__ == "__main__":
    process_main_page()
