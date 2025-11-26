import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# ----------------- Load model & artifacts (cached) -----------------

@st.cache_resource
def load_model_and_artifacts():
    model = tf.keras.models.load_model('model.h5')

    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)

    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)

    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    return model, label_encoder_gender, onehot_encoder_geo, scaler


model, label_encoder_gender, onehot_encoder_geo, scaler = load_model_and_artifacts()

# ----------------- Streamlit UI -----------------

st.title('Customer Churn Prediction')

st.write("Fill the details below and click **Predict churn** to see the probability.")

# User inputs
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92, 40)
balance = st.number_input('Balance', value=0.0)
credit_score = st.number_input('Credit Score', value=600.0)
estimated_salary = st.number_input('Estimated Salary', value=50000.0)
tenure = st.slider('Tenure (years with bank)', 0, 10, 3)
num_of_products = st.slider('Number of Products', 1, 4, 1)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# ----------------- Prediction button -----------------

if st.button('Predict churn'):
    try:
        # 1) Prepare numerical part
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [label_encoder_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary]
        })

        # 2) One-hot encode Geography
        geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
        geo_cols = onehot_encoder_geo.get_feature_names_out(['Geography'])
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_cols)

        # 3) Combine both
        input_data = pd.concat(
            [input_data.reset_index(drop=True), geo_encoded_df],
            axis=1
        )

        st.write("Debug: Features going into scaler/model:")
        st.dataframe(input_data)

        # 4) Scale
        input_data_scaled = scaler.transform(input_data)

        # 5) Predict
        prediction = model.predict(input_data_scaled)
        prediction_proba = float(prediction[0][0])

        st.subheader("Result")
        st.write(f'**Churn Probability:** {prediction_proba:.2%}')

        if prediction_proba > 0.5:
            st.error('The customer is likely to churn.')
        else:
            st.success('The customer is not likely to churn.')

    except Exception as e:
        st.error(f"❌ Error while predicting: {e}")
        st.info("Check that model, scaler, and encoders were trained on the same feature order.")
