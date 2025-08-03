import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

st.title("Price Prediction App")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview", data.head())

    columns = data.columns.tolist()
    target = st.selectbox("Select target column (price)", columns)
    features = st.multiselect("Select feature columns", [col for col in columns if col != target])

    model_choice = st.selectbox("Choose model", ["Linear Regression", "SVR"])

    if features and target:
        X = data[features].values
        y = data[target].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if model_choice == "Linear Regression":
            model = LinearRegression()
        else:
            model = SVR(kernel='rbf')

        model.fit(X_scaled, y)
        preds = model.predict(X_scaled)
        rmse = np.sqrt(mean_squared_error(y, preds))
        st.metric("Training RMSE", f"{rmse:.2f}")

        st.subheader("Enter feature values for prediction")
        input_data = []
        for feat in features:
            val = st.number_input(f"{feat}", value=0.0)
            input_data.append(val)

        if st.button("Predict"):
            input_scaled = scaler.transform([input_data])
            result = model.predict(input_scaled)[0]
            st.success(f"Predicted Price: {result:.2f}")