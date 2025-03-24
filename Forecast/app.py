import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet
import os

# **Streamlit App Configuration**
st.set_page_config(page_title="Revenue Forecasting", page_icon="📈", layout="wide")

# **UI Title**
st.title("📊 AI-Driven Revenue Forecasting with Prophet")

# **Upload Excel File**
uploaded_file = st.file_uploader("Upload an Excel file with Date and Revenue columns", type=["xlsx", "xls"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, parse_dates=["Date"])
    df = df.rename(columns={"Date": "ds", "Revenue": "y"})  # Prophet requires 'ds' and 'y' columns

    # **Initialize and Fit Prophet Model**
    model = Prophet()
    model.fit(df)

    # **Create Future Dataframe**
    future = model.make_future_dataframe(periods=30)  # Forecasting 30 days ahead
    forecast = model.predict(future)

    # **Plot Forecast Results**
    st.subheader("🔮 Forecast Results")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    # **Show Forecast Data**
    st.subheader("📋 Forecasted Data")
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30))

    # **Plot Forecast Components**
    st.subheader("📈 Forecast Components")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

else:
    st.info("📤 Please upload an Excel file to generate forecasts.")
