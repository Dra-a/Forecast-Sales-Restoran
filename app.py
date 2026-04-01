import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from datetime import timedelta
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Revenue Forecasting App", layout="wide")
st.title("💰 Revenue Forecasting Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("processed_sales.csv", parse_dates=['date'])
    df = df.sort_values('date')
    df.set_index('date', inplace=True)
    return df

@st.cache_resource
def load_model():
    with open("best_sarimax_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

try:
    df = load_data()
    model = load_model()
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

st.sidebar.header("Forecast Settings")
forecast_days = st.sidebar.slider("Days to forecast into the future", min_value=1, max_value=30, value=7)

target_column = 'total_transaction_amount'
quantity_column = 'total_quantity_all_items' # We need this to calculate the proper lags!

st.sidebar.write(f"**Predicting:** `{target_column}`")
st.sidebar.write(f"**Using Exog Lags from:** `{quantity_column}`")
st.subheader(f"Forecast for the next {forecast_days} days")

try:
    exog_cols = model.model.exog_names
    
    # Extract full history, matching exactly how the model was trained
    full_df = pd.concat([df[target_column], df[exog_cols]], axis=1).dropna()
    full_endog = full_df[target_column].copy()
    full_exog = full_df[exog_cols].copy()

    forecast_mean = []
    lower_bounds = []
    upper_bounds = []
    future_dates = []

    # Trackers
    temp_revenue = df[target_column].copy()
    temp_quantity = df[quantity_column].copy() # Used STRICTLY to calculate lags
    
    with st.spinner('Synchronizing model states and generating dynamic forecast...'):
        for i in range(forecast_days):
            next_date = temp_revenue.index[-1] + timedelta(days=1)
            future_dates.append(next_date)
            
            # --- CALCULATE Lags based on QUANTITY, not Revenue! ---
            lag_1 = temp_quantity.iloc[-1]
            lag_7 = temp_quantity.iloc[-7]
            rolling_mean_7_days = temp_quantity.iloc[-7:].mean()
            is_weekend = int(next_date.dayofweek in [5, 6])
            
            future_exog = pd.DataFrame({
                'lag_1': [lag_1],
                'lag_7': [lag_7],
                'rolling_mean_7_days': [rolling_mean_7_days],
                'is_weekend': [is_weekend]
            }, index=[next_date])
            
            # --- APPLY MODEL ---
            current_model = model.apply(full_endog, exog=full_exog, refit=False)
            
            # --- GET FORECAST ---
            forecast = current_model.get_forecast(steps=1, exog=future_exog)
            pred_revenue = max(0, forecast.predicted_mean.iloc[0]) 
            
            conf_int = forecast.conf_int().iloc[0]
            lower_bound = max(0, conf_int.iloc[0])
            upper_bound = max(0, conf_int.iloc[1])
            
            forecast_mean.append(pred_revenue)
            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)
            
            # --- UPDATE HISTORIES ---
            temp_revenue.loc[next_date] = pred_revenue
            
            # CRITICAL STEP: We need a future quantity to calculate tomorrow's lag_1. 
            # We assume a "seasonal naive" approach: tomorrow's quantity will match the quantity from 7 days ago.
            assumed_next_quantity = temp_quantity.iloc[-7]
            temp_quantity.loc[next_date] = assumed_next_quantity
            
            new_endog = pd.Series([pred_revenue], index=[next_date])
            full_endog = pd.concat([full_endog, new_endog])
            full_exog = pd.concat([full_exog, future_exog])
            
    # Final dataframe
    forecast_df = pd.DataFrame({
        'Predicted Revenue ($)': forecast_mean,
        'Lower Bound (95% CI)': lower_bounds,
        'Upper Bound (95% CI)': upper_bounds
    }, index=future_dates)
    
    st.write(forecast_df.round(2))
    
    # Plotting
    st.subheader("Historical vs. Forecast Plot")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    historical_subset = df[target_column].iloc[-30:]
    ax.plot(historical_subset.index, historical_subset.values, label='Historical Actual Revenue', marker='o', color='blue')
    ax.plot(forecast_df.index, forecast_df['Predicted Revenue ($)'], color='green', label='Forecasted Revenue', marker='o')
    ax.fill_between(
        forecast_df.index, 
        forecast_df['Lower Bound (95% CI)'], 
        forecast_df['Upper Bound (95% CI)'], 
        color='lightgreen', alpha=0.3, label='95% Confidence Interval'
    )
    
    ax.set_title("Revenue Forecast using SARIMAX (Quantity-Weighted Lags)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Transaction Amount ($)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    st.pyplot(fig)

except Exception as e:
    st.error(f"An error occurred while forecasting: {e}")