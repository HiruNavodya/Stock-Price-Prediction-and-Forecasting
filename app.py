import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import streamlit as st
from keras.models import load_model
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# Define the start and end dates
start = '2010-01-01'
end = '2019-12-31'

st.title("Stock Trend Prediction App")

# Dropdown to select a company
companies = {
    'Apple Inc. (AAPL)': 'AAPL',
    'Microsoft Corp. (MSFT)': 'MSFT',
    'Tesla Inc. (TSLA)': 'TSLA',
    'Amazon.com Inc. (AMZN)': 'AMZN',
    'Google (Alphabet) (GOOGL)': 'GOOGL',
    'Facebook Inc. (META)': 'META',
    'NVIDIA Corporation (NVDA)': 'NVDA',
    'Netflix Inc. (NFLX)': 'NFLX',
    'Alibaba Group (BABA)': 'BABA',
    'Walmart Inc. (WMT)': 'WMT'
}

# Select box to choose a company
selected_company = st.selectbox('Select a Company', list(companies.keys()))
user_input = companies[selected_company]

# Fetch stock data
df = yf.download(user_input, start=start, end=end)

# Describing Data
st.subheader(f'Data for {selected_company} (2010-2019)')
st.write(df.describe())

# Visualizations
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, label='100MA', color='blue')
plt.plot(df.Close, label='Closing Price', color='green')
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA and 200MA')
# Calculate moving averages
ma200 = df.Close.rolling(200).mean()
# Plotting
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, label='100MA', color='blue')
plt.plot(ma200, label='200MA', color='red')
plt.plot(df.Close, label='Closing Price', color='green')
plt.legend()
st.pyplot(fig)

# Splitting data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

st.write(f"Training Data Shape: {data_training.shape}")
st.write(f"Testing Data Shape: {data_testing.shape}")

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Load pre-trained model
model = load_model('keras_model.h5')

# Preparing testing data
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Making predictions
y_predicted = model.predict(x_test)
scaler_scale = scaler.scale_
scale_factor = 1 / scaler_scale[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor



import plotly.graph_objects as go

# Final Graph: Prediction vs Actual
st.subheader(f"Prediction vs Actual for {selected_company}")

# Create Plotly figure
fig2 = go.Figure()

# Add actual stock price line
fig2.add_trace(go.Scatter(
    y=y_test,
    mode='lines',
    name='Original Price',
    line=dict(color='blue'),
    hovertemplate='Day: %{x}<br>Price: %{y:.2f}'
))

# Add predicted stock price line
fig2.add_trace(go.Scatter(
    y=y_predicted.flatten(),
    mode='lines',
    name='Predicted Price',
    line=dict(color='red'),
    hovertemplate='Day: %{x}<br>Price: %{y:.2f}'
))

# Update layout
fig2.update_layout(
    title=f"Prediction vs Actual for {selected_company}",
    xaxis_title='Time',
    yaxis_title='Price',
    hovermode="x",  # Enable hover at the same x-value for both lines
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    template="plotly_white"
)

# Display the interactive plot in Streamlit
st.plotly_chart(fig2)







# Forecasting next 10 days
st.subheader(f"Forecasting Next 10 Days for {selected_company}")
recent_data = input_data[-100:]  # Take the last 100 days as input for forecasting
next_10_days = []

for _ in range(10):
    recent_data_reshaped = recent_data.reshape(1, 100, 1)
    next_predicted = model.predict(recent_data_reshaped)[0, 0]
    next_10_days.append(next_predicted)
    # Update recent_data to include the new prediction
    recent_data = np.append(recent_data[1:], [[next_predicted]], axis=0)

# Rescale predictions back to original scale
next_10_days = np.array(next_10_days) * scale_factor

# Display forecast as a table
forecast_df = pd.DataFrame({
    'Day': [f"Day {i+1}" for i in range(10)],
    'Predicted Price': next_10_days
})
st.write(forecast_df)





# User-Selected Date Range for Historical Data Analysis
st.subheader("Select Date Range for Historical Data Analysis")

# Streamlit date pickers
start_date = st.date_input("Start Date", value=pd.to_datetime('2010-01-01'))
end_date = st.date_input("End Date", value=pd.to_datetime('2019-12-31'))

# Validation to ensure valid date range
if start_date >= end_date:
    st.error("End Date must be after Start Date!")
else:
    # Fetch stock data for the selected date range
    df_custom = yf.download(user_input, start=start_date, end=end_date)

    # Check if data exists for the selected range
    if df_custom.empty:
        st.warning("No data available for the selected date range. Please choose a different range.")
    else:
        # Display data summary
        st.subheader(f'Data for {selected_company} ({start_date} to {end_date})')
        st.write(df_custom.describe())

        # Plot closing prices for the selected range
        st.subheader('Closing Price vs Time chart (Selected Range)')
        fig_custom = plt.figure(figsize=(12, 6))
        plt.plot(df_custom['Close'], label='Closing Price', color='green')
        plt.title(f'Closing Price for {selected_company}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig_custom)













