# Stock-Price-Prediction-and-Forecasting
Description:
Developed an intuitive and interactive Stock Trend Prediction App using Streamlit, enabling users to analyze and forecast stock trends with the integration of machine learning models. The app provides the ability to select from popular companies like Apple, Tesla, and Google, visualize historical trends, and generate stock price forecasts.

Key Features:

Company Selection: Users can choose from a list of top companies (e.g., Apple, Tesla, Google) to analyze their stock trends.

Historical Data Analysis:
Visualizations of historical stock prices with customizable date ranges.
Includes trends with 100-day and 200-day moving averages for better decision-making.

Prediction and Comparison:
Comparison of predicted vs. actual stock prices using interactive Plotly charts for a detailed exploration.

Stock Price Forecasting:
Generates a 30-day stock price forecast based on historical trends using a state-of-the-art Long Short-Term Memory (LSTM) model.
User-Friendly Interface: Built on Streamlit, the app ensures smooth navigation and interactive features, making it accessible for both experts and beginners.


LSTM Model Design:

Architecture: Four layers of stacked LSTMs with progressively increasing units (50, 60, 80, 120).

Dropout Regularization: Dropout layers with rates ranging from 20% to 50% are added after each LSTM layer to prevent overfitting.

Training: The model is trained for 50 epochs using the Adam optimizer and Mean Squared Error (MSE) loss function, ensuring effective learning of temporal patterns in stock price data.


Technologies Used:

Frontend: Streamlit for an interactive user experience.

Backend: Keras with TensorFlow for building and training the LSTM model.

Data Source: Yahoo Finance API for fetching historical stock data.

Visualization: Matplotlib, Seaborn, and Plotly for dynamic and clear data representation.

This app serves as a powerful tool for investors and financial analysts, combining intuitive design with machine learning to provide actionable insights into stock market trends and future price predictions.
