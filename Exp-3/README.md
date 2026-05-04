Experiment 3 - Predicting Stock prices using Linear Regression
 
Predicting Stock Prices: Develop a time series prediction model to forecast stock prices

1. Problem Definition
The objective is to predict future stock closing prices using Linear Regression applied to historical time-series data. The model treats the date (converted to a numerical index) as the independent variable and the closing price as the target. After training on 80% of historical data, the model forecasts prices 30 days into the future using a recursive prediction loop. Performance is measured using RMSE and R² score.

2. Data Understanding
2.1 Features
Date (converted to numerical index), Open, High, Low, Close, Volume

2.2 Target Variable
Close (continuous — daily closing stock price)

3. Data Preprocessing
(i) Date Conversion: The Date column is parsed using pd.to_datetime() and converted to an integer index (days since origin) using date.toordinal(). This transforms the temporal feature into a numerical value that Linear Regression can process.

(ii) Feature–Target Separation: X = numerical date index (shape: N×1). y = Close price. A univariate regression is used — date alone is the predictor, capturing the linear trend component of price movement.

(iii) Train–Test Split: The first 80% of chronologically ordered rows form the training set; the remaining 20% form the test set. This respects the temporal order of the data and prevents future data from leaking into training — a critical requirement for time-series tasks.

(iv) Feature Scaling: Not applied — Linear Regression on a single numerical feature does not require scaling since no penalty terms or distance metrics are involved.

4. Model Training
Linear Regression (Trend Model): Fits a straight line through the historical Close prices as a function of the date index. While stock prices are inherently non-linear, a linear trend model captures the overall directional movement (uptrend/downtrend) and provides a useful baseline for short-horizon forecasting.

5. Prediction and Model Evaluation
predict() is called on the test date indices to generate price forecasts. For 30-day future forecasting, the model generates date indices 30 days beyond the last observed date and predicts recursively.

Metric	Value
RMSE	2.0964
R² Score	0.9825

6. Comparative Analysis
An R² of 0.9825 indicates that the linear trend model explains 98.25% of the variance in stock prices over the test period — a strong result for a single-feature model. The RMSE of 2.0964 represents the average deviation in price units between predicted and actual closing values. The Actual vs Predicted plot and 30-day forecast visualization confirm the model tracks the overall trend accurately, though it cannot capture short-term volatility or reversals by design.

For more accurate stock price forecasting, time-series specific models (LSTM, ARIMA, Prophet) that capture autocorrelation and seasonality would be more appropriate than a static Linear Regression.

7. Conclusion
●	Linear Regression achieved RMSE = 2.0964 and R² = 0.9825, demonstrating strong trend-fitting capability on this stock price dataset.
●	The date-as-index approach effectively converts a time-series problem into a standard regression task, suitable for trend extraction.
●	The 30-day recursive forecast provides directional guidance but cannot model volatility — a fundamental limitation of linear trend models on financial data.
●	For production-grade stock price prediction, non-linear and sequence-aware models (LSTM, ARIMA) are recommended to capture market dynamics beyond simple trend.

