Experiment 2 - Predicting House prices using Linear Regression
Develop a regression model to predict house prices based on features like location, size, and amenities.
1. Problem Definition
The objective is to predict house prices based on structural and locational features such as area, number of bedrooms, bathrooms, balcony count, parking availability, and furnishing status. A Linear Regression model is trained on historical listing data to learn the mapping between these features and the continuous target variable (Price). The model is then evaluated using MSE and R² score and used to forecast prices for unseen properties.

2. Data Understanding
2.1 Features
HouseID, Location, Area_sqft, Bedrooms, Bathrooms, Balcony, Parking, Age, Furnished, Price

2.2 Target Variable
Price (continuous — property value in currency units)
	
3. Data Preprocessing
(i) One-Hot Encoding: Applied to the Location and Furnished categorical columns using OneHotEncoder within a ColumnTransformer pipeline. This converts categorical text values into binary indicator columns that the regression model can process numerically.

(ii) Feature–Target Separation: X = all columns except Price (shape: 120×9). y = Price. This ensures the model learns the functional mapping f(X)→y without target leakage during training.

(iii) Train–Test Split: 
Dataset	Total	Train (80%)	Test (20%)
House Price (Regression)	120	96	24

(iv) Pipeline Integration: A sklearn Pipeline wraps the ColumnTransformer (preprocessing) and LinearRegression (model) into a single unified workflow. This prevents data leakage and simplifies prediction on unseen data.

4. Model Training
Linear Regression: Minimizes the Ordinary Least Squares (OLS) loss — the sum of squared differences between actual and predicted prices. The pipeline applies One-Hot Encoding to categorical columns and passes all features to the regressor in one step. No regularization is applied — this serves as a direct baseline for the relationship between housing features and price.

5. Prediction and Model Evaluation
predict() is applied to the test set through the pipeline (which automatically applies the same encoding as training) to generate continuous house price estimates.

Metric	Value
MSE	6.839 × 10⁻²⁴
RMSE	2.615 × 10⁻¹²
R² Score	1.0000

6. Comparative Analysis
The model achieves a perfect R² of 1.0 and effectively zero MSE/RMSE, indicating that the housing features in this synthetic dataset are perfectly linearly related to price. The regression coefficients confirm that Location_Urban carries the highest weight (+25 units), followed by Location_Suburban (+10 units), while furnishing has a negligible effect (~4×10⁻¹⁴). The Actual vs Predicted scatter plot shows all points lying exactly on the diagonal, confirming perfect fit.

Note: A perfect R² on real-world data would be suspicious and suggest overfitting or data leakage. In this case it reflects the synthetic, deterministic nature of the dataset where price is a direct linear function of the features.

7. Conclusion
●	Linear Regression achieved a perfect R² of 1.0000 with near-zero MSE, confirming a perfectly linear relationship between the housing features and price in this dataset.
●	Location is the dominant predictor — Urban properties command a +25 unit price premium over Rural, and Suburban a +10 unit premium.
●	The pipeline architecture (ColumnTransformer + LinearRegression) cleanly handles categorical encoding and regression in a single reproducible workflow.
●	On real-world datasets with noise and non-linearity, performance would be lower; tree-based or ensemble models would likely outperform a plain Linear Regression.







