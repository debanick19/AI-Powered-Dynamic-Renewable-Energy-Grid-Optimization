#!/usr/bin/env python
# coding: utf-8

# ### Python Forecasting Model: Predicting Renewable Energy Generation

# In[241]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split




# In[ ]:





# In[143]:


# Load datasets
energy_data = pd.read_csv('Final_dataset.csv',encoding = 'unicode_escape')


# In[151]:


energy_data.shape


# In[219]:


energy_data.head()


# In[ ]:





# In[229]:


print(energy_data.columns)


# In[ ]:





# In[167]:


# Convert 'DateTime' to proper datetime format
energy_data ['Datetime'] = pd.to_datetime(energy_data['DateTime'])  # Use 'DateTime' instead of 'Date'


# In[ ]:





# In[231]:


print(energy_data.head())  # Check the first few rows


# In[ ]:





# In[215]:


## If the first row appears as column names, re-load the dataset with headers:
energy_data = pd.read_csv('Final_dataset.csv', header=0)


# In[233]:


# Convert DateTime column to datetime format

energy_data['Datetime'] = pd.to_datetime(energy_data['DateTime'])  # Use 'DateTime' instead of 'Timestamp'
energy_data.set_index('Datetime', inplace=True)


# In[ ]:





# In[239]:


# Check dataset structure
print(energy_data.head())
print(energy_data.info())


# In[ ]:





# In[243]:


## Energy Source for Prediction


# In[ ]:


# Select the target variable (e.g., Solar Energy)
target_column = 'Solar_Energy'  # Change to 'Wind_Energy' or others if needed
energy_data = energy_data[[target_column]]

# Plot the energy production trend
plt.figure(figsize=(12, 5))
plt.plot(energy_data, label="Actual Energy Production")
plt.title(f"Trend of {target_column} Production")
plt.xlabel("Time")
plt.ylabel("Energy Generated (kWh)")
plt.legend()
plt.show()


# In[ ]:





# In[265]:


## Split data into training and testing sets


# In[ ]:


# 80% Training, 20% Testing
train_size = int(len(energy_data) * 0.8)
train, test = energy_data.iloc[:train_size], energy_data.iloc[train_size:]

print(f"Train size: {len(train)}, Test size: {len(test)}")


# In[ ]:





# In[269]:


##ARIMA Model for Forecasting
#ARIMA is a statistical model used for time-series forecasting.


# In[ ]:


# Fit ARIMA Model
arima_model = ARIMA(train, order=(5,1,0))  # (p,d,q) parameters
arima_fit = arima_model.fit()

# Forecast
arima_forecast = arima_fit.forecast(steps=len(test))

# Plot results
plt.figure(figsize=(12,5))
plt.plot(train, label="Training Data")
plt.plot(test, label="Actual Energy Production", color='orange')
plt.plot(test.index, arima_forecast, label="ARIMA Forecast", color='red')
plt.title(f"ARIMA Forecast for {target_column}")
plt.legend()
plt.show()


# In[ ]:





# In[271]:


## Random forest


# In[273]:


#Prepare Data for Random Forest
#Since Random Forest requires a tabular format (X, y), we create lag features from historical data.


# In[ ]:


# Create lag features
df_rf = energy_data.copy()
df_rf['Lag_1'] = df_rf[target_column].shift(1)
df_rf['Lag_2'] = df_rf[target_column].shift(2)
df_rf['Lag_3'] = df_rf[target_column].shift(3)

# Drop NaN values caused by shifting
df_rf.dropna(inplace=True)

# Define Features (X) and Target (y)
X = df_rf.drop(columns=[target_column])
y = df_rf[target_column]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Check dataset shapes
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")


# In[ ]:





# #Train a Machine Learning Model (Random Forest Regressor)

# In[ ]:


# Initialize Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train model
rf_model.fit(X_train, y_train)

# Make predictions
rf_predictions = rf_model.predict(X_test)

# Convert to DataFrame for plotting
rf_predictions_df = pd.DataFrame(rf_predictions, index=y_test.index, columns=["RF_Predictions"])


# In[278]:


#evealuate Rf performance


# In[ ]:


# Compute performance metrics
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_mse = mean_squared_error(y_test, rf_predictions)

print(f"Random Forest MAE: {rf_mae:.2f}")
print(f"Random Forest MSE: {rf_mse:.2f}")


# In[280]:


## plot Rf


# In[ ]:


plt.figure(figsize=(12,5))
plt.plot(y_test, label="Actual Energy Production", color='orange')
plt.plot(rf_predictions_df, label="Random Forest Forecast", color='blue')
plt.title(f"Random Forest Forecast for {target_column}")
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[282]:


#To generate visuals and evaluate model performance:


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Function to plot actual vs. predicted values
def plot_actual_vs_predicted(y_true, y_pred, model_name):
    plt.figure(figsize=(10, 5))
    plt.plot(y_true.index, y_true, label="Actual", color='blue')
    plt.plot(y_true.index, y_pred, label="Predicted", color='red', linestyle="dashed")
    plt.xlabel("Time")
    plt.ylabel("Energy Generation")
    plt.title(f"Actual vs. Predicted Energy ({model_name})")
    plt.legend()
    plt.show()

# Function to plot residuals
def plot_residuals(y_true, y_pred, model_name):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 4))
    sns.histplot(residuals, bins=20, kde=True, color="purple")
    plt.xlabel("Residuals")
    plt.title(f"Residual Distribution ({model_name})")
    plt.show()

# Function to evaluate model performance
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"Model Performance: {model_name}")
    print(f" Mean Absolute Error (MAE): {mae:.2f}")
    print(f" Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f" R² Score: {r2:.2f}")
    
    return mae, rmse, r2



# ## Visual Insights
# 

# In[287]:


# If residuals are normally distributed → Model is unbiased
# Lower MAE & RMSE → Better model performance
#R² close to 1 → High predictive power


# In[ ]:





# In[79]:


# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)


# In[ ]:




