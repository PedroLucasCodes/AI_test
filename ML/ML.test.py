import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Read dataset
print("[INFO] Reading dataset from 'dataset.csv'...")
df = pd.read_csv("dataset.csv")


######################## Preencha o código que falta ######################## 
# Step 2: Extract features and target
print("[INFO] Extracting 'feature_1' and 'target' columns...")
'''
Here We need some corretions 🙀
Fix it! Plz! 🙀
'''
X = df
y = df

# Step 3: Normalize the data
print("[INFO] Normalizing features and target...")
'''
Here We need some corretions 🙀
Fix it! Plz! 🙀
'''
X_scaled = X
y_scaled = y
########################        Até aqui apenas        ######################## 


############### Explique os trechos de código daqui em diante  ################ 
# Step 4: EXPLAIN PLEASE! 😸
print("[INFO] Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Step 5: EXPLAIN PLEASE! 😸
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Step 6: EXPLAIN PLEASE! 😸
y_train_pred = model.predict(X_train_poly)
mse_train = mean_squared_error(y_train, y_train_pred)
print(f"[RESULT] Training MSE: {mse_train:.4f}")

# Step 7: EXPLAIN PLEASE! 😸
print("[INFO] Predicting on the test set...")
y_pred_scaled = model.predict(X_test_poly)

# Step 8: EXPLAIN PLEASE! 😸
print("[INFO] Inversely transforming the predictions to original scale...")
y_test_orig = scaler_y.inverse_transform(y_test)
y_pred_orig = scaler_y.inverse_transform(y_pred_scaled)

# Step 9: EXPLAIN PLEASE! 😸
print("[INFO] Plotting results and saving to 'prediction_plot.png'...")
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test_orig, color='blue', label='Actual Y (Test)')
plt.scatter(X_test, y_pred_orig, color='red', label='Predicted Y')
plt.xlabel('Normalized Feature 1')
plt.ylabel('Target')
plt.title('Actual vs Predicted Values (Polynomial Regression)')
plt.legend()
plt.grid(True)
plt.savefig("prediction_plot.png")
plt.close()
print("[DONE] Plot saved as 'prediction_plot.png'.")
########################        Até aqui apenas        ######################## 
