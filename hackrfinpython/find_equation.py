#insert data and create equation with polynomial regression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Given median signal strengths
signal_strengths = np.array([
    -121.24317450272409, 
    -120.92755908877761, 
    -120.43141103254047, 
    -120.33340991290972, 
    -120.60623627053427, 
    -121.10241654447452
])

# Assume distances (or replace with actual measured distances)
distances = np.array([0.3, 0.6, 0.9, 1.2, 1.5, 1.8]).reshape(-1, 1)  # Reshape for ML model

# Choose model: Linear Regression or Polynomial Regression
use_polynomial = True  # Set to False for simple linear regression

def train_model(distances, signal_strengths, degree=2):
    if use_polynomial:
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    else:
        model = LinearRegression()
    
    model.fit(distances, signal_strengths)
    return model

# Train model
model = train_model(distances, signal_strengths)

# Predict values
distance_range = np.linspace(min(distances), max(distances), 100).reshape(-1, 1)
predicted_signal = model.predict(distance_range)

# Extract model coefficients
if use_polynomial:
    poly_features = model.named_steps['polynomialfeatures']
    lin_reg = model.named_steps['linearregression']
    coeffs = lin_reg.coef_
    intercept = lin_reg.intercept_
    equation_terms = [f"{coeff:.4f} * d^{i}" for i, coeff in enumerate(coeffs)]
    equation = "Signal Strength = " + " + ".join(equation_terms) + f" + {intercept:.4f}"
else:
    lin_reg = model
    coeffs = lin_reg.coef_[0]
    intercept = lin_reg.intercept_
    equation = f"Signal Strength = {coeffs:.4f} * d + {intercept:.4f}"

print("Model Equation:", equation)

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(distances, signal_strengths, color='red', label='Measured Data')
plt.plot(distance_range, predicted_signal, 'b-', label='ML Regression Fit')
plt.xlabel("Distance (m)")
plt.ylabel("Signal Strength (dB)")
plt.legend()
plt.grid(True)
plt.show()

# Take user input for a new signal strength value
new_signal_strength = float(input("Enter a new signal strength value (dB): "))

# Predict the corresponding distance
predicted_distance = model.predict([[new_signal_strength]])

# Plot original data
plt.figure(figsize=(8, 5))
plt.scatter(distances, signal_strengths, color='red', label='Measured Data')
plt.plot(distance_range, predicted_signal, 'b-', label='ML Regression Fit')

# Plot the new input point in a different color
plt.scatter(predicted_distance, [new_signal_strength], color='green', s=100, label="New Prediction")
plt.annotate(f"({predicted_distance[0]:.2f}, {new_signal_strength})", 
             (predicted_distance, new_signal_strength), 
             textcoords="offset points", xytext=(10,10), ha='center')

# Labels and legend
plt.xlabel("Distance (m)")
plt.ylabel("Signal Strength (dB)")
plt.legend()
plt.grid(True)
plt.show()

print(f"Predicted Distance for Signal Strength {new_signal_strength} dB: {predicted_distance[0]:.2f} meters")
