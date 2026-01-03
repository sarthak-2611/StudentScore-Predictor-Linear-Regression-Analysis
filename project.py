# GDG Assignment 2: Introduction to Linear Regression
# Personal Learning Project

## Task 1: Dataset Understanding & Basic Exploration

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load the dataset from local file
df = pd.read_csv("StudentsPerformance.csv")

# Display first 5 rows
print("First 5 rows of dataset:")
print(df.head())
print("\n")

# Check dataset shape
print(f"Dataset shape: {df.shape}")
print("\n")

# Column names
print("Column names:")
print(df.columns.tolist())
print("\n")

# Data types and info
print("Data types and info:")
df.info()
print("\n")

# Statistical summary
print("Statistical summary:")
print(df.describe())


## Task 2: Data Preprocessing

# Check for missing values
print("Missing values in each column:")
missing = df.isnull().sum()
print(missing)
print("\n")

# Handle missing values by replacing with mean
for column in df.columns:
    if df[column].dtype in ['float64', 'int64']:
        df[column] = df[column].fillna(df[column].mean())

print("Missing values after handling:")
print(df.isnull().sum())
print("\n")

# Select numerical columns for prediction
# Using reading score to predict math score
X = df[['reading score']].values  # Feature matrix
y = df['math score'].values        # Target vector

print(f"Feature matrix X shape: {X.shape}")
print(f"Target vector y shape: {y.shape}")
print("\n")

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")


## Task 3: Implement Linear Regression Model

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Extract model parameters
slope = model.coef_[0]
intercept = model.intercept_

print(f"\nModel Coefficient (Slope): {slope:.4f}")
print(f"Model Intercept: {intercept:.4f}")
print("\n")

# Explanation of coefficient
print("Coefficient Explanation:")
print(f"The slope of {slope:.4f} means that for every 1 unit increase in reading score,")
print(f"the math score is expected to increase by {slope:.4f} units on average.")
print(f"The intercept of {intercept:.4f} represents the predicted math score")
print(f"when the reading score is 0.")


## Task 4: Prediction & Visualization

# Make predictions on test data
y_pred = model.predict(X_test)

# Create visualization
plt.figure(figsize=(10, 6))

# Scatter plot of actual data
plt.scatter(X_test, y_test, color='blue', label='Actual Scores', alpha=0.6, s=50)

# Regression line
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')

# Labels and title
plt.xlabel('Reading Score', fontsize=12)
plt.ylabel('Math Score', fontsize=12)
plt.title('Linear Regression: Reading Score vs Math Score', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Visualization complete!")


## Task 5: Model Evaluation

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")
print("\n")

# Model performance explanation
print("Model Performance Analysis:")
print(f"The R² score of {r2:.4f} indicates that approximately {r2*100:.2f}% of the variance")
print(f"in math scores is explained by reading scores. An MAE of {mae:.4f} means that,")
print(f"on average, the model's predictions are off by {mae:.4f} points.")
if r2 > 0.7:
    print("This suggests a strong linear relationship between reading and math scores.")
elif r2 > 0.5:
    print("This suggests a moderate linear relationship between reading and math scores.")
else:
    print("This suggests a weak linear relationship between reading and math scores.")


## Bonus Task: Alternative Feature Analysis

print("\n" + "="*60)
print("BONUS TASK: Using Writing Score Instead")
print("="*60 + "\n")

# Train model with writing score
X_bonus = df[['writing score']].values
X_bonus_train, X_bonus_test, y_bonus_train, y_bonus_test = train_test_split(
    X_bonus, y, test_size=0.2, random_state=42
)

model_bonus = LinearRegression()
model_bonus.fit(X_bonus_train, y_bonus_train)

y_bonus_pred = model_bonus.predict(X_bonus_test)

mae_bonus = mean_absolute_error(y_bonus_test, y_bonus_pred)
r2_bonus = r2_score(y_bonus_test, y_bonus_pred)

print(f"Bonus Model Coefficient: {model_bonus.coef_[0]:.4f}")
print(f"Bonus Model R² Score: {r2_bonus:.4f}")
print(f"Bonus Model MAE: {mae_bonus:.4f}")
print("\n")

# Comparison
print("Comparison of Features:")
print(f"Reading Score - R²: {r2:.4f}, MAE: {mae:.4f}")
print(f"Writing Score - R²: {r2_bonus:.4f}, MAE: {mae_bonus:.4f}")
print("\n")

if r2 > r2_bonus:
    print(f"Reading score is a better predictor ({r2:.4f} > {r2_bonus:.4f})")
else:
    print(f"Writing score is a better predictor ({r2_bonus:.4f} > {r2:.4f})")

# Bonus visualization
plt.figure(figsize=(12, 5))

# Reading score subplot
plt.subplot(1, 2, 1)
plt.scatter(X_test, y_test, color='blue', alpha=0.6, s=50)
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.xlabel('Reading Score')
plt.ylabel('Math Score')
plt.title(f'Reading Score (R²: {r2:.4f})')
plt.grid(True, alpha=0.3)

# Writing score subplot
plt.subplot(1, 2, 2)
plt.scatter(X_bonus_test, y_bonus_test, color='green', alpha=0.6, s=50)
plt.plot(X_bonus_test, y_bonus_pred, color='red', linewidth=2)
plt.xlabel('Writing Score')
plt.ylabel('Math Score')
plt.title(f'Writing Score (R²: {r2_bonus:.4f})')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Bonus analysis complete!")
