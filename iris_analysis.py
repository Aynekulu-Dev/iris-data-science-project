import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures

# Load the dataset
df = pd.read_csv('iris.csv')

# Basic data exploration
print("Dataset Info:")
print(df.info())
print("\nDescriptive Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Visualize pairwise relationships
sns.pairplot(df, hue='variety')
plt.savefig('pairplot.png')
plt.close()

# Box plot for sepal length by species
plt.figure(figsize=(10, 6))
sns.boxplot(x='variety', y='sepal.length', data=df)
plt.title('Sepal Length by Species')
plt.savefig('boxplot_sepal_length.png')
plt.close()

# Correlation matrix for numeric columns only
numeric_df = df.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation_heatmap.png')
plt.close()

# Define the regression function (unchanged)
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X, y, cv=5)
    return X_test, y_test, y_pred, mse, r2, cv_scores

# Prepare data and apply polynomial transformation
X = df[['sepal.length', 'sepal.width']]
y = df['petal.width']
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
X_test, y_test, y_pred, mse, r2, cv_scores = train_model(X_poly, y)

# Evaluate and print
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")
print(f"Cross-validation scores: {cv_scores}")
print(f"Average CV score: {cv_scores.mean()} (+/- {cv_scores.std() * 2})")

# Visualize regression (adjust for polynomial features)
plt.figure(figsize=(10, 6))
# Use the first original feature (sepal.length) for a 2D approximation
plt.scatter(X_test[:, 0], y_test, color='blue', label='Actual')
plt.plot(X_test[:, 0], y_pred, color='red', label='Predicted')
plt.xlabel('Sepal Length (Polynomial Approximation)')
plt.ylabel('Petal Width')
plt.title('Polynomial Regression: Sepal Length and Width vs Petal Width')
plt.legend()
plt.savefig('regression_plot.png')
plt.close()

# Residual plot
plt.figure(figsize=(10, 6))
residuals = y_test - y_pred
plt.scatter(y_test, residuals, color='green')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Actual Petal Width')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.savefig('residuals.png')
plt.close()

print("Visualizations saved as pairplot.png, boxplot_sepal_length.png, correlation_heatmap.png, regression_plot.png, and residuals.png")