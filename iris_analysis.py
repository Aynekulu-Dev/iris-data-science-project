import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

print("Visualizations saved as pairplot.png, boxplot_sepal_length.png, and correlation_heatmap.png")