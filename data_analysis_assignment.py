#  Data Analysis & Visualization Assignment


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# ------------------------------------------------------------
# Task 1: Load and Explore the Dataset
# ------------------------------------------------------------

try:
    # Load Iris dataset using sklearn and convert to DataFrame
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    print(" Dataset loaded successfully!")
except Exception as e:
    print(f" Error loading dataset: {e}")

# Display the first few rows
print("\n First 5 rows of the dataset:")
print(df.head())

# Check data types and missing values
print("\n Dataset Info:")
print(df.info())

print("\n Missing Values:")
print(df.isnull().sum())

# In this dataset there are no missing values, but if there were:
# df = df.fillna(df.mean())   # Example of filling missing values

# ------------------------------------------------------------
# Task 2: Basic Data Analysis
# ------------------------------------------------------------

# Basic statistics
print("\n Basic Statistics:")
print(df.describe())

# Grouping: mean of numerical columns by species
grouped = df.groupby('species').mean(numeric_only=True)
print("\n Mean values by species:")
print(grouped)

# Identify some patterns manually
print("\n Observations:")
print("- Setosa generally has smaller petal and sepal sizes.")
print("- Virginica tends to have the largest petal length and width on average.")
print("- Versicolor falls in between the two.")

# ------------------------------------------------------------
# Task 3: Data Visualization
# ------------------------------------------------------------

# Style
sns.set(style="whitegrid")

# 1. Line Chart (using sepal length as a "trend" by index)
plt.figure(figsize=(8, 5))
for species in df['species'].unique():
    subset = df[df['species'] == species]
    plt.plot(subset.index, subset['sepal length (cm)'], label=species)
plt.title("Line Chart: Sepal Length Trend by Species")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.tight_layout()
plt.show()

# 2. Bar Chart: Average petal length per species
plt.figure(figsize=(6, 5))
grouped['petal length (cm)'].plot(kind='bar', color=['skyblue', 'salmon', 'lightgreen'])
plt.title("Bar Chart: Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 3. Histogram: Sepal length distribution
plt.figure(figsize=(8, 5))
plt.hist(df['sepal length (cm)'], bins=15, color='purple', edgecolor='black')
plt.title("Histogram: Sepal Length Distribution")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 4. Scatter Plot: Sepal Length vs Petal Length
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.tight_layout()
plt.show()

print("\n✅ All tasks completed successfully!")
