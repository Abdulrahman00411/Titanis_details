import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV data
df = pd.read_csv('titanic.csv')

# Clean null values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = df.drop('Cabin', axis=1)
df = df.dropna()

# Create age groups
bins = [0, 12, 18, 30, 50, 100]
labels = ['Child (0-12)', 'Teen (13-18)', 'Young Adult (19-30)', 'Adult (31-50)', 'Senior (51+)']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# Map Pclass to descriptive labels
df['Class'] = df['Pclass'].map({1: 'First', 2: 'Second', 3: 'Third'})

# Calculate survival rates
sex_stats = df.groupby('Sex').agg({
    'PassengerId': 'count',
    'Survived': 'mean'
}).rename(columns={'PassengerId': 'Count', 'Survived': 'SurvivalRate'})

age_stats = df.groupby('AgeGroup').agg({
    'PassengerId': 'count',
    'Survived': 'mean'
}).rename(columns={'PassengerId': 'Count', 'Survived': 'SurvivalRate'})

class_stats = df.groupby('Class').agg({
    'PassengerId': 'count',
    'Survived': 'mean'
}).rename(columns={'PassengerId': 'Count', 'Survived': 'SurvivalRate'})

# Plot bar graphs
plt.figure(figsize=(15, 5))

# Survival Rate by Gender
plt.subplot(1, 3, 1)
sex_stats['SurvivalRate'].plot(kind='bar', color='skyblue')
plt.title('Survival Rate by Gender')
plt.ylabel('Survival Rate')
plt.xlabel('Gender')
plt.xticks(rotation=0)

# Survival Rate by Age Group
plt.subplot(1, 3, 2)
age_stats['SurvivalRate'].plot(kind='bar', color='lightgreen')
plt.title('Survival Rate by Age Group')
plt.ylabel('Survival Rate')
plt.xlabel('Age Group')
plt.xticks(rotation=45)

# Survival Rate by Class
plt.subplot(1, 3, 3)
class_stats['SurvivalRate'].plot(kind='bar', color='salmon')
plt.title('Survival Rate by Passenger Class')
plt.ylabel('Survival Rate')
plt.xlabel('Class')
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig('titanic_survival_rates.png')
plt.show()