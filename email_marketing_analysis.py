import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Set the style for plots
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Load the datasets

email_data = pd.read_csv('email_table.csv')
email_opened = pd.read_csv('email_opened_table.csv')
link_clicked = pd.read_csv('link_clicked_table.csv')

# Display basic information about the datasets
print("\nEmail data shape:", email_data.shape)
print("Email opened data shape:", email_opened.shape)
print("Link clicked data shape:", link_clicked.shape)

# Display the first few rows of each dataset

print(email_data.head())

# Check for missing values
print("\nMissing values in email_data:")
print(email_data.isnull().sum())

# Data preprocessing
print("\nPreprocessing data...")

# Create target variables
# 1. Email opened (binary)
email_data['email_opened'] = email_data['email_id'].isin(email_opened['email_id']).astype(int)

# 2. Link clicked (binary) - our success metric
email_data['link_clicked'] = email_data['email_id'].isin(link_clicked['email_id']).astype(int)

# Calculate open rate and click rate
open_rate = email_data['email_opened'].mean() * 100
click_rate = email_data['link_clicked'].mean() * 100
click_through_rate = email_data.loc[email_data['email_opened'] == 1, 'link_clicked'].mean() * 100

print(f"\nOverall open rate: {open_rate:.2f}%")
print(f"Overall click rate: {click_rate:.2f}%")
print(f"Click-through rate (clicks among opened emails): {click_through_rate:.2f}%")

# Encode categorical variables
print("\nEncoding categorical variables...")
categorical_cols = ['email_text', 'email_version', 'weekday', 'user_country']

# Create a copy of the dataframe for analysis
df_analysis = email_data.copy()

# Label encoding for categorical variables
label_encoder = LabelEncoder()
for col in categorical_cols:
    df_analysis[col + '_encoded'] = label_encoder.fit_transform(df_analysis[col])

# Convert hour to categorical (time of day)
df_analysis['time_of_day'] = pd.cut(
    df_analysis['hour'], 
    bins=[0, 6, 12, 18, 24], 
    labels=['Night', 'Morning', 'Afternoon', 'Evening']
)

# Save the processed data
df_analysis.to_csv('processed_email_data.csv', index=False)
print("Processed data saved to 'processed_email_data.csv'")

# Exploratory Data Analysis (EDA)
print("\nPerforming Exploratory Data Analysis...")

# 1. Distribution of email types
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.countplot(x='email_text', data=df_analysis, hue='link_clicked')
plt.title('Email Text Type vs. Click Rate')
plt.xlabel('Email Text Type')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
sns.countplot(x='email_version', data=df_analysis, hue='link_clicked')
plt.title('Email Version vs. Click Rate')
plt.xlabel('Email Version')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('email_type_analysis.png')

# 2. Time-based analysis
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.barplot(x='time_of_day', y='link_clicked', data=df_analysis, estimator=np.mean)
plt.title('Click Rate by Time of Day')
plt.xlabel('Time of Day')
plt.ylabel('Click Rate')

plt.subplot(1, 2, 2)
sns.barplot(x='weekday', y='link_clicked', data=df_analysis, estimator=np.mean, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.title('Click Rate by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Click Rate')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('time_analysis.png')

# 3. User demographics analysis
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.barplot(x='user_country', y='link_clicked', data=df_analysis, estimator=np.mean)
plt.title('Click Rate by Country')
plt.xlabel('Country')
plt.ylabel('Click Rate')

plt.subplot(1, 2, 2)
# Group users by past purchase behavior
df_analysis['purchase_group'] = pd.cut(
    df_analysis['user_past_purchases'], 
    bins=[-1, 0, 3, 7, 100], 
    labels=['No purchases', '1-3 purchases', '4-7 purchases', '8+ purchases']
)
sns.barplot(x='purchase_group', y='link_clicked', data=df_analysis, estimator=np.mean)
plt.title('Click Rate by Purchase History')
plt.xlabel('Past Purchase Group')
plt.ylabel('Click Rate')

plt.tight_layout()
plt.savefig('user_demographics.png')

# 4. Correlation analysis
plt.figure(figsize=(10, 8))
correlation_cols = ['hour', 'user_past_purchases', 'email_text_encoded', 
                    'email_version_encoded', 'weekday_encoded', 
                    'user_country_encoded', 'email_opened', 'link_clicked']
correlation = df_analysis[correlation_cols].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')

# 5. Combined factors analysis
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.barplot(x='email_text', y='link_clicked', hue='email_version', data=df_analysis, estimator=np.mean)
plt.title('Click Rate by Email Text and Version')
plt.xlabel('Email Text')
plt.ylabel('Click Rate')
plt.legend(title='Email Version')

plt.subplot(1, 2, 2)
# Create a new column for time and email type combination
df_analysis['time_email_combo'] = df_analysis['time_of_day'].astype(str) + '_' + df_analysis['email_text']
top_combos = df_analysis.groupby('time_email_combo')['link_clicked'].mean().sort_values(ascending=False).head(5).index
sns.barplot(x='time_email_combo', y='link_clicked', data=df_analysis[df_analysis['time_email_combo'].isin(top_combos)], estimator=np.mean)
plt.title('Top 5 Time-Email Type Combinations')
plt.xlabel('Time - Email Type')
plt.ylabel('Click Rate')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('combined_factors.png')

print("EDA completed. Visualizations saved as PNG files.")

# Print summary of findings
print("\nSummary of Key Findings:")
print("-----------------------")

# Email type effectiveness
email_text_click_rate = df_analysis.groupby('email_text')['link_clicked'].mean()
email_version_click_rate = df_analysis.groupby('email_version')['link_clicked'].mean()
print(f"Email Text Click Rates: {dict(email_text_click_rate.items())}")
print(f"Email Version Click Rates: {dict(email_version_click_rate.items())}")

# Best time to send
time_of_day_click_rate = df_analysis.groupby('time_of_day')['link_clicked'].mean()
weekday_click_rate = df_analysis.groupby('weekday')['link_clicked'].mean()
best_time = time_of_day_click_rate.idxmax()
best_day = weekday_click_rate.idxmax()
print(f"Best time of day: {best_time} (Click rate: {time_of_day_click_rate.max():.4f})")
print(f"Best day of week: {best_day} (Click rate: {weekday_click_rate.max():.4f})")

# User demographics
country_click_rate = df_analysis.groupby('user_country')['link_clicked'].mean()
purchase_group_click_rate = df_analysis.groupby('purchase_group')['link_clicked'].mean()
best_country = country_click_rate.idxmax()
best_purchase_group = purchase_group_click_rate.idxmax()
print(f"Best performing country: {best_country} (Click rate: {country_click_rate.max():.4f})")
print(f"Best performing purchase group: {best_purchase_group} (Click rate: {purchase_group_click_rate.max():.4f})")

# Best combination
combo_click_rate = df_analysis.groupby(['email_text', 'email_version', 'time_of_day', 'weekday'])['link_clicked'].mean()
best_combo = combo_click_rate.idxmax()
print(f"Best overall combination: {best_combo} (Click rate: {combo_click_rate.max():.4f})")

print("\nAnalysis complete!")
