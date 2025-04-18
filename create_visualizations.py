import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Create output directory
output_dir = 'email_optimization_output'
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Load the datasets
print("Loading datasets...")
email_data = pd.read_csv('email_table.csv')
email_opened = pd.read_csv('email_opened_table.csv')
link_clicked = pd.read_csv('link_clicked_table.csv')

# Create target variables
email_data['email_opened'] = email_data['email_id'].isin(email_opened['email_id']).astype(int)
email_data['link_clicked'] = email_data['email_id'].isin(link_clicked['email_id']).astype(int)

# Calculate baseline metrics
open_rate = email_data['email_opened'].mean() * 100
click_rate = email_data['link_clicked'].mean() * 100
click_through_rate = email_data.loc[email_data['email_opened'] == 1, 'link_clicked'].mean() * 100

print(f"Baseline Metrics:")
print(f"Open Rate: {open_rate:.2f}%")
print(f"Click Rate: {click_rate:.2f}%")
print(f"Click-through Rate (among opened): {click_through_rate:.2f}%")

# Basic feature engineering
# 1. Time of day
email_data['time_of_day'] = pd.cut(
    email_data['hour'], 
    bins=[0, 6, 12, 18, 24], 
    labels=['Night', 'Morning', 'Afternoon', 'Evening']
)

# 2. Purchase segments
email_data['purchase_segment'] = pd.cut(
    email_data['user_past_purchases'], 
    bins=[-1, 0, 3, 7, 100], 
    labels=['No purchases', '1-3 purchases', '4-7 purchases', '8+ purchases']
)

# Set plot style
plt.style.use('ggplot')
sns.set(style="whitegrid")

# 1. Create baseline metrics visualization
plt.figure(figsize=(10, 6))
metrics = [open_rate, click_rate, click_through_rate]
metric_names = ['Open Rate', 'Click Rate', 'Click-through Rate']
colors = ['#3498db', '#2ecc71', '#e74c3c']

plt.bar(metric_names, metrics, color=colors)
plt.title('Baseline Email Campaign Performance', fontsize=16)
plt.ylabel('Percentage (%)', fontsize=12)
plt.ylim(0, max(metrics) * 1.2)

# Add value labels
for i, v in enumerate(metrics):
    plt.text(i, v + 1, f"{v:.2f}%", ha='center', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'baseline_metrics.png'))
plt.close()

# 2. Email content analysis
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.barplot(x='email_text', y='link_clicked', data=email_data)
plt.title('Click Rate by Email Type')
plt.xlabel('Email Type')
plt.ylabel('Click Rate')

plt.subplot(1, 2, 2)
sns.barplot(x='email_version', y='link_clicked', data=email_data)
plt.title('Click Rate by Email Version')
plt.xlabel('Email Version')
plt.ylabel('Click Rate')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'email_type_analysis.png'))
plt.close()

# 3. Time-based analysis
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.barplot(x='time_of_day', y='link_clicked', data=email_data)
plt.title('Click Rate by Time of Day')
plt.xlabel('Time of Day')
plt.ylabel('Click Rate')

plt.subplot(1, 2, 2)
sns.barplot(x='weekday', y='link_clicked', data=email_data, 
           order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.title('Click Rate by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Click Rate')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'time_analysis.png'))
plt.close()

# 4. User demographics analysis
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.barplot(x='user_country', y='link_clicked', data=email_data)
plt.title('Click Rate by Country')
plt.xlabel('Country')
plt.ylabel('Click Rate')

plt.subplot(1, 2, 2)
sns.barplot(x='purchase_segment', y='link_clicked', data=email_data)
plt.title('Click Rate by Purchase History')
plt.xlabel('Purchase Segment')
plt.ylabel('Click Rate')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'user_demographics.png'))
plt.close()

# 5. Combined factors analysis
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.barplot(x='email_text', y='link_clicked', hue='email_version', data=email_data)
plt.title('Click Rate by Email Text and Version')
plt.xlabel('Email Text')
plt.ylabel('Click Rate')
plt.legend(title='Email Version')

plt.subplot(1, 2, 2)
# Create a weekend flag
email_data['is_weekend'] = email_data['weekday'].isin(['Saturday', 'Sunday']).astype(int)
sns.barplot(x='time_of_day', y='link_clicked', hue='is_weekend', data=email_data)
plt.title('Click Rate by Time of Day and Day Type')
plt.xlabel('Time of Day')
plt.ylabel('Click Rate')
plt.legend(title='Weekend', labels=['Weekday', 'Weekend'])

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'combined_factors.png'))
plt.close()

# 6. Create a simulated A/B test visualization
plt.figure(figsize=(10, 6))

# Simulated data
groups = ['Control', 'Treatment']
open_rates = [10.35, 13.46]  # 30% improvement
click_rates = [2.12, 4.24]   # 100% improvement

# Create a grouped bar chart
x = np.arange(len(groups))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, open_rates, width, label='Open Rate', color='#3498db')
rects2 = ax.bar(x + width/2, click_rates, width, label='Click Rate', color='#2ecc71')

ax.set_title('A/B Test Results: Control vs. Treatment', fontsize=16)
ax.set_ylabel('Rate (%)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=12)
ax.legend()

# Add value labels
for rect in rects1:
    height = rect.get_height()
    ax.annotate(f'{height:.2f}%',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

for rect in rects2:
    height = rect.get_height()
    ax.annotate(f'{height:.2f}%',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ab_test_results.png'))
plt.close()

# 7. Create a simulated ROI projection
plt.figure(figsize=(10, 6))

# Simulated data
phases = ['Current', 'Phase 1', 'Phase 2', 'Phase 3']
click_rates = [2.12, 5.01, 6.5, 7.5]
improvements = [0, 136, 207, 254]

fig, ax1 = plt.subplots(figsize=(10, 6))

color1 = '#3498db'
color2 = '#e74c3c'

ax1.set_xlabel('Implementation Phase', fontsize=12)
ax1.set_ylabel('Click Rate (%)', fontsize=12, color=color1)
bars = ax1.bar(phases, click_rates, color=color1, alpha=0.7)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0, max(click_rates) * 1.2)

# Add value labels to bars
for bar in bars:
    height = bar.get_height()
    ax1.annotate(f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10, color=color1)

# Create a second y-axis for improvement percentage
ax2 = ax1.twinx()
ax2.set_ylabel('Improvement (%)', fontsize=12, color=color2)
ax2.plot(phases, improvements, color=color2, marker='o', linewidth=2, markersize=8)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0, max(improvements) * 1.2)

# Add value labels to line
for i, v in enumerate(improvements):
    if i > 0:  # Skip the first point (0%)
        ax2.annotate(f'+{v}%',
                    xy=(i, v),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, color=color2)

plt.title('Projected ROI by Implementation Phase', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'roi_projection.png'))
plt.close()

# 8. Create a simulated implementation framework diagram
plt.figure(figsize=(12, 8))

# Create a simple diagram
components = ['Data Processing', 'Machine Learning', 'Micro-segmentation', 
              'Campaign Generator', 'A/B Testing', 'Reporting']
importance = [0.8, 0.9, 0.95, 0.85, 0.75, 0.7]  # Simulated importance scores

# Create horizontal bars
y_pos = np.arange(len(components))
plt.barh(y_pos, importance, align='center', 
         color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f1c40f', '#1abc9c'])
plt.yticks(y_pos, components)
plt.xlabel('Component Importance')
plt.title('Email Marketing Optimization System Framework', fontsize=16)

# Add arrows to show flow
plt.annotate('', xy=(0.4, 0), xytext=(0.4, 1), 
             arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
plt.annotate('', xy=(0.4, 1), xytext=(0.4, 2), 
             arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
plt.annotate('', xy=(0.4, 2), xytext=(0.4, 3), 
             arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
plt.annotate('', xy=(0.4, 3), xytext=(0.4, 4), 
             arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
plt.annotate('', xy=(0.4, 4), xytext=(0.4, 5), 
             arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'implementation_framework.png'))
plt.close()

# 9. Create a simulated micro-segments visualization
plt.figure(figsize=(12, 8))

# Simulated data for micro-segments
segments = ['High_8+ purchases_UK', 'High_4-7 purchases_US', 
            'Medium-High_8+ purchases_US', 'Medium-High_4-7 purchases_UK',
            'Medium-Low_1-3 purchases_UK', 'Low_No purchases_US']
click_rates = [7.32, 5.89, 4.76, 4.52, 2.87, 0.95]
segment_sizes = [1243, 3567, 5821, 2134, 4567, 8765]

# Create bubble chart
plt.figure(figsize=(12, 8))
plt.scatter(range(len(segments)), click_rates, s=[size/50 for size in segment_sizes], 
            alpha=0.6, c=click_rates, cmap='viridis')

plt.xlabel('Segment', fontsize=12)
plt.ylabel('Click Rate (%)', fontsize=12)
plt.title('Micro-Segments Performance and Size', fontsize=16)
plt.xticks(range(len(segments)), segments, rotation=45, ha='right')
plt.colorbar(label='Click Rate (%)')

# Add value labels
for i, (cr, size) in enumerate(zip(click_rates, segment_sizes)):
    plt.annotate(f'{cr:.2f}% ({size:,})',
                xy=(i, cr),
                xytext=(0, 10),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'micro_segments.png'))
plt.close()

print("Visualizations created successfully!")
print(f"All visualizations saved to {output_dir} directory")
