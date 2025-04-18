import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.feature_selection import SelectFromModel
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Loading and preprocessing data...")

# Load the datasets
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

# Feature Engineering
print("\nPerforming advanced feature engineering...")

# 1. Time-based features
email_data['hour_sin'] = np.sin(2 * np.pi * email_data['hour']/24)
email_data['hour_cos'] = np.cos(2 * np.pi * email_data['hour']/24)

# 2. Create time of day categories
email_data['time_of_day'] = pd.cut(
    email_data['hour'], 
    bins=[0, 6, 12, 18, 24], 
    labels=['Night', 'Morning', 'Afternoon', 'Evening']
)

# 3. Create purchase history segments
email_data['purchase_segment'] = pd.cut(
    email_data['user_past_purchases'], 
    bins=[-1, 0, 3, 7, 100], 
    labels=['No purchases', '1-3 purchases', '4-7 purchases', '8+ purchases']
)

# 4. Create interaction features
email_data['personalized_short'] = ((email_data['email_text'] == 'short_email') & 
                                   (email_data['email_version'] == 'personalized')).astype(int)
email_data['personalized_long'] = ((email_data['email_text'] == 'long_email') & 
                                  (email_data['email_version'] == 'personalized')).astype(int)
email_data['generic_short'] = ((email_data['email_text'] == 'short_email') & 
                              (email_data['email_version'] == 'generic')).astype(int)
email_data['generic_long'] = ((email_data['email_text'] == 'long_email') & 
                             (email_data['email_version'] == 'generic')).astype(int)

# 5. Day type feature
email_data['is_weekend'] = email_data['weekday'].isin(['Saturday', 'Sunday']).astype(int)

# 6. Create weekday encoding with cyclical features
weekday_map = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
    'Friday': 4, 'Saturday': 5, 'Sunday': 6
}
email_data['weekday_num'] = email_data['weekday'].map(weekday_map)
email_data['weekday_sin'] = np.sin(2 * np.pi * email_data['weekday_num']/7)
email_data['weekday_cos'] = np.cos(2 * np.pi * email_data['weekday_num']/7)

# 7. User segmentation using clustering
user_features = email_data[['user_past_purchases']].copy()
scaler = StandardScaler()
user_features_scaled = scaler.fit_transform(user_features)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
email_data['user_cluster'] = kmeans.fit_predict(user_features_scaled)

# 8. Country-specific features
for country in email_data['user_country'].unique():
    email_data[f'country_{country}'] = (email_data['user_country'] == country).astype(int)

# Define features and target
print("\nPreparing model features and target...")
target = 'link_clicked'

# Features to use
categorical_features = ['email_text', 'email_version', 'weekday', 'user_country', 'time_of_day', 
                        'purchase_segment', 'user_cluster']
numerical_features = ['hour', 'user_past_purchases', 'hour_sin', 'hour_cos', 
                      'weekday_sin', 'weekday_cos', 'is_weekend']
binary_features = ['personalized_short', 'personalized_long', 'generic_short', 'generic_long'] + \
                  [f'country_{country}' for country in email_data['user_country'].unique()]

all_features = numerical_features + categorical_features + binary_features

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('bin', 'passthrough', binary_features)
    ])

# Split the data
X = email_data[all_features]
y = email_data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Positive class distribution in training set: {y_train.mean():.4f}")

# Handle class imbalance with SMOTE
print("\nApplying SMOTE to handle class imbalance...")
smote = SMOTE(random_state=42)
X_train_processed = preprocessor.fit_transform(X_train)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

print(f"Original training set shape: {X_train.shape}")
print(f"Resampled training set shape: {X_train_resampled.shape}")
print(f"Original positive class distribution: {y_train.mean():.4f}")
print(f"Resampled positive class distribution: {y_train_resampled.mean():.4f}")

# Train advanced models
print("\nTraining advanced machine learning models...")

# 1. XGBoost
print("Training XGBoost model...")
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    scale_pos_weight=1,
    random_state=42
)
xgb_model.fit(X_train_resampled, y_train_resampled)

# 2. LightGBM
print("Training LightGBM model...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
lgb_model.fit(X_train_resampled, y_train_resampled)

# 3. Random Forest
print("Training Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
rf_model.fit(X_train_resampled, y_train_resampled)

# 4. Gradient Boosting
print("Training Gradient Boosting model...")
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
gb_model.fit(X_train_resampled, y_train_resampled)

# 5. Ensemble (Voting Classifier)
print("Creating ensemble model...")
ensemble_model = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('rf', rf_model),
        ('gb', gb_model)
    ],
    voting='soft'
)
ensemble_model.fit(X_train_resampled, y_train_resampled)

# Evaluate models
print("\nEvaluating models on test set...")
models = {
    'XGBoost': xgb_model,
    'LightGBM': lgb_model,
    'Random Forest': rf_model,
    'Gradient Boosting': gb_model,
    'Ensemble': ensemble_model
}

X_test_processed = preprocessor.transform(X_test)
results = {}

for name, model in models.items():
    # Predict
    y_pred = model.predict(X_test_processed)
    y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
    
    # Calculate metrics
    precision = classification_report(y_test, y_pred, output_dict=True)['1']['precision']
    recall = classification_report(y_test, y_pred, output_dict=True)['1']['recall']
    f1 = classification_report(y_test, y_pred, output_dict=True)['1']['f1-score']
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    # Store results
    results[name] = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_precision': avg_precision,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    # Print results
    print(f"\nResults for {name}:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Select best model based on F1 score
best_model_name = max(results.keys(), key=lambda name: results[name]['f1'])
print(f"\nBest model: {best_model_name} (F1 Score: {results[best_model_name]['f1']:.4f})")

# Feature importance analysis for the best model
if best_model_name in ['XGBoost', 'LightGBM', 'Random Forest', 'Gradient Boosting']:
    best_model = models[best_model_name]
    
    # Get feature names after preprocessing
    cat_encoder = preprocessor.transformers_[1][1]
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
    feature_names = np.concatenate([numerical_features, cat_feature_names, binary_features])
    
    # Get feature importances
    if best_model_name == 'XGBoost':
        importances = best_model.feature_importances_
    elif best_model_name == 'LightGBM':
        importances = best_model.feature_importances_
    elif best_model_name == 'Random Forest':
        importances = best_model.feature_importances_
    else:  # Gradient Boosting
        importances = best_model.feature_importances_
    
    # Sort feature importances
    indices = np.argsort(importances)[::-1]
    
    # Print top 20 features
    print("\nTop 20 most important features:")
    for i in range(min(20, len(feature_names))):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    plt.title(f'Feature Importances ({best_model_name})')
    plt.bar(range(min(20, len(importances))), importances[indices[:20]], align='center')
    plt.xticks(range(min(20, len(importances))), feature_names[indices[:20]], rotation=90)
    plt.tight_layout()
    plt.savefig('advanced_feature_importances.png')

# Save the best model and preprocessor
print("\nSaving best model and preprocessor...")
joblib.dump(models[best_model_name], 'best_email_model.pkl')
joblib.dump(preprocessor, 'email_preprocessor.pkl')

# Generate micro-segments for targeted campaigns
print("\nGenerating micro-segments for targeted campaigns...")

# Create a function to predict click probability
def predict_click_probability(df, model, preprocessor):
    X = df[all_features]
    X_processed = preprocessor.transform(X)
    return model.predict_proba(X_processed)[:, 1]

# Add click probability to the dataset
email_data['click_probability'] = predict_click_probability(email_data, models[best_model_name], preprocessor)

# Create micro-segments
print("Creating micro-segments based on user characteristics and predicted engagement...")

# 1. High-value segments (top 10% click probability)
high_value_threshold = email_data['click_probability'].quantile(0.9)
email_data['high_value_segment'] = (email_data['click_probability'] >= high_value_threshold).astype(int)

# 2. Create engagement segments
email_data['engagement_segment'] = pd.qcut(email_data['click_probability'], 4, 
                                          labels=['Low', 'Medium-Low', 'Medium-High', 'High'])

# 3. Create combined segments
email_data['micro_segment'] = email_data['engagement_segment'].astype(str) + '_' + \
                             email_data['purchase_segment'].astype(str) + '_' + \
                             email_data['user_country']

# Analyze segment performance
segment_performance = email_data.groupby('micro_segment')['link_clicked'].agg(['mean', 'count']).reset_index()
segment_performance = segment_performance.rename(columns={'mean': 'click_rate'})
segment_performance['click_rate_pct'] = segment_performance['click_rate'] * 100
segment_performance = segment_performance.sort_values('click_rate', ascending=False)

# Print top 10 performing segments
print("\nTop 10 performing micro-segments:")
print(segment_performance.head(10))

# Generate optimization recommendations
print("\nGenerating optimization recommendations...")

# 1. Best email type and version by segment
email_type_by_segment = email_data.groupby(['micro_segment', 'email_text', 'email_version'])['link_clicked'].mean().reset_index()
email_type_by_segment = email_type_by_segment.sort_values(['micro_segment', 'link_clicked'], ascending=[True, False])
best_email_type = email_type_by_segment.groupby('micro_segment').first().reset_index()

# 2. Best time to send by segment
time_by_segment = email_data.groupby(['micro_segment', 'time_of_day', 'weekday'])['link_clicked'].mean().reset_index()
time_by_segment = time_by_segment.sort_values(['micro_segment', 'link_clicked'], ascending=[True, False])
best_time = time_by_segment.groupby('micro_segment').first().reset_index()

# Combine recommendations
recommendations = pd.merge(best_email_type, best_time, on='micro_segment')
recommendations = recommendations.rename(columns={
    'link_clicked_x': 'click_rate_email_type',
    'link_clicked_y': 'click_rate_time'
})

# Add segment size
segment_size = email_data.groupby('micro_segment').size().reset_index(name='segment_size')
recommendations = pd.merge(recommendations, segment_size, on='micro_segment')

# Sort by segment size (to prioritize larger segments)
recommendations = recommendations.sort_values('segment_size', ascending=False)

# Print top recommendations
print("\nTop optimization recommendations by segment size:")
print(recommendations[['micro_segment', 'segment_size', 'email_text', 'email_version', 
                      'time_of_day', 'weekday', 'click_rate_email_type', 'click_rate_time']].head(10))

# Calculate potential improvement
current_click_rate = email_data['link_clicked'].mean()
optimized_click_rate = 0

# For each segment, use the best configuration
for _, row in recommendations.iterrows():
    segment = row['micro_segment']
    segment_size = row['segment_size']
    segment_click_rate = max(row['click_rate_email_type'], row['click_rate_time'])
    optimized_click_rate += segment_size * segment_click_rate

optimized_click_rate = optimized_click_rate / len(email_data)
improvement = (optimized_click_rate - current_click_rate) / current_click_rate * 100

print(f"\nCurrent overall click rate: {current_click_rate:.4f} ({current_click_rate*100:.2f}%)")
print(f"Projected optimized click rate: {optimized_click_rate:.4f} ({optimized_click_rate*100:.2f}%)")
print(f"Potential improvement: {improvement:.2f}%")

# Save recommendations for implementation
recommendations.to_csv('email_optimization_recommendations.csv', index=False)
segment_performance.to_csv('micro_segment_performance.csv', index=False)

print("\nAnalysis complete! Recommendations saved to 'email_optimization_recommendations.csv'")
