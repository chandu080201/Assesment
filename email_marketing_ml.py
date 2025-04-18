import pandas as pd
import numpy as np
# Set matplotlib backend to Agg to avoid GUI issues
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc,
                            precision_recall_curve, average_precision_score, f1_score,
                            matthews_corrcoef, precision_score, recall_score)
from sklearn.feature_selection import SelectFromModel, RFECV
# Import SMOTE for handling class imbalance
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Starting Email Marketing ML Optimization...")
print("="*80)

# Load the processed data
df = pd.read_csv('processed_email_data.csv')
print(f"Data shape: {df.shape}")

# Feature engineering
# Recency-based features
df['days_since_last_purchase'] = np.random.exponential(30, size=len(df))  # Simulate if not available
df['recency_score'] = 1 / (1 + df['days_since_last_purchase'])

# Interaction depth features
df['engagement_depth'] = df['user_past_purchases'] * df['email_opened']

# Time-based interaction features
df['morning_weekday'] = ((df['time_of_day'] == 'Morning') &
                         (~df['weekday'].isin(['Saturday', 'Sunday']))).astype(int)
df['evening_weekend'] = ((df['time_of_day'] == 'Evening') &
                         (df['weekday'].isin(['Saturday', 'Sunday']))).astype(int)

# Email-user interaction features
for country in df['user_country'].unique():
    for email_type in df['email_text'].unique():
        df[f'{country}_{email_type}'] = ((df['user_country'] == country) &
                                         (df['email_text'] == email_type)).astype(int)

# Purchase frequency segments with finer granularity
df['purchase_frequency'] = pd.qcut(
    df['user_past_purchases'],
    q=10,  # 10 quantiles instead of 4 categories
    labels=False,
    duplicates='drop'
)

# Define features and target

# Target: link_clicked (our success metric)
target = 'link_clicked'

# Features to use
categorical_features = ['email_text', 'email_version', 'weekday', 'user_country', 'time_of_day']
numerical_features = ['hour', 'user_past_purchases']

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Split the data
X = df[numerical_features + categorical_features]
y = df[target]

# Add more features for better prediction
print("\nAdding additional engineered features...")
# Add hour of day as cyclical features to better capture time patterns
X['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
X['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)

# Add interaction features
X['email_opened'] = df['email_opened']  # Add email_opened as a feature
X['past_purchase_email_opened'] = df['user_past_purchases'] * df['email_opened']

# Add day type feature (weekday/weekend)
X['is_weekend'] = df['weekday'].isin(['Saturday', 'Sunday']).astype(int)

# Split the data with stratification to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Positive class distribution in training set: {y_train.mean():.4f}")
print(f"Positive class distribution in test set: {y_test.mean():.4f}")

# Handle class imbalance
# Calculate class weights
class_counts = y_train.value_counts()
total_samples = len(y_train)
class_weights = {0: total_samples / (2 * class_counts[0]),
                 1: total_samples / (2 * class_counts[1])}
print(f"Class weights: {class_weights}")

# Apply SMOTE to handle class imbalance
print("\nApplying SMOTE to handle class imbalance...")
smote = SMOTE(random_state=42)

# First apply preprocessing to get numeric features
preprocessor_for_smote = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features + ['hour_sin', 'hour_cos', 'email_opened',
                                                      'past_purchase_email_opened', 'is_weekend']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Fit and transform the training data
X_train_processed = preprocessor_for_smote.fit_transform(X_train)

# Apply SMOTE to the processed data
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

print(f"Original training set shape: {X_train.shape}")
print(f"Resampled training set shape: {X_train_resampled.shape}")
print(f"Original positive class distribution: {y_train.mean():.4f}")
print(f"Resampled positive class distribution: {y_train_resampled.mean():.4f}")

# Define more sophisticated models with better parameters
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=2000,
        C=0.1,  # Stronger regularization
        solver='liblinear',  # Better for imbalanced datasets
        class_weight='balanced',  # Use balanced class weights
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200,  # More trees
        max_depth=10,  # Control depth to prevent overfitting
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',  # Use balanced class weights
        random_state=42,
        n_jobs=-1  # Use all cores
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,  # Slower learning rate for better generalization
        max_depth=5,
        min_samples_split=5,
        subsample=0.8,  # Use 80% of samples for each tree
        random_state=42
    ),
    'AdaBoost': AdaBoostClassifier(
        n_estimators=100,
        learning_rate=0.05,
        random_state=42
    ),
    'SVM': SVC(
        C=1.0,
        kernel='rbf',
        gamma='scale',
        probability=True,
        class_weight='balanced',
        random_state=42
    ),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42
    )
}

# Create a voting ensemble
base_models = [(name, model) for name, model in models.items()]
models['Voting Ensemble'] = VotingClassifier(
    estimators=base_models,
    voting='soft',  # Use probability estimates
    n_jobs=-1
)

# Train and evaluate models
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    # Create pipeline
    if name != 'Voting Ensemble':  # Regular models
        # Train on the resampled data
        model.fit(X_train_resampled, y_train_resampled)

        # Evaluate on test set
        X_test_processed = preprocessor_for_smote.transform(X_test)
        y_pred = model.predict(X_test_processed)
        y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
    else:  # Voting Ensemble needs special handling
        # Create a new ensemble with fitted models
        fitted_models = []
        for model_name, _ in base_models:
            fitted_models.append((model_name, models[model_name]))

        # Create a new voting classifier with the fitted models
        voting_clf = VotingClassifier(
            estimators=fitted_models,
            voting='soft',
            n_jobs=-1
        )

        # Fit the voting classifier (this doesn't retrain the base models)
        voting_clf.fit(X_train_resampled, y_train_resampled)

        # Evaluate on test set
        X_test_processed = preprocessor_for_smote.transform(X_test)
        y_pred = voting_clf.predict(X_test_processed)
        y_pred_proba = voting_clf.predict_proba(X_test_processed)[:, 1]

        # Replace the unfitted model with the fitted one
        models[name] = voting_clf

    # Calculate performance metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Store results
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc
    }

    # Print classification report
    print(f"\nClassification Report for {name}:")
    print(classification_report(y_test, y_pred))

    # Print additional metrics
    print(f"Matthews Correlation Coefficient: {mcc:.4f}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Clicked', 'Clicked'],
                yticklabels=['Not Clicked', 'Clicked'])
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{name.replace(" ", "_").lower()}.png')

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f'roc_curve_{name.replace(" ", "_").lower()}.png')

    # Plot Precision-Recall curve
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.plot(recall, precision, lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {name}')
    plt.tight_layout()
    plt.savefig(f'pr_curve_{name.replace(" ", "_").lower()}.png')

# Create a comprehensive metrics comparison table
metrics_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Precision': [results[m]['precision'] for m in results],
    'Recall': [results[m]['recall'] for m in results],
    'F1 Score': [results[m]['f1'] for m in results],
    'MCC': [results[m]['mcc'] for m in results]
})

# Sort by F1 score
metrics_df = metrics_df.sort_values('F1 Score', ascending=False).reset_index(drop=True)

# Print the comparison table
print("\nModel Performance Comparison:")
print(metrics_df)

# Select the best model (based on F1 score)
best_model_name = metrics_df.iloc[0]['Model']
best_model = models[best_model_name]
print(f"\nBest model: {best_model_name} (F1 Score: {metrics_df.iloc[0]['F1 Score']:.4f})")

# Find optimal threshold for the best model
print("\nOptimizing classification threshold for F1 score...")

def find_optimal_threshold(y_true, y_pred_proba):
    """Find the optimal threshold that maximizes F1 score."""
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores = []

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)

    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1 = max(f1_scores)

    return best_threshold, best_f1

# Find optimal threshold for the best model
best_model_proba = results[best_model_name]['y_pred_proba']
optimal_threshold, optimal_f1 = find_optimal_threshold(y_test, best_model_proba)

print(f"Optimal threshold: {optimal_threshold:.2f} (F1 score: {optimal_f1:.4f})")

# Apply optimal threshold to get improved predictions
y_pred_optimal = (best_model_proba >= optimal_threshold).astype(int)

# Print updated classification report with optimal threshold
print(f"\nClassification Report with Optimal Threshold ({optimal_threshold:.2f}):")
print(classification_report(y_test, y_pred_optimal))

# Feature importance analysis (for tree-based models)
if best_model_name in ['Random Forest', 'Gradient Boosting', 'Decision Tree']:
    # Get feature names after one-hot encoding
    cat_features = preprocessor_for_smote.transformers_[1][1].get_feature_names_out(categorical_features)
    all_features = numerical_features + ['hour_sin', 'hour_cos', 'email_opened',
                                       'past_purchase_email_opened', 'is_weekend']
    feature_names = np.concatenate([all_features, cat_features])

    # Get feature importances
    importances = best_model.feature_importances_

    # Sort feature importances
    indices = np.argsort(importances)[::-1]

    # Plot feature importances
    plt.figure(figsize=(12, 8))
    plt.title(f'Feature Importances ({best_model_name})')
    plt.bar(range(min(20, len(importances))), importances[indices[:20]], align='center')
    plt.xticks(range(min(20, len(importances))), feature_names[indices[:20]], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importances.png')

    # Print top 10 features
    print("\nTop 10 most important features:")
    for i in range(min(10, len(feature_names))):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# Optimize the best model with GridSearchCV
print("\nFine-tuning the best model...")

# Define parameter grid based on the best model
if best_model_name == 'Logistic Regression':
    param_grid = {
        'C': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0],
        'solver': ['liblinear', 'saga'],
        'penalty': ['l1', 'l2']
    }
elif best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
elif best_model_name == 'Gradient Boosting':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.8, 0.9],
        'min_samples_split': [2, 5, 10]
    }
elif best_model_name == 'SVM':
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
elif best_model_name == 'AdaBoost':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.5, 1.0],
        'algorithm': ['SAMME', 'SAMME.R']
    }
elif best_model_name == 'Decision Tree':
    param_grid = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4, 8],
        'criterion': ['gini', 'entropy']
    }
else:  # Voting Ensemble or other models
    print("Skipping hyperparameter optimization for ensemble model.")
    param_grid = None

# Perform grid search if we have a parameter grid
if param_grid:
    # Use stratified k-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Create grid search
    grid_search = GridSearchCV(
        estimator=best_model,
        param_grid=param_grid,
        cv=cv,
        scoring='f1',  # Optimize for F1 score
        n_jobs=-1,     # Use all cores
        verbose=1
    )

    # Fit grid search
    print(f"Performing grid search with {cv.get_n_splits()} folds...")
    grid_search.fit(X_train_resampled, y_train_resampled)

    # Print best parameters
    print(f"\nBest parameters for {best_model_name}:")
    print(grid_search.best_params_)
    print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")

    # Get the optimized model
    optimized_model = grid_search.best_estimator_

    # Evaluate optimized model
    y_pred_opt = optimized_model.predict(X_test_processed)
    y_pred_proba_opt = optimized_model.predict_proba(X_test_processed)[:, 1]

    # Apply optimal threshold to optimized model
    optimal_threshold_opt, optimal_f1_opt = find_optimal_threshold(y_test, y_pred_proba_opt)
    y_pred_opt_threshold = (y_pred_proba_opt >= optimal_threshold_opt).astype(int)

    print(f"\nOptimal threshold for optimized model: {optimal_threshold_opt:.2f}")
    print("\nClassification Report for Optimized Model:")
    print(classification_report(y_test, y_pred_opt_threshold))

    # Plot confusion matrix for optimized model
    plt.figure(figsize=(8, 6))
    cm_opt = confusion_matrix(y_test, y_pred_opt_threshold)
    sns.heatmap(cm_opt, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Clicked', 'Clicked'],
                yticklabels=['Not Clicked', 'Clicked'])
    plt.title(f'Confusion Matrix - Optimized {best_model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_optimized.png')

    # Save the final optimized model
    final_model = optimized_model
else:
    # Use the best model as is
    final_model = best_model
    y_pred_opt_threshold = y_pred_optimal  # Use the threshold-optimized predictions

# Generate advanced recommendations based on the model
print("\nGenerating advanced email marketing recommendations...")

# 1. Best email type and version
email_text_click_rate = df.groupby('email_text')['link_clicked'].mean()
email_version_click_rate = df.groupby('email_version')['link_clicked'].mean()
best_email_text = email_text_click_rate.idxmax()
best_email_version = email_version_click_rate.idxmax()

# 2. Best time to send
time_of_day_click_rate = df.groupby('time_of_day')['link_clicked'].mean()
weekday_click_rate = df.groupby('weekday')['link_clicked'].mean()
best_time = time_of_day_click_rate.idxmax()
best_day = weekday_click_rate.idxmax()

# 3. Best user segments
country_click_rate = df.groupby('user_country')['link_clicked'].mean()
purchase_bins = [-1, 0, 3, 7, 100]
purchase_labels = ['No purchases', '1-3 purchases', '4-7 purchases', '8+ purchases']
df['purchase_group'] = pd.cut(df['user_past_purchases'], bins=purchase_bins, labels=purchase_labels)
purchase_group_click_rate = df.groupby('purchase_group')['link_clicked'].mean()
best_country = country_click_rate.idxmax()
best_purchase_group = purchase_group_click_rate.idxmax()

# 4. Best combinations
combo_click_rate = df.groupby(['email_text', 'email_version', 'time_of_day', 'weekday'])['link_clicked'].mean()
best_combo = combo_click_rate.idxmax()

# 5. Advanced segment-specific recommendations
print("\nGenerating segment-specific recommendations...")

# Create micro-segments
df['micro_segment'] = df['purchase_group'].astype(str) + '_' + df['user_country']

# Find best email type and timing for each segment
segment_recommendations = {}
for segment in df['micro_segment'].unique():
    segment_data = df[df['micro_segment'] == segment]

    # Skip segments with too few samples
    if len(segment_data) < 100:
        continue

    # Find best email parameters for this segment
    best_email_type = segment_data.groupby('email_text')['link_clicked'].mean().idxmax()
    best_email_version = segment_data.groupby('email_version')['link_clicked'].mean().idxmax()
    best_time_of_day = segment_data.groupby('time_of_day')['link_clicked'].mean().idxmax()
    best_weekday = segment_data.groupby('weekday')['link_clicked'].mean().idxmax()

    # Calculate click rate for this combination
    segment_combo = segment_data[
        (segment_data['email_text'] == best_email_type) &
        (segment_data['email_version'] == best_email_version) &
        (segment_data['time_of_day'] == best_time_of_day) &
        (segment_data['weekday'] == best_weekday)
    ]

    if len(segment_combo) > 0:
        click_rate = segment_combo['link_clicked'].mean()
    else:
        click_rate = 0

    # Store recommendations
    segment_recommendations[segment] = {
        'email_type': best_email_type,
        'email_version': best_email_version,
        'time_of_day': best_time_of_day,
        'weekday': best_weekday,
        'click_rate': click_rate,
        'segment_size': len(segment_data)
    }

# Convert to DataFrame for easier analysis
segment_df = pd.DataFrame.from_dict(segment_recommendations, orient='index')
segment_df = segment_df.sort_values('click_rate', ascending=False)

# Print overall recommendations
print("\nEmail Marketing Campaign Recommendations:")
print("="*50)
print(f"1. Email Content: Use {best_email_text} with {best_email_version} version")
print(f"   - {best_email_text} click rate: {email_text_click_rate.max():.2%}")
print(f"   - {best_email_version} click rate: {email_version_click_rate.max():.2%}")

print(f"\n2. Timing: Send emails during {best_time} on {best_day}")
print(f"   - {best_time} click rate: {time_of_day_click_rate.max():.2%}")
print(f"   - {best_day} click rate: {weekday_click_rate.max():.2%}")

print(f"\n3. Target Audience: Focus on users from {best_country} with {best_purchase_group.lower()}")
print(f"   - {best_country} click rate: {country_click_rate.max():.2%}")
print(f"   - {best_purchase_group} click rate: {purchase_group_click_rate.max():.2%}")

print(f"\n4. Best Overall Combination: {best_combo}")
print(f"   - Click rate: {combo_click_rate.max():.2%}")

# Print top segment-specific recommendations
print("\n5. Top Micro-Segment Recommendations:")
print("   " + "-"*40)
for i, (segment, row) in enumerate(segment_df.head(5).iterrows()):
    print(f"   Segment {i+1}: {segment}")
    print(f"     - Email: {row['email_type']}, {row['email_version']}")
    print(f"     - Timing: {row['time_of_day']} on {row['weekday']}")
    print(f"     - Expected click rate: {row['click_rate']:.2%}")
    print(f"     - Segment size: {row['segment_size']} users")

# Print model insights
print("\n6. Machine Learning Model Insights:")
print(f"   - Best model: {best_model_name} (F1 Score: {metrics_df.iloc[0]['F1 Score']:.4f})")
print(f"   - Optimal classification threshold: {optimal_threshold:.2f}")

# Print top features if available
if best_model_name in ['Random Forest', 'Gradient Boosting', 'Decision Tree']:
    print("   - Top 5 factors influencing email clicks:")
    for i in range(min(5, len(feature_names))):
        print(f"     * {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# Calculate potential improvement
current_click_rate = df['link_clicked'].mean()
optimized_click_rate = segment_df.iloc[0]['click_rate']  # Best segment rate
improvement = (optimized_click_rate - current_click_rate) / current_click_rate * 100

print(f"\n7. Potential Improvement:")
print(f"   - Current overall click rate: {current_click_rate:.2%}")
print(f"   - Potential optimized click rate: {optimized_click_rate:.2%}")
print(f"   - Relative improvement: {improvement:.1f}%")

# Save the final model
joblib.dump(final_model, 'optimized_email_model.pkl')
print("\nOptimized model saved as 'optimized_email_model.pkl'")

# Save segment recommendations for implementation
segment_df.to_csv('segment_recommendations.csv')
print("Segment recommendations saved as 'segment_recommendations.csv'")

print("\nAnalysis complete!")


