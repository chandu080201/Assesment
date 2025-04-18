import pandas as pd
import numpy as np
import joblib
import datetime
import json
from pathlib import Path

class EmailCampaignGenerator:
    """
    A class to generate optimized email campaigns based on machine learning predictions.
    This system implements the micro-segmentation approach and provides personalized
    email recommendations for each user.
    """
    
    def __init__(self, model_path='best_email_model.pkl', preprocessor_path='email_preprocessor.pkl',
                 recommendations_path='email_optimization_recommendations.csv'):
        """Initialize the campaign generator with the trained model and preprocessor."""
        print("Initializing Email Campaign Generator...")
        
        # Load the model and preprocessor
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        
        # Load recommendations
        self.recommendations = pd.read_csv(recommendations_path)
        
        # Define feature columns needed for prediction
        self.numerical_features = ['hour', 'user_past_purchases', 'hour_sin', 'hour_cos', 
                                  'weekday_sin', 'weekday_cos', 'is_weekend']
        self.categorical_features = ['email_text', 'email_version', 'weekday', 'user_country', 
                                    'time_of_day', 'purchase_segment', 'user_cluster']
        self.binary_features = ['personalized_short', 'personalized_long', 'generic_short', 'generic_long',
                               'country_US', 'country_UK', 'country_FR', 'country_ES']
        
        self.all_features = self.numerical_features + self.categorical_features + self.binary_features
        
        print("Campaign Generator initialized successfully!")
    
    def _preprocess_user_data(self, user_data):
        """Preprocess user data to prepare for prediction."""
        # Create a copy to avoid modifying the original
        df = user_data.copy()
        
        # Add cyclical time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
        
        # Create time of day categories
        df['time_of_day'] = pd.cut(
            df['hour'], 
            bins=[0, 6, 12, 18, 24], 
            labels=['Night', 'Morning', 'Afternoon', 'Evening']
        )
        
        # Create purchase history segments
        df['purchase_segment'] = pd.cut(
            df['user_past_purchases'], 
            bins=[-1, 0, 3, 7, 100], 
            labels=['No purchases', '1-3 purchases', '4-7 purchases', '8+ purchases']
        )
        
        # Create interaction features
        df['personalized_short'] = ((df['email_text'] == 'short_email') & 
                                   (df['email_version'] == 'personalized')).astype(int)
        df['personalized_long'] = ((df['email_text'] == 'long_email') & 
                                  (df['email_version'] == 'personalized')).astype(int)
        df['generic_short'] = ((df['email_text'] == 'short_email') & 
                              (df['email_version'] == 'generic')).astype(int)
        df['generic_long'] = ((df['email_text'] == 'long_email') & 
                             (df['email_version'] == 'generic')).astype(int)
        
        # Day type feature
        df['is_weekend'] = df['weekday'].isin(['Saturday', 'Sunday']).astype(int)
        
        # Create weekday encoding with cyclical features
        weekday_map = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
            'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }
        df['weekday_num'] = df['weekday'].map(weekday_map)
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday_num']/7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday_num']/7)
        
        # Country-specific features
        for country in ['US', 'UK', 'FR', 'ES']:
            df[f'country_{country}'] = (df['user_country'] == country).astype(int)
        
        # Assign user cluster (simplified approach - in production would use the actual clustering model)
        # Here we're using a simple rule-based approach for demonstration
        df['user_cluster'] = pd.cut(
            df['user_past_purchases'],
            bins=[-1, 1, 4, 8, 100],
            labels=[0, 1, 2, 3]
        ).astype(int)
        
        return df
    
    def predict_click_probability(self, user_data):
        """Predict the probability of a user clicking on an email link."""
        # Preprocess the data
        processed_data = self._preprocess_user_data(user_data)
        
        # Extract features for prediction
        X = processed_data[self.all_features]
        
        # Transform using the preprocessor
        X_processed = self.preprocessor.transform(X)
        
        # Predict probabilities
        click_probs = self.model.predict_proba(X_processed)[:, 1]
        
        # Add predictions to the data
        processed_data['click_probability'] = click_probs
        
        return processed_data
    
    def get_optimal_campaign_parameters(self, user_data):
        """
        For each user, determine the optimal email parameters (type, version, time, day)
        that maximize the probability of clicking.
        """
        # Generate all possible combinations for each user
        users = user_data[['email_id', 'user_country', 'user_past_purchases']].drop_duplicates()
        
        # Create empty list to store results
        all_combinations = []
        
        # Email parameters to test
        email_texts = ['short_email', 'long_email']
        email_versions = ['personalized', 'generic']
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        hours = [9, 12, 15, 18]  # 9am, 12pm, 3pm, 6pm
        
        # Generate all combinations for each user
        for _, user in users.iterrows():
            for email_text in email_texts:
                for email_version in email_versions:
                    for weekday in weekdays:
                        for hour in hours:
                            all_combinations.append({
                                'email_id': user['email_id'],
                                'user_country': user['user_country'],
                                'user_past_purchases': user['user_past_purchases'],
                                'email_text': email_text,
                                'email_version': email_version,
                                'weekday': weekday,
                                'hour': hour
                            })
        
        # Convert to DataFrame
        combinations_df = pd.DataFrame(all_combinations)
        
        # Predict click probability for all combinations
        predictions = self.predict_click_probability(combinations_df)
        
        # For each user, find the combination with the highest click probability
        optimal_params = predictions.loc[predictions.groupby('email_id')['click_probability'].idxmax()]
        
        # Select relevant columns
        result = optimal_params[['email_id', 'user_country', 'user_past_purchases', 
                                'email_text', 'email_version', 'weekday', 'hour', 
                                'click_probability']]
        
        return result
    
    def generate_campaign_schedule(self, user_data, campaign_name, start_date=None):
        """
        Generate a complete email campaign schedule with personalized parameters for each user.
        
        Parameters:
        -----------
        user_data : DataFrame
            User data containing email_id, user_country, and user_past_purchases
        campaign_name : str
            Name of the campaign
        start_date : str, optional
            Start date for the campaign in 'YYYY-MM-DD' format. If None, uses current date.
            
        Returns:
        --------
        DataFrame with complete campaign schedule
        """
        if start_date is None:
            start_date = datetime.datetime.now().strftime('%Y-%m-%d')
        
        # Get optimal parameters for each user
        print(f"Generating optimal campaign parameters for {len(user_data)} users...")
        optimal_params = self.get_optimal_campaign_parameters(user_data)
        
        # Create campaign schedule
        campaign_schedule = optimal_params.copy()
        campaign_schedule['campaign_name'] = campaign_name
        campaign_schedule['start_date'] = start_date
        
        # Calculate send date based on weekday
        start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        weekday_map = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
            'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }
        
        # Calculate days to add to reach the target weekday
        campaign_schedule['days_to_add'] = campaign_schedule['weekday'].map(weekday_map) - start_date_obj.weekday()
        campaign_schedule.loc[campaign_schedule['days_to_add'] < 0, 'days_to_add'] += 7
        
        # Calculate send date
        campaign_schedule['send_date'] = campaign_schedule.apply(
            lambda row: (start_date_obj + datetime.timedelta(days=int(row['days_to_add']))).strftime('%Y-%m-%d'),
            axis=1
        )
        
        # Format send time
        campaign_schedule['send_time'] = campaign_schedule['hour'].apply(lambda x: f"{x:02d}:00:00")
        
        # Create full send datetime
        campaign_schedule['send_datetime'] = campaign_schedule['send_date'] + ' ' + campaign_schedule['send_time']
        
        # Clean up intermediate columns
        campaign_schedule = campaign_schedule.drop(['days_to_add'], axis=1)
        
        # Sort by send datetime
        campaign_schedule = campaign_schedule.sort_values('send_datetime')
        
        print(f"Campaign schedule generated with {len(campaign_schedule)} personalized emails")
        return campaign_schedule
    
    def export_campaign(self, campaign_schedule, output_dir='campaigns'):
        """Export the campaign schedule to CSV and JSON formats."""
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Get campaign name and create sanitized filename
        campaign_name = campaign_schedule['campaign_name'].iloc[0]
        filename_base = campaign_name.lower().replace(' ', '_')
        
        # Export to CSV
        csv_path = f"{output_dir}/{filename_base}_schedule.csv"
        campaign_schedule.to_csv(csv_path, index=False)
        
        # Export to JSON (for API integration)
        json_path = f"{output_dir}/{filename_base}_schedule.json"
        
        # Convert to JSON-friendly format
        campaign_json = {
            "campaign_name": campaign_name,
            "generated_date": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_recipients": len(campaign_schedule),
            "start_date": campaign_schedule['start_date'].iloc[0],
            "emails": campaign_schedule.to_dict(orient='records')
        }
        
        with open(json_path, 'w') as f:
            json.dump(campaign_json, f, indent=2)
        
        print(f"Campaign exported to:")
        print(f"- CSV: {csv_path}")
        print(f"- JSON: {json_path}")
        
        return csv_path, json_path
    
    def generate_campaign_summary(self, campaign_schedule):
        """Generate a summary of the campaign with key metrics and insights."""
        summary = {
            "campaign_name": campaign_schedule['campaign_name'].iloc[0],
            "total_recipients": len(campaign_schedule),
            "start_date": campaign_schedule['start_date'].iloc[0],
            "end_date": campaign_schedule['send_date'].max(),
            "avg_click_probability": campaign_schedule['click_probability'].mean(),
            "email_type_distribution": campaign_schedule['email_text'].value_counts().to_dict(),
            "email_version_distribution": campaign_schedule['email_version'].value_counts().to_dict(),
            "weekday_distribution": campaign_schedule['weekday'].value_counts().to_dict(),
            "hour_distribution": campaign_schedule['hour'].value_counts().to_dict(),
            "country_distribution": campaign_schedule['user_country'].value_counts().to_dict()
        }
        
        # Calculate expected clicks
        summary["expected_clicks"] = int(campaign_schedule['click_probability'].sum())
        summary["expected_click_rate"] = summary["expected_clicks"] / summary["total_recipients"]
        
        return summary

# Example usage function
def run_example():
    """Run an example campaign generation."""
    print("Loading sample user data...")
    
    # Load the original data for demonstration
    email_data = pd.read_csv('email_table.csv')
    
    # Take a sample of users for the example
    sample_users = email_data[['email_id', 'user_country', 'user_past_purchases']].sample(1000, random_state=42)
    
    # Initialize the campaign generator
    generator = EmailCampaignGenerator()
    
    # Generate a campaign schedule
    campaign_schedule = generator.generate_campaign_schedule(
        sample_users, 
        campaign_name="Optimized Feature Announcement",
        start_date="2023-06-01"
    )
    
    # Export the campaign
    generator.export_campaign(campaign_schedule)
    
    # Generate and print campaign summary
    summary = generator.generate_campaign_summary(campaign_schedule)
    print("\nCampaign Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Print sample of the schedule
    print("\nSample of campaign schedule (first 5 emails):")
    print(campaign_schedule.head(5))
    
    return campaign_schedule

if __name__ == "__main__":
    run_example()
