import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import datetime
import joblib
from pathlib import Path

# Import our custom modules
from email_campaign_generator import EmailCampaignGenerator
from ab_testing_framework import ABTestingFramework

class EmailMarketingOptimizationSystem:
    """
    A comprehensive system for optimizing email marketing campaigns using machine learning.
    This system integrates data preprocessing, model training, campaign generation, and A/B testing.
    """
    
    def __init__(self, data_dir='.'):
        """Initialize the email marketing optimization system."""
        self.data_dir = data_dir
        self.output_dir = 'email_optimization_output'
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        print("Initializing Email Marketing Optimization System...")
        
        # Check if required files exist
        required_files = [
            'email_table.csv',
            'email_opened_table.csv',
            'link_clicked_table.csv'
        ]
        
        for file in required_files:
            file_path = os.path.join(data_dir, file)
            if not os.path.exists(file_path):
                print(f"Warning: Required file {file} not found at {file_path}")
    
    def load_data(self):
        """Load and preprocess the email campaign data."""
        print("Loading and preprocessing data...")
        
        # Load the datasets
        email_data = pd.read_csv(os.path.join(self.data_dir, 'email_table.csv'))
        email_opened = pd.read_csv(os.path.join(self.data_dir, 'email_opened_table.csv'))
        link_clicked = pd.read_csv(os.path.join(self.data_dir, 'link_clicked_table.csv'))
        
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
        
        # 3. Weekend flag
        email_data['is_weekend'] = email_data['weekday'].isin(['Saturday', 'Sunday']).astype(int)
        
        # Save processed data
        processed_data_path = os.path.join(self.output_dir, 'processed_email_data.csv')
        email_data.to_csv(processed_data_path, index=False)
        print(f"Processed data saved to {processed_data_path}")
        
        return email_data
    
    def run_exploratory_analysis(self, data):
        """Run exploratory data analysis on the email campaign data."""
        print("Running exploratory data analysis...")
        
        # Set plot style
        plt.style.use('ggplot')
        sns.set(style="whitegrid")
        
        # 1. Overall metrics by email type and version
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.barplot(x='email_text', y='link_clicked', data=data)
        plt.title('Click Rate by Email Type')
        plt.xlabel('Email Type')
        plt.ylabel('Click Rate')
        
        plt.subplot(1, 2, 2)
        sns.barplot(x='email_version', y='link_clicked', data=data)
        plt.title('Click Rate by Email Version')
        plt.xlabel('Email Version')
        plt.ylabel('Click Rate')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'email_type_analysis.png'))
        
        # 2. Time-based analysis
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.barplot(x='time_of_day', y='link_clicked', data=data)
        plt.title('Click Rate by Time of Day')
        plt.xlabel('Time of Day')
        plt.ylabel('Click Rate')
        
        plt.subplot(1, 2, 2)
        sns.barplot(x='weekday', y='link_clicked', data=data, 
                   order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        plt.title('Click Rate by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Click Rate')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'time_analysis.png'))
        
        # 3. User demographics analysis
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.barplot(x='user_country', y='link_clicked', data=data)
        plt.title('Click Rate by Country')
        plt.xlabel('Country')
        plt.ylabel('Click Rate')
        
        plt.subplot(1, 2, 2)
        sns.barplot(x='purchase_segment', y='link_clicked', data=data)
        plt.title('Click Rate by Purchase History')
        plt.xlabel('Purchase Segment')
        plt.ylabel('Click Rate')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'user_demographics.png'))
        
        # 4. Combined factors analysis
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.barplot(x='email_text', y='link_clicked', hue='email_version', data=data)
        plt.title('Click Rate by Email Text and Version')
        plt.xlabel('Email Text')
        plt.ylabel('Click Rate')
        plt.legend(title='Email Version')
        
        plt.subplot(1, 2, 2)
        sns.barplot(x='time_of_day', y='link_clicked', hue='is_weekend', data=data)
        plt.title('Click Rate by Time of Day and Day Type')
        plt.xlabel('Time of Day')
        plt.ylabel('Click Rate')
        plt.legend(title='Weekend', labels=['Weekday', 'Weekend'])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'combined_factors.png'))
        
        # 5. Generate summary statistics
        summary = {
            "overall": {
                "open_rate": data['email_opened'].mean(),
                "click_rate": data['link_clicked'].mean(),
                "click_through_rate": data.loc[data['email_opened'] == 1, 'link_clicked'].mean()
            },
            "email_type": data.groupby('email_text')['link_clicked'].mean().to_dict(),
            "email_version": data.groupby('email_version')['link_clicked'].mean().to_dict(),
            "time_of_day": data.groupby('time_of_day')['link_clicked'].mean().to_dict(),
            "weekday": data.groupby('weekday')['link_clicked'].mean().to_dict(),
            "country": data.groupby('user_country')['link_clicked'].mean().to_dict(),
            "purchase_segment": data.groupby('purchase_segment')['link_clicked'].mean().to_dict()
        }
        
        # Find best combinations
        best_combos = data.groupby(['email_text', 'email_version', 'time_of_day', 'weekday'])['link_clicked'].mean()
        top_5_combos = best_combos.nlargest(5)
        
        summary["top_combinations"] = {str(combo): rate for combo, rate in zip(top_5_combos.index, top_5_combos.values)}
        
        # Print summary
        print("\nEDA Summary:")
        print(f"Overall click rate: {summary['overall']['click_rate']:.2%}")
        print("\nBest performing factors:")
        print(f"Email type: {max(summary['email_type'].items(), key=lambda x: x[1])[0]}")
        print(f"Email version: {max(summary['email_version'].items(), key=lambda x: x[1])[0]}")
        print(f"Time of day: {max(summary['time_of_day'].items(), key=lambda x: x[1])[0]}")
        print(f"Day of week: {max(summary['weekday'].items(), key=lambda x: x[1])[0]}")
        print(f"Country: {max(summary['country'].items(), key=lambda x: x[1])[0]}")
        print(f"Purchase segment: {max(summary['purchase_segment'].items(), key=lambda x: x[1])[0]}")
        
        print("\nTop combination:")
        top_combo, top_rate = max(summary["top_combinations"].items(), key=lambda x: x[1])
        print(f"{top_combo}: {top_rate:.2%}")
        
        return summary
    
    def train_model(self, data=None):
        """Train the machine learning model for email optimization."""
        print("Training machine learning model...")
        
        # Check if we need to load data
        if data is None:
            try:
                data = pd.read_csv(os.path.join(self.output_dir, 'processed_email_data.csv'))
            except:
                data = self.load_data()
        
        # Run the advanced model training script
        try:
            import advanced_email_optimization
            print("Advanced email optimization model training completed.")
            
            # Check if model files were created
            model_path = 'best_email_model.pkl'
            preprocessor_path = 'email_preprocessor.pkl'
            
            if os.path.exists(model_path) and os.path.exists(preprocessor_path):
                print(f"Model saved to {model_path}")
                print(f"Preprocessor saved to {preprocessor_path}")
                return True
            else:
                print("Warning: Model files not found after training.")
                return False
            
        except Exception as e:
            print(f"Error training advanced model: {str(e)}")
            print("Falling back to basic model training...")
            
            # Implement basic model training here if needed
            # For now, we'll just return False to indicate failure
            return False
    
    def generate_campaign(self, user_data=None, campaign_name="Optimized Email Campaign"):
        """Generate an optimized email campaign."""
        print("Generating optimized email campaign...")
        
        # Check if we need to load user data
        if user_data is None:
            try:
                # Load the original data and extract user information
                email_data = pd.read_csv(os.path.join(self.data_dir, 'email_table.csv'))
                user_data = email_data[['email_id', 'user_country', 'user_past_purchases']].drop_duplicates()
                
                # Take a sample for demonstration
                user_data = user_data.sample(min(5000, len(user_data)), random_state=42)
                
            except Exception as e:
                print(f"Error loading user data: {str(e)}")
                return None
        
        # Initialize the campaign generator
        try:
            generator = EmailCampaignGenerator()
            
            # Generate campaign schedule
            campaign_schedule = generator.generate_campaign_schedule(
                user_data,
                campaign_name=campaign_name,
                start_date=datetime.datetime.now().strftime('%Y-%m-%d')
            )
            
            # Export the campaign
            csv_path, json_path = generator.export_campaign(
                campaign_schedule,
                output_dir=os.path.join(self.output_dir, 'campaigns')
            )
            
            # Generate campaign summary
            summary = generator.generate_campaign_summary(campaign_schedule)
            
            print("\nCampaign Generation Summary:")
            print(f"Total recipients: {summary['total_recipients']}")
            print(f"Expected click rate: {summary['expected_click_rate']:.2%}")
            print(f"Expected clicks: {summary['expected_clicks']}")
            
            return campaign_schedule
            
        except Exception as e:
            print(f"Error generating campaign: {str(e)}")
            return None
    
    def setup_ab_test(self, user_data=None, test_name="Optimized vs Standard Email Test"):
        """Set up an A/B test to validate the optimization approach."""
        print("Setting up A/B test...")
        
        # Check if we need to load user data
        if user_data is None:
            try:
                # Load the original data and extract user information
                email_data = pd.read_csv(os.path.join(self.data_dir, 'email_table.csv'))
                user_data = email_data[['email_id', 'user_country', 'user_past_purchases']].drop_duplicates()
                
                # Add user segments
                user_data['purchase_segment'] = pd.cut(
                    user_data['user_past_purchases'],
                    bins=[-1, 0, 3, 7, 100],
                    labels=['No purchases', '1-3 purchases', '4-7 purchases', '8+ purchases']
                )
                user_data['user_segment'] = user_data['purchase_segment']
                
            except Exception as e:
                print(f"Error loading user data: {str(e)}")
                return None
        
        # Initialize the A/B testing framework
        try:
            ab_framework = ABTestingFramework()
            
            # Design the test
            test_design = ab_framework.design_test(
                user_data,
                test_name=test_name,
                control_size=0.2
            )
            
            # Export the test design
            csv_path, json_path = ab_framework.export_test_design(
                test_design,
                output_dir=os.path.join(self.output_dir, 'ab_tests')
            )
            
            print(f"A/B test design exported to {csv_path}")
            
            # For demonstration, simulate test results
            test_results = ab_framework.simulate_test_results(
                test_design,
                control_open_rate=0.10,
                control_click_rate=0.02,
                treatment_open_lift=0.30,
                treatment_click_lift=1.00
            )
            
            # Analyze the results
            analysis_results = ab_framework.analyze_test_results(test_results)
            
            # Visualize the results
            visualization_paths = ab_framework.visualize_test_results(
                test_results,
                output_dir=os.path.join(self.output_dir, 'ab_tests')
            )
            
            # Export the results and analysis
            results_csv, analysis_json = ab_framework.export_test_results(
                test_results,
                analysis_results,
                output_dir=os.path.join(self.output_dir, 'ab_tests')
            )
            
            print(f"A/B test results exported to {results_csv}")
            print(f"A/B test analysis exported to {analysis_json}")
            
            return test_results, analysis_results
            
        except Exception as e:
            print(f"Error setting up A/B test: {str(e)}")
            return None
    
    def run_full_pipeline(self):
        """Run the complete email marketing optimization pipeline."""
        print("Running full email marketing optimization pipeline...")
        
        # 1. Load and preprocess data
        data = self.load_data()
        
        # 2. Run exploratory analysis
        eda_summary = self.run_exploratory_analysis(data)
        
        # 3. Train the machine learning model
        model_trained = self.train_model(data)
        
        # 4. Generate optimized campaign
        if model_trained:
            campaign = self.generate_campaign(
                user_data=data[['email_id', 'user_country', 'user_past_purchases']].drop_duplicates(),
                campaign_name="Fully Optimized Email Campaign"
            )
        else:
            print("Skipping campaign generation due to model training failure.")
            campaign = None
        
        # 5. Set up A/B test
        ab_test_results = self.setup_ab_test(
            user_data=data[['email_id', 'user_country', 'user_past_purchases']].drop_duplicates(),
            test_name="Optimized vs Standard Email Test"
        )
        
        print("\nEmail Marketing Optimization Pipeline Complete!")
        print(f"All outputs saved to {self.output_dir}")
        
        return {
            "data": data,
            "eda_summary": eda_summary,
            "model_trained": model_trained,
            "campaign": campaign,
            "ab_test_results": ab_test_results
        }

def main():
    """Main function to run the email marketing optimization system."""
    parser = argparse.ArgumentParser(description='Email Marketing Optimization System')
    parser.add_argument('--data-dir', type=str, default='.', help='Directory containing the data files')
    parser.add_argument('--action', type=str, default='full', 
                        choices=['full', 'eda', 'train', 'campaign', 'abtest'],
                        help='Action to perform')
    
    args = parser.parse_args()
    
    # Initialize the system
    system = EmailMarketingOptimizationSystem(data_dir=args.data_dir)
    
    # Run the requested action
    if args.action == 'full':
        system.run_full_pipeline()
    elif args.action == 'eda':
        data = system.load_data()
        system.run_exploratory_analysis(data)
    elif args.action == 'train':
        system.train_model()
    elif args.action == 'campaign':
        system.generate_campaign()
    elif args.action == 'abtest':
        system.setup_ab_test()

if __name__ == "__main__":
    main()
