import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import datetime
import json
from pathlib import Path

class ABTestingFramework:
    """
    A framework for designing, implementing, and analyzing A/B tests for email campaigns.
    This framework helps validate the effectiveness of the ML-optimized email campaigns
    compared to traditional approaches.
    """
    
    def __init__(self):
        """Initialize the A/B testing framework."""
        print("Initializing A/B Testing Framework...")
    
    def design_test(self, user_data, test_name, control_size=0.2, random_seed=42):
        """
        Design an A/B test by splitting users into control and treatment groups.
        
        Parameters:
        -----------
        user_data : DataFrame
            User data containing email_id and other user attributes
        test_name : str
            Name of the A/B test
        control_size : float, optional
            Proportion of users to assign to the control group (default: 0.2)
        random_seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        DataFrame with users assigned to control and treatment groups
        """
        # Create a copy of the user data
        test_design = user_data.copy()
        
        # Assign users to control and treatment groups
        np.random.seed(random_seed)
        test_design['group'] = np.random.choice(
            ['control', 'treatment'], 
            size=len(test_design), 
            p=[control_size, 1-control_size]
        )
        
        # Add test metadata
        test_design['test_name'] = test_name
        test_design['test_date'] = datetime.datetime.now().strftime('%Y-%m-%d')
        
        # Print test design summary
        control_count = (test_design['group'] == 'control').sum()
        treatment_count = (test_design['group'] == 'treatment').sum()
        
        print(f"A/B Test Design: {test_name}")
        print(f"Total users: {len(test_design)}")
        print(f"Control group: {control_count} users ({control_count/len(test_design):.1%})")
        print(f"Treatment group: {treatment_count} users ({treatment_count/len(test_design):.1%})")
        
        return test_design
    
    def export_test_design(self, test_design, output_dir='ab_tests'):
        """Export the A/B test design to CSV and JSON formats."""
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Get test name and create sanitized filename
        test_name = test_design['test_name'].iloc[0]
        filename_base = test_name.lower().replace(' ', '_')
        
        # Export to CSV
        csv_path = f"{output_dir}/{filename_base}_design.csv"
        test_design.to_csv(csv_path, index=False)
        
        # Export to JSON
        json_path = f"{output_dir}/{filename_base}_design.json"
        
        # Convert to JSON-friendly format
        test_json = {
            "test_name": test_name,
            "test_date": test_design['test_date'].iloc[0],
            "total_users": len(test_design),
            "control_users": (test_design['group'] == 'control').sum(),
            "treatment_users": (test_design['group'] == 'treatment').sum(),
            "user_assignments": test_design[['email_id', 'group']].to_dict(orient='records')
        }
        
        with open(json_path, 'w') as f:
            json.dump(test_json, f, indent=2)
        
        print(f"Test design exported to:")
        print(f"- CSV: {csv_path}")
        print(f"- JSON: {json_path}")
        
        return csv_path, json_path
    
    def analyze_test_results(self, test_results, confidence_level=0.95):
        """
        Analyze the results of an A/B test.
        
        Parameters:
        -----------
        test_results : DataFrame
            Results of the A/B test with columns:
            - email_id: Unique identifier for each user
            - group: 'control' or 'treatment'
            - opened: Whether the user opened the email (0 or 1)
            - clicked: Whether the user clicked the link (0 or 1)
        confidence_level : float, optional
            Confidence level for statistical tests (default: 0.95)
            
        Returns:
        --------
        Dictionary with test analysis results
        """
        print("Analyzing A/B test results...")
        
        # Calculate metrics for each group
        control_results = test_results[test_results['group'] == 'control']
        treatment_results = test_results[test_results['group'] == 'treatment']
        
        # Calculate open rates
        control_open_rate = control_results['opened'].mean()
        treatment_open_rate = treatment_results['opened'].mean()
        open_rate_lift = (treatment_open_rate - control_open_rate) / control_open_rate
        
        # Calculate click rates
        control_click_rate = control_results['clicked'].mean()
        treatment_click_rate = treatment_results['clicked'].mean()
        click_rate_lift = (treatment_click_rate - control_click_rate) / control_click_rate
        
        # Calculate click-through rates (clicks among opened)
        control_ctr = control_results.loc[control_results['opened'] == 1, 'clicked'].mean() if control_results['opened'].sum() > 0 else 0
        treatment_ctr = treatment_results.loc[treatment_results['opened'] == 1, 'clicked'].mean() if treatment_results['opened'].sum() > 0 else 0
        ctr_lift = (treatment_ctr - control_ctr) / control_ctr if control_ctr > 0 else float('inf')
        
        # Perform statistical tests
        # 1. Open rate
        open_rate_pvalue = stats.proportions_ztest(
            [treatment_results['opened'].sum(), control_results['opened'].sum()],
            [len(treatment_results), len(control_results)]
        )[1]
        
        # 2. Click rate
        click_rate_pvalue = stats.proportions_ztest(
            [treatment_results['clicked'].sum(), control_results['clicked'].sum()],
            [len(treatment_results), len(control_results)]
        )[1]
        
        # 3. Click-through rate
        if control_results['opened'].sum() > 0 and treatment_results['opened'].sum() > 0:
            ctr_pvalue = stats.proportions_ztest(
                [treatment_results.loc[treatment_results['opened'] == 1, 'clicked'].sum(), 
                 control_results.loc[control_results['opened'] == 1, 'clicked'].sum()],
                [treatment_results['opened'].sum(), control_results['opened'].sum()]
            )[1]
        else:
            ctr_pvalue = float('nan')
        
        # Determine statistical significance
        alpha = 1 - confidence_level
        open_rate_significant = open_rate_pvalue < alpha
        click_rate_significant = click_rate_pvalue < alpha
        ctr_significant = ctr_pvalue < alpha if not np.isnan(ctr_pvalue) else False
        
        # Compile results
        results = {
            "metrics": {
                "open_rate": {
                    "control": control_open_rate,
                    "treatment": treatment_open_rate,
                    "absolute_difference": treatment_open_rate - control_open_rate,
                    "relative_lift": open_rate_lift,
                    "p_value": open_rate_pvalue,
                    "significant": open_rate_significant
                },
                "click_rate": {
                    "control": control_click_rate,
                    "treatment": treatment_click_rate,
                    "absolute_difference": treatment_click_rate - control_click_rate,
                    "relative_lift": click_rate_lift,
                    "p_value": click_rate_pvalue,
                    "significant": click_rate_significant
                },
                "click_through_rate": {
                    "control": control_ctr,
                    "treatment": treatment_ctr,
                    "absolute_difference": treatment_ctr - control_ctr,
                    "relative_lift": ctr_lift,
                    "p_value": float(ctr_pvalue) if not np.isnan(ctr_pvalue) else None,
                    "significant": ctr_significant
                }
            },
            "sample_sizes": {
                "control": len(control_results),
                "treatment": len(treatment_results),
                "control_opened": control_results['opened'].sum(),
                "treatment_opened": treatment_results['opened'].sum(),
                "control_clicked": control_results['clicked'].sum(),
                "treatment_clicked": treatment_results['clicked'].sum()
            },
            "confidence_level": confidence_level
        }
        
        # Print summary
        print("\nA/B Test Results Summary:")
        print(f"Sample sizes: Control={len(control_results)}, Treatment={len(treatment_results)}")
        
        print("\nOpen Rate:")
        print(f"Control: {control_open_rate:.2%}")
        print(f"Treatment: {treatment_open_rate:.2%}")
        print(f"Lift: {open_rate_lift:.2%}")
        print(f"P-value: {open_rate_pvalue:.4f} ({'Significant' if open_rate_significant else 'Not significant'})")
        
        print("\nClick Rate:")
        print(f"Control: {control_click_rate:.2%}")
        print(f"Treatment: {treatment_click_rate:.2%}")
        print(f"Lift: {click_rate_lift:.2%}")
        print(f"P-value: {click_rate_pvalue:.4f} ({'Significant' if click_rate_significant else 'Not significant'})")
        
        print("\nClick-Through Rate (among opened):")
        print(f"Control: {control_ctr:.2%}")
        print(f"Treatment: {treatment_ctr:.2%}")
        print(f"Lift: {ctr_lift:.2%}")
        if not np.isnan(ctr_pvalue):
            print(f"P-value: {ctr_pvalue:.4f} ({'Significant' if ctr_significant else 'Not significant'})")
        else:
            print("P-value: Not available (insufficient data)")
        
        return results
    
    def visualize_test_results(self, test_results, output_dir='ab_tests'):
        """
        Create visualizations of A/B test results.
        
        Parameters:
        -----------
        test_results : DataFrame
            Results of the A/B test
        output_dir : str, optional
            Directory to save visualizations
            
        Returns:
        --------
        List of paths to saved visualizations
        """
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Set plot style
        plt.style.use('ggplot')
        sns.set(style="whitegrid")
        
        # 1. Bar chart of open and click rates
        plt.figure(figsize=(12, 6))
        
        # Calculate metrics
        metrics = test_results.groupby('group').agg({
            'opened': 'mean',
            'clicked': 'mean'
        })
        
        # Reshape for plotting
        metrics_plot = metrics.reset_index().melt(
            id_vars='group',
            value_vars=['opened', 'clicked'],
            var_name='metric',
            value_name='rate'
        )
        
        # Create plot
        ax = sns.barplot(x='metric', y='rate', hue='group', data=metrics_plot)
        ax.set_title('Email Campaign Performance by Group', fontsize=16)
        ax.set_xlabel('Metric', fontsize=12)
        ax.set_ylabel('Rate', fontsize=12)
        ax.set_xticklabels(['Open Rate', 'Click Rate'])
        ax.set_ylim(0, max(metrics_plot['rate']) * 1.2)
        
        # Add value labels
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2%}", 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'bottom', fontsize=10)
        
        # Save plot
        bar_chart_path = f"{output_dir}/ab_test_rates.png"
        plt.tight_layout()
        plt.savefig(bar_chart_path)
        
        # 2. Funnel visualization
        plt.figure(figsize=(10, 6))
        
        # Calculate funnel metrics
        funnel_data = []
        for group in ['control', 'treatment']:
            group_data = test_results[test_results['group'] == group]
            total = len(group_data)
            opened = group_data['opened'].sum()
            clicked = group_data['clicked'].sum()
            
            funnel_data.append({
                'group': group,
                'stage': 'Received',
                'count': total,
                'rate': 1.0
            })
            funnel_data.append({
                'group': group,
                'stage': 'Opened',
                'count': opened,
                'rate': opened / total if total > 0 else 0
            })
            funnel_data.append({
                'group': group,
                'stage': 'Clicked',
                'count': clicked,
                'rate': clicked / total if total > 0 else 0
            })
        
        funnel_df = pd.DataFrame(funnel_data)
        
        # Create plot
        ax = sns.barplot(x='stage', y='rate', hue='group', data=funnel_df)
        ax.set_title('Email Campaign Funnel', fontsize=16)
        ax.set_xlabel('Stage', fontsize=12)
        ax.set_ylabel('Conversion Rate', fontsize=12)
        ax.set_ylim(0, 1.1)
        
        # Add value labels
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2%}", 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'bottom', fontsize=10)
        
        # Save plot
        funnel_chart_path = f"{output_dir}/ab_test_funnel.png"
        plt.tight_layout()
        plt.savefig(funnel_chart_path)
        
        # 3. Segment analysis (if user segments are available)
        if 'user_segment' in test_results.columns:
            plt.figure(figsize=(14, 8))
            
            # Calculate click rates by segment and group
            segment_metrics = test_results.groupby(['user_segment', 'group'])['clicked'].mean().reset_index()
            
            # Create plot
            ax = sns.barplot(x='user_segment', y='clicked', hue='group', data=segment_metrics)
            ax.set_title('Click Rate by User Segment', fontsize=16)
            ax.set_xlabel('User Segment', fontsize=12)
            ax.set_ylabel('Click Rate', fontsize=12)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Add value labels
            for p in ax.patches:
                ax.annotate(f"{p.get_height():.2%}", 
                           (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha = 'center', va = 'bottom', fontsize=9)
            
            # Save plot
            segment_chart_path = f"{output_dir}/ab_test_segments.png"
            plt.tight_layout()
            plt.savefig(segment_chart_path)
            
            return [bar_chart_path, funnel_chart_path, segment_chart_path]
        
        return [bar_chart_path, funnel_chart_path]
    
    def export_test_results(self, test_results, analysis_results, output_dir='ab_tests'):
        """Export the A/B test results and analysis to CSV and JSON formats."""
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Get test name if available, otherwise use a timestamp
        if 'test_name' in test_results.columns:
            test_name = test_results['test_name'].iloc[0]
        else:
            test_name = f"ab_test_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        filename_base = test_name.lower().replace(' ', '_')
        
        # Export results to CSV
        csv_path = f"{output_dir}/{filename_base}_results.csv"
        test_results.to_csv(csv_path, index=False)
        
        # Export analysis to JSON
        json_path = f"{output_dir}/{filename_base}_analysis.json"
        
        # Add metadata to analysis results
        analysis_export = analysis_results.copy()
        analysis_export["test_name"] = test_name
        analysis_export["analysis_date"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(json_path, 'w') as f:
            json.dump(analysis_export, f, indent=2)
        
        print(f"Test results and analysis exported to:")
        print(f"- Results CSV: {csv_path}")
        print(f"- Analysis JSON: {json_path}")
        
        return csv_path, json_path
    
    def simulate_test_results(self, test_design, control_open_rate=0.1, control_click_rate=0.02,
                             treatment_open_lift=0.3, treatment_click_lift=0.5, random_seed=42):
        """
        Simulate A/B test results for demonstration purposes.
        
        Parameters:
        -----------
        test_design : DataFrame
            A/B test design with user assignments
        control_open_rate : float, optional
            Open rate for the control group (default: 0.1)
        control_click_rate : float, optional
            Click rate for the control group (default: 0.02)
        treatment_open_lift : float, optional
            Relative lift in open rate for treatment group (default: 0.3)
        treatment_click_lift : float, optional
            Relative lift in click rate for treatment group (default: 0.5)
        random_seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        DataFrame with simulated test results
        """
        np.random.seed(random_seed)
        
        # Create a copy of the test design
        results = test_design.copy()
        
        # Calculate treatment rates
        treatment_open_rate = control_open_rate * (1 + treatment_open_lift)
        treatment_click_rate = control_click_rate * (1 + treatment_click_lift)
        
        # Simulate open and click events
        results['opened'] = 0
        results['clicked'] = 0
        
        # Control group
        control_mask = results['group'] == 'control'
        control_users = results[control_mask].index
        
        # Simulate opens for control
        open_probs = np.random.random(len(control_users))
        results.loc[control_users[open_probs < control_open_rate], 'opened'] = 1
        
        # Simulate clicks for control (only for opened emails)
        opened_control = results.loc[control_users & (results['opened'] == 1)].index
        click_probs = np.random.random(len(opened_control))
        click_rate_among_opened = control_click_rate / control_open_rate
        results.loc[opened_control[click_probs < click_rate_among_opened], 'clicked'] = 1
        
        # Treatment group
        treatment_mask = results['group'] == 'treatment'
        treatment_users = results[treatment_mask].index
        
        # Simulate opens for treatment
        open_probs = np.random.random(len(treatment_users))
        results.loc[treatment_users[open_probs < treatment_open_rate], 'opened'] = 1
        
        # Simulate clicks for treatment (only for opened emails)
        opened_treatment = results.loc[treatment_users & (results['opened'] == 1)].index
        click_probs = np.random.random(len(opened_treatment))
        click_rate_among_opened = treatment_click_rate / treatment_open_rate
        results.loc[opened_treatment[click_probs < click_rate_among_opened], 'clicked'] = 1
        
        print(f"Simulated test results:")
        print(f"Control: {control_open_rate:.2%} open rate, {control_click_rate:.2%} click rate")
        print(f"Treatment: {treatment_open_rate:.2%} open rate, {treatment_click_rate:.2%} click rate")
        print(f"Expected lift: {treatment_open_lift:.2%} in opens, {treatment_click_lift:.2%} in clicks")
        
        return results

# Example usage function
def run_example():
    """Run an example A/B test simulation and analysis."""
    print("Running A/B testing example...")
    
    # Load the original data for demonstration
    try:
        email_data = pd.read_csv('email_table.csv')
        # Take a sample of users for the example
        sample_users = email_data[['email_id', 'user_country', 'user_past_purchases']].sample(5000, random_state=42)
    except:
        # Create dummy data if file doesn't exist
        print("Creating dummy data for demonstration...")
        sample_users = pd.DataFrame({
            'email_id': [f'user_{i}' for i in range(5000)],
            'user_country': np.random.choice(['US', 'UK', 'FR', 'ES'], size=5000),
            'user_past_purchases': np.random.randint(0, 20, size=5000)
        })
    
    # Add user segments for demonstration
    sample_users['user_segment'] = pd.cut(
        sample_users['user_past_purchases'],
        bins=[-1, 0, 3, 7, 100],
        labels=['No purchases', '1-3 purchases', '4-7 purchases', '8+ purchases']
    )
    
    # Initialize the A/B testing framework
    ab_framework = ABTestingFramework()
    
    # Design the test
    test_design = ab_framework.design_test(
        sample_users,
        test_name="Optimized Email Campaign Test",
        control_size=0.2
    )
    
    # Export the test design
    ab_framework.export_test_design(test_design)
    
    # Simulate test results
    # Assuming our optimization improves open rate by 30% and click rate by 100%
    test_results = ab_framework.simulate_test_results(
        test_design,
        control_open_rate=0.10,  # 10% open rate for control
        control_click_rate=0.02,  # 2% click rate for control
        treatment_open_lift=0.30,  # 30% lift in open rate
        treatment_click_lift=1.00   # 100% lift in click rate
    )
    
    # Analyze the results
    analysis_results = ab_framework.analyze_test_results(test_results)
    
    # Visualize the results
    visualization_paths = ab_framework.visualize_test_results(test_results)
    
    # Export the results and analysis
    ab_framework.export_test_results(test_results, analysis_results)
    
    return test_results, analysis_results

if __name__ == "__main__":
    run_example()
