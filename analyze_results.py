import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

class ChatbotResultsAnalyzer:
    def __init__(self, raw_csv_path: str, rag_csv_path: str):
        """Initialize the analyzer with paths to both CSV files."""
        self.raw_csv_path = raw_csv_path
        self.rag_csv_path = rag_csv_path
        self.raw_df = None
        self.rag_df = None
        self.comparison_df = None
        
    def load_data(self):
        """Load both CSV files and prepare data for analysis."""
        try:
            print("Loading raw chatbot results...")
            self.raw_df = pd.read_csv(self.raw_csv_path)
            print(f"Loaded {len(self.raw_df)} raw chatbot responses")
            
            print("Loading RAG chatbot results...")
            self.rag_df = pd.read_csv(self.rag_csv_path)
            print(f"Loaded {len(self.rag_df)} RAG chatbot responses")
            
            # Ensure both dataframes have the same questions
            if len(self.raw_df) != len(self.rag_df):
                print("Warning: Different number of responses in raw vs RAG results")
            
            # Create comparison dataframe
            self.create_comparison_dataframe()
            
        except FileNotFoundError as e:
            print(f"Error: Could not find file - {e}")
            raise
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def create_comparison_dataframe(self):
        """Create a combined dataframe for easier comparison."""
        # Merge on Question column
        merged = pd.merge(
            self.raw_df, 
            self.rag_df, 
            on='Question', 
            suffixes=('_raw', '_rag'),
            how='inner'
        )
        
        # Calculate differences
        metrics = ['semantic_similarity', 'answer_length', 'flesch_reading_ease', 
                  'flesch_kincaid_grade', 'sentiment_polarity', 'sentiment_subjectivity']
        
        for metric in metrics:
            if f'{metric}_raw' in merged.columns and f'{metric}_rag' in merged.columns:
                merged[f'{metric}_diff'] = merged[f'{metric}_rag'] - merged[f'{metric}_raw']
                merged[f'{metric}_improvement'] = (merged[f'{metric}_diff'] / merged[f'{metric}_raw']) * 100
        
        self.comparison_df = merged
        print(f"Created comparison dataframe with {len(merged)} matched questions")
    
    def generate_summary_statistics(self) -> Dict:
        """Generate summary statistics for both approaches."""
        summary = {
            'raw': {},
            'rag': {},
            'comparison': {}
        }
        
        metrics = ['semantic_similarity', 'answer_length', 'flesch_reading_ease', 
                  'flesch_kincaid_grade', 'sentiment_polarity', 'sentiment_subjectivity']
        
        # Raw statistics
        for metric in metrics:
            if metric in self.raw_df.columns:
                summary['raw'][metric] = {
                    'mean': self.raw_df[metric].mean(),
                    'std': self.raw_df[metric].std(),
                    'median': self.raw_df[metric].median(),
                    'min': self.raw_df[metric].min(),
                    'max': self.raw_df[metric].max()
                }
        
        # RAG statistics
        for metric in metrics:
            if metric in self.rag_df.columns:
                summary['rag'][metric] = {
                    'mean': self.rag_df[metric].mean(),
                    'std': self.rag_df[metric].std(),
                    'median': self.rag_df[metric].median(),
                    'min': self.rag_df[metric].min(),
                    'max': self.rag_df[metric].max()
                }
        
        # Comparison statistics
        for metric in metrics:
            if f'{metric}_diff' in self.comparison_df.columns:
                summary['comparison'][f'{metric}_improvement'] = {
                    'mean': self.comparison_df[f'{metric}_improvement'].mean(),
                    'median': self.comparison_df[f'{metric}_improvement'].median(),
                    'positive_improvements': (self.comparison_df[f'{metric}_diff'] > 0).sum(),
                    'total_comparisons': len(self.comparison_df)
                }
        
        return summary
    
    def perform_statistical_tests(self) -> Dict:
        """Perform statistical tests to compare the two approaches."""
        results = {}
        metrics = ['semantic_similarity', 'answer_length', 'flesch_reading_ease', 
                  'flesch_kincaid_grade', 'sentiment_polarity', 'sentiment_subjectivity']
        
        for metric in metrics:
            if f'{metric}_raw' in self.comparison_df.columns and f'{metric}_rag' in self.comparison_df.columns:
                raw_values = self.comparison_df[f'{metric}_raw'].dropna()
                rag_values = self.comparison_df[f'{metric}_rag'].dropna()
                
                # Paired t-test
                t_stat, p_value = stats.ttest_rel(rag_values, raw_values)
                
                # Wilcoxon signed-rank test (non-parametric alternative)
                w_stat, w_p_value = stats.wilcoxon(rag_values, raw_values, alternative='two-sided')
                
                # Effect size (Cohen's d)
                diff = rag_values - raw_values
                pooled_std = np.sqrt((raw_values.var() + rag_values.var()) / 2)
                cohens_d = diff.mean() / pooled_std if pooled_std != 0 else 0
                
                results[metric] = {
                    't_statistic': t_stat,
                    't_p_value': p_value,
                    'wilcoxon_statistic': w_stat,
                    'wilcoxon_p_value': w_p_value,
                    'cohens_d': cohens_d,
                    'mean_difference': diff.mean(),
                    'significant_at_05': p_value < 0.05
                }
        
        return results
    
    def create_visualizations(self, output_dir: str = "analysis_plots"):
        """Create comprehensive visualizations comparing both approaches."""
        os.makedirs(output_dir, exist_ok=True)
        
        metrics = ['semantic_similarity', 'answer_length', 'flesch_reading_ease', 
                  'flesch_kincaid_grade', 'sentiment_polarity', 'sentiment_subjectivity']
        
        # 1. Box plots comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if f'{metric}_raw' in self.comparison_df.columns and f'{metric}_rag' in self.comparison_df.columns:
                data_to_plot = [
                    self.comparison_df[f'{metric}_raw'].dropna(),
                    self.comparison_df[f'{metric}_rag'].dropna()
                ]
                
                axes[i].boxplot(data_to_plot, labels=['Raw', 'RAG'])
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_comparison_boxplots.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Improvement distribution
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if f'{metric}_diff' in self.comparison_df.columns:
                diff_data = self.comparison_df[f'{metric}_diff'].dropna()
                
                axes[i].hist(diff_data, bins=20, alpha=0.7, edgecolor='black')
                axes[i].axvline(x=0, color='red', linestyle='--', label='No Change')
                axes[i].axvline(x=diff_data.mean(), color='green', linestyle='-', label=f'Mean: {diff_data.mean():.3f}')
                axes[i].set_title(f'{metric.replace("_", " ").title()} - Improvement Distribution')
                axes[i].set_xlabel('RAG - Raw (Positive = RAG Better)')
                axes[i].set_ylabel('Frequency')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'improvement_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Scatter plots showing correlation
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if f'{metric}_raw' in self.comparison_df.columns and f'{metric}_rag' in self.comparison_df.columns:
                x = self.comparison_df[f'{metric}_raw'].dropna()
                y = self.comparison_df[f'{metric}_rag'].dropna()
                
                # Ensure same length
                min_len = min(len(x), len(y))
                x, y = x[:min_len], y[:min_len]
                
                axes[i].scatter(x, y, alpha=0.6)
                
                # Add diagonal line (y=x)
                min_val = min(x.min(), y.min())
                max_val = max(x.max(), y.max())
                axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
                
                # Calculate correlation
                correlation = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0
                
                axes[i].set_xlabel(f'Raw {metric.replace("_", " ").title()}')
                axes[i].set_ylabel(f'RAG {metric.replace("_", " ").title()}')
                axes[i].set_title(f'{metric.replace("_", " ").title()}\nCorrelation: {correlation:.3f}')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'raw_vs_rag_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Answer length comparison
        plt.figure(figsize=(12, 8))
        
        if 'Answer_raw' in self.comparison_df.columns and 'Answer_rag' in self.comparison_df.columns:
            raw_lengths = self.comparison_df['Answer_raw'].str.len().dropna()
            rag_lengths = self.comparison_df['Answer_rag'].str.len().dropna()
            
            plt.subplot(2, 2, 1)
            plt.hist([raw_lengths, rag_lengths], bins=20, alpha=0.7, label=['Raw', 'RAG'], edgecolor='black')
            plt.xlabel('Answer Length (characters)')
            plt.ylabel('Frequency')
            plt.title('Answer Length Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 2)
            plt.boxplot([raw_lengths, rag_lengths], labels=['Raw', 'RAG'])
            plt.ylabel('Answer Length (characters)')
            plt.title('Answer Length Box Plot')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 3)
            plt.scatter(raw_lengths, rag_lengths, alpha=0.6)
            min_len = min(raw_lengths.min(), rag_lengths.min())
            max_len = max(raw_lengths.max(), rag_lengths.max())
            plt.plot([min_len, max_len], [min_len, max_len], 'r--', label='y=x')
            plt.xlabel('Raw Answer Length')
            plt.ylabel('RAG Answer Length')
            plt.title('Answer Length Correlation')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 4)
            length_diff = rag_lengths - raw_lengths
            plt.hist(length_diff, bins=20, alpha=0.7, edgecolor='black')
            plt.axvline(x=0, color='red', linestyle='--', label='No Change')
            plt.axvline(x=length_diff.mean(), color='green', linestyle='-', label=f'Mean: {length_diff.mean():.1f}')
            plt.xlabel('Length Difference (RAG - Raw)')
            plt.ylabel('Frequency')
            plt.title('Answer Length Improvement')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'answer_length_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_dir}/")
    
    def generate_detailed_report(self, output_file: str = "chatbot_comparison_report.txt"):
        """Generate a detailed text report of the analysis."""
        summary_stats = self.generate_summary_statistics()
        statistical_tests = self.perform_statistical_tests()
        
        with open(output_file, 'w') as f:
            f.write("CHATBOT COMPARISON ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Raw Chatbot Results: {self.raw_csv_path}\n")
            f.write(f"RAG Chatbot Results: {self.rag_csv_path}\n")
            f.write(f"Total Questions Analyzed: {len(self.comparison_df)}\n\n")
            
            # Summary Statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 20 + "\n\n")
            
            metrics = ['semantic_similarity', 'answer_length', 'flesch_reading_ease', 
                      'flesch_kincaid_grade', 'sentiment_polarity', 'sentiment_subjectivity']
            
            for metric in metrics:
                if metric in summary_stats['raw'] and metric in summary_stats['rag']:
                    f.write(f"{metric.replace('_', ' ').title()}:\n")
                    f.write(f"  Raw - Mean: {summary_stats['raw'][metric]['mean']:.3f}, "
                           f"Std: {summary_stats['raw'][metric]['std']:.3f}\n")
                    f.write(f"  RAG - Mean: {summary_stats['rag'][metric]['mean']:.3f}, "
                           f"Std: {summary_stats['rag'][metric]['std']:.3f}\n")
                    
                    if f'{metric}_improvement' in summary_stats['comparison']:
                        improvement = summary_stats['comparison'][f'{metric}_improvement']
                        f.write(f"  Average Improvement: {improvement['mean']:.2f}%\n")
                        f.write(f"  Questions with Improvement: {improvement['positive_improvements']}/{improvement['total_comparisons']}\n")
                    f.write("\n")
            
            # Statistical Tests
            f.write("STATISTICAL SIGNIFICANCE TESTS\n")
            f.write("-" * 30 + "\n\n")
            
            for metric, results in statistical_tests.items():
                f.write(f"{metric.replace('_', ' ').title()}:\n")
                f.write(f"  Paired t-test: t={results['t_statistic']:.3f}, p={results['t_p_value']:.6f}\n")
                f.write(f"  Wilcoxon test: W={results['wilcoxon_statistic']:.3f}, p={results['wilcoxon_p_value']:.6f}\n")
                f.write(f"  Effect size (Cohen's d): {results['cohens_d']:.3f}\n")
                f.write(f"  Mean difference: {results['mean_difference']:.3f}\n")
                f.write(f"  Significant at α=0.05: {'Yes' if results['significant_at_05'] else 'No'}\n\n")
            
            # Key Findings
            f.write("KEY FINDINGS\n")
            f.write("-" * 12 + "\n\n")
            
            # Find metrics with significant improvements
            significant_improvements = []
            significant_degradations = []
            
            for metric, results in statistical_tests.items():
                if results['significant_at_05']:
                    if results['mean_difference'] > 0:
                        significant_improvements.append((metric, results['mean_difference']))
                    else:
                        significant_degradations.append((metric, results['mean_difference']))
            
            if significant_improvements:
                f.write("Significant Improvements (RAG > Raw):\n")
                for metric, diff in significant_improvements:
                    f.write(f"  - {metric.replace('_', ' ').title()}: +{diff:.3f}\n")
                f.write("\n")
            
            if significant_degradations:
                f.write("Significant Degradations (RAG < Raw):\n")
                for metric, diff in significant_degradations:
                    f.write(f"  - {metric.replace('_', ' ').title()}: {diff:.3f}\n")
                f.write("\n")
            
            if not significant_improvements and not significant_degradations:
                f.write("No statistically significant differences found between approaches.\n\n")
            
            # Answer quality analysis
            if 'Answer_raw' in self.comparison_df.columns and 'Answer_rag' in self.comparison_df.columns:
                raw_lengths = self.comparison_df['Answer_raw'].str.len()
                rag_lengths = self.comparison_df['Answer_rag'].str.len()
                
                f.write("ANSWER QUALITY ANALYSIS\n")
                f.write("-" * 23 + "\n\n")
                f.write(f"Average Raw Answer Length: {raw_lengths.mean():.1f} characters\n")
                f.write(f"Average RAG Answer Length: {rag_lengths.mean():.1f} characters\n")
                f.write(f"Length Improvement: {((rag_lengths.mean() - raw_lengths.mean()) / raw_lengths.mean() * 100):.1f}%\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 15 + "\n\n")
            
            if significant_improvements:
                f.write("✓ RAG approach shows significant improvements in several metrics.\n")
                f.write("✓ Consider deploying the RAG-enhanced chatbot for production use.\n")
            else:
                f.write("• No significant improvements detected with RAG approach.\n")
                f.write("• Consider refining the RAG implementation or document quality.\n")
            
            if significant_degradations:
                f.write("⚠ Some metrics show degradation with RAG approach.\n")
                f.write("⚠ Investigate and address these issues before deployment.\n")
            
            f.write("\nEnd of Report\n")
        
        print(f"Detailed report saved to {output_file}")
    
    def run_complete_analysis(self, output_dir: str = "analysis_results"):
        """Run the complete analysis pipeline."""
        print("Starting comprehensive chatbot comparison analysis...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        self.load_data()
        
        # Generate visualizations
        plots_dir = os.path.join(output_dir, "plots")
        self.create_visualizations(plots_dir)
        
        # Generate detailed report
        report_file = os.path.join(output_dir, "comparison_report.txt")
        self.generate_detailed_report(report_file)
        
        # Save summary statistics to CSV
        summary_stats = self.generate_summary_statistics()
        statistical_tests = self.perform_statistical_tests()
        
        # Create summary CSV
        summary_data = []
        for metric in ['semantic_similarity', 'answer_length', 'flesch_reading_ease', 
                      'flesch_kincaid_grade', 'sentiment_polarity', 'sentiment_subjectivity']:
            if metric in summary_stats['raw'] and metric in summary_stats['rag']:
                row = {
                    'Metric': metric,
                    'Raw_Mean': summary_stats['raw'][metric]['mean'],
                    'RAG_Mean': summary_stats['rag'][metric]['mean'],
                    'Improvement': summary_stats['rag'][metric]['mean'] - summary_stats['raw'][metric]['mean'],
                    'Improvement_Percent': ((summary_stats['rag'][metric]['mean'] - summary_stats['raw'][metric]['mean']) / summary_stats['raw'][metric]['mean']) * 100,
                    'P_Value': statistical_tests.get(metric, {}).get('t_p_value', 'N/A'),
                    'Significant': statistical_tests.get(metric, {}).get('significant_at_05', False),
                    'Effect_Size': statistical_tests.get(metric, {}).get('cohens_d', 'N/A')
                }
                summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_csv = os.path.join(output_dir, "summary_statistics.csv")
        summary_df.to_csv(summary_csv, index=False)
        
        print(f"\nAnalysis complete! Results saved to {output_dir}/")
        print(f"- Plots: {plots_dir}/")
        print(f"- Report: {report_file}")
        print(f"- Summary: {summary_csv}")
        
        return summary_stats, statistical_tests

def main():
    # File paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_csv = os.path.join(base_dir, "qa_outputs", "questions_answers_raw.csv")
    rag_csv = os.path.join(base_dir, "qa_outputs", "questions_answers_rag.csv")
    
    # Check if files exist
    if not os.path.exists(raw_csv):
        print(f"Error: Raw results file not found at {raw_csv}")
        return
    
    if not os.path.exists(rag_csv):
        print(f"Error: RAG results file not found at {rag_csv}")
        return
    
    # Run analysis
    analyzer = ChatbotResultsAnalyzer(raw_csv, rag_csv)
    summary_stats, statistical_tests = analyzer.run_complete_analysis()
    
    # Print quick summary to console
    print("\n" + "="*60)
    print("QUICK SUMMARY")
    print("="*60)
    
    significant_improvements = 0
    significant_degradations = 0
    
    for metric, results in statistical_tests.items():
        if results['significant_at_05']:
            if results['mean_difference'] > 0:
                significant_improvements += 1
                print(f"✓ {metric.replace('_', ' ').title()}: Significant improvement")
            else:
                significant_degradations += 1
                print(f"✗ {metric.replace('_', ' ').title()}: Significant degradation")
    
    print(f"\nTotal significant improvements: {significant_improvements}")
    print(f"Total significant degradations: {significant_degradations}")
    
    if significant_improvements > significant_degradations:
        print("\n🎉 Overall: RAG approach shows net improvement!")
    elif significant_degradations > significant_improvements:
        print("\n⚠️  Overall: RAG approach shows net degradation.")
    else:
        print("\n➖ Overall: Mixed or no significant differences.")

if __name__ == "__main__":
    main() 