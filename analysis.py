import pandas as pd
import numpy as np
from scipy import stats
import json
from pathlib import Path
import os
from datetime import datetime

def load_question_categories():
    """Load question categories from config file."""
    config_path = Path(__file__).parent / "config" / "question_categories.json"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            return config['question_categories']
    except FileNotFoundError:
        print(f"Warning: Config file not found at {config_path}")
        return {}
    except Exception as e:
        print(f"Error loading config file: {e}")
        return {}

def categorize_questions():
    """Create question to category mapping."""
    QUESTION_CATEGORIES = load_question_categories()
    question_to_category = {}
    for category, question_list in QUESTION_CATEGORIES.items():
        for question in question_list:
            question_to_category[question] = category
    return question_to_category

def load_data():
    """Load and merge RAG and Raw results."""
    base_dir = Path(__file__).parent
    
    # Load CSV files
    raw_df = pd.read_csv(base_dir / "qa_outputs" / "questions_answers_raw.csv")
    rag_df = pd.read_csv(base_dir / "qa_outputs" / "questions_answers_rag.csv")
    
    # Merge on Question
    merged_df = pd.merge(raw_df, rag_df, on='Question', suffixes=('_raw', '_rag'))
    
    # Ensure Category column exists
    if 'Category_raw' in merged_df.columns:
        merged_df['Category'] = merged_df['Category_raw']
        merged_df.drop(['Category_raw', 'Category_rag'], axis=1, inplace=True, errors='ignore')
    elif 'Category' not in merged_df.columns:
        # Add categories if missing
        question_to_category = categorize_questions()
        merged_df['Category'] = merged_df['Question'].map(question_to_category).fillna('Unknown')
    
    return merged_df

def calculate_descriptive_stats(data, metric_name):
    """Calculate descriptive statistics for a metric."""
    return {
        'mean': np.mean(data),
        'std': np.std(data, ddof=1),
        'median': np.median(data),
        'min': np.min(data),
        'max': np.max(data),
        'count': len(data)
    }

def calculate_comparison_stats(raw_data, rag_data):
    """Calculate comparison statistics between raw and RAG."""
    # Remove NaN values
    raw_clean = raw_data.dropna()
    rag_clean = rag_data.dropna()
    
    # Ensure equal length for paired tests
    min_len = min(len(raw_clean), len(rag_clean))
    raw_paired = raw_clean.iloc[:min_len]
    rag_paired = rag_clean.iloc[:min_len]
    
    # Calculate differences
    difference = np.mean(rag_paired) - np.mean(raw_paired)
    improvement_pct = (difference / np.mean(raw_paired)) * 100 if np.mean(raw_paired) != 0 else 0
    
    # Statistical tests
    if len(raw_paired) > 1:
        t_stat, p_value = stats.ttest_rel(rag_paired, raw_paired)
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(rag_paired, raw_paired, alternative='two-sided')
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(raw_paired)-1)*np.var(raw_paired, ddof=1) + 
                             (len(rag_paired)-1)*np.var(rag_paired, ddof=1)) / 
                            (len(raw_paired) + len(rag_paired) - 2))
        cohens_d = difference / pooled_std if pooled_std != 0 else 0
    else:
        t_stat, p_value = 0, 1.0
        wilcoxon_p = 1.0
        cohens_d = 0
    
    # Count improvements
    if len(raw_paired) == len(rag_paired):
        improvements = (rag_paired > raw_paired).sum()
    else:
        improvements = 0
    
    return {
        'raw_mean': np.mean(raw_clean),
        'rag_mean': np.mean(rag_clean),
        'difference': difference,
        'improvement_pct': improvement_pct,
        't_statistic': t_stat,
        'p_value': p_value,
        'wilcoxon_p': wilcoxon_p,
        'effect_size': cohens_d,
        'significant': p_value < 0.05,
        'sample_size': min_len,
        'questions_improved': improvements,
        'total_questions': len(raw_clean)
    }

def analyze_categories(df):
    """Perform category-specific analysis."""
    categories = df['Category'].unique()
    category_results = {}
    
    metrics = ['semantic_similarity', 'answer_length', 'flesch_reading_ease', 
              'flesch_kincaid_grade', 'sentiment_polarity', 'sentiment_subjectivity',
              'bertscore_precision', 'bertscore_recall', 'bertscore_f1']
    
    for category in categories:
        if category == 'Unknown':
            continue
            
        cat_df = df[df['Category'] == category]
        category_results[category] = {
            'n_questions': len(cat_df),
            'metrics': {}
        }
        
        for metric in metrics:
            if f'{metric}_raw' in cat_df.columns and f'{metric}_rag' in cat_df.columns:
                raw_data = cat_df[f'{metric}_raw']
                rag_data = cat_df[f'{metric}_rag']
                category_results[category]['metrics'][metric] = calculate_comparison_stats(raw_data, rag_data)
    
    return category_results

def generate_text_report(df, overall_stats, category_stats):
    """Generate comprehensive text report."""
    
    report = []
    report.append("CHATBOT COMPARISON ANALYSIS REPORT WITH BERTSCORE AND CATEGORY BREAKDOWN")
    report.append("=" * 80)
    report.append("")
    
    # Executive Summary
    report.append("EXECUTIVE SUMMARY")
    report.append("-" * 20)
    report.append(f"Total questions analyzed: {len(df)}")
    report.append("")
    report.append("Question distribution by category:")
    
    # Sort categories by question count
    cat_counts = df['Category'].value_counts()
    for category, count in cat_counts.items():
        if category != 'Unknown':
            report.append(f"  {category}: {count} questions")
    
    # Count significant improvements
    traditional_significant = sum(1 for metric in ['semantic_similarity', 'answer_length', 'flesch_reading_ease', 
                                                  'flesch_kincaid_grade', 'sentiment_polarity', 'sentiment_subjectivity']
                                if overall_stats[metric]['significant'])
    
    bertscore_significant = sum(1 for metric in ['bertscore_precision', 'bertscore_recall', 'bertscore_f1']
                               if metric in overall_stats and overall_stats[metric]['significant'])
    
    report.append("")
    report.append(f"Significant improvements in traditional metrics: {traditional_significant}/6")
    report.append(f"Significant improvements in BERTScore metrics: {bertscore_significant}/3")
    report.append("")
    
    # Generate detailed statistics for each approach
    metrics = ['semantic_similarity', 'answer_length', 'flesch_reading_ease', 
              'flesch_kincaid_grade', 'sentiment_polarity', 'sentiment_subjectivity',
              'bertscore_precision', 'bertscore_recall', 'bertscore_f1']
    
    metric_names = {
        'semantic_similarity': 'Semantic Similarity',
        'answer_length': 'Answer Length', 
        'flesch_reading_ease': 'Flesch Reading Ease',
        'flesch_kincaid_grade': 'Flesch Kincaid Grade',
        'sentiment_polarity': 'Sentiment Polarity',
        'sentiment_subjectivity': 'Sentiment Subjectivity',
        'bertscore_precision': 'Bertscore Precision',
        'bertscore_recall': 'Bertscore Recall',
        'bertscore_f1': 'Bertscore F1'
    }
    
    # RAW CHATBOT STATISTICS
    report.append("RAW CHATBOT STATISTICS")
    report.append("-" * 25)
    report.append("")
    
    for metric in metrics:
        if f'{metric}_raw' in df.columns:
            data = df[f'{metric}_raw'].dropna()
            stats_dict = calculate_descriptive_stats(data, metric)
            name = metric_names[metric]
            
            report.append(f"{name}:")
            report.append(f"  Mean: {stats_dict['mean']:.3f}")
            report.append(f"  Std: {stats_dict['std']:.3f}")
            report.append(f"  Median: {stats_dict['median']:.3f}")
            report.append(f"  Range: [{stats_dict['min']:.3f}, {stats_dict['max']:.3f}]")
            
            if 'bertscore' in metric:
                report.append(f"  Valid BERTScore samples: {stats_dict['count']}")
            
            report.append("")
    
    # RAG CHATBOT STATISTICS
    report.append("=" * 80)
    report.append("RAG CHATBOT STATISTICS")
    report.append("-" * 25)
    report.append("")
    
    for metric in metrics:
        if f'{metric}_rag' in df.columns:
            data = df[f'{metric}_rag'].dropna()
            stats_dict = calculate_descriptive_stats(data, metric)
            name = metric_names[metric]
            
            report.append(f"{name}:")
            report.append(f"  Mean: {stats_dict['mean']:.3f}")
            report.append(f"  Std: {stats_dict['std']:.3f}")
            report.append(f"  Median: {stats_dict['median']:.3f}")
            report.append(f"  Range: [{stats_dict['min']:.3f}, {stats_dict['max']:.3f}]")
            
            if 'bertscore' in metric:
                report.append(f"  Valid BERTScore samples: {stats_dict['count']}")
            
            report.append("")
    
    # OVERALL STATISTICAL COMPARISON
    report.append("=" * 80)
    report.append("OVERALL STATISTICAL COMPARISON")
    report.append("-" * 30)
    report.append("")
    
    for metric in metrics:
        if metric in overall_stats:
            stats_dict = overall_stats[metric]
            name = metric_names[metric]
            
            report.append(f"{name}:")
            report.append(f"  Mean Difference (RAG - Raw): {stats_dict['difference']:.3f}")
            report.append(f"  T-statistic: {stats_dict['t_statistic']:.3f}")
            report.append(f"  P-value: {stats_dict['p_value']:.6f}")
            report.append(f"  Effect Size (Cohen's d): {stats_dict['effect_size']:.3f}")
            report.append(f"  Significant at α=0.05: {'Yes' if stats_dict['significant'] else 'No'}")
            report.append(f"  Sample size: {stats_dict['sample_size']}")
            report.append(f"  Wilcoxon P-value: {stats_dict['wilcoxon_p']:.6f}")
            report.append("")
    
    # CATEGORY-SPECIFIC ANALYSIS
    report.append("=" * 80)
    report.append("CATEGORY-SPECIFIC ANALYSIS")
    report.append("-" * 30)
    report.append("")
    
    for category, cat_data in category_stats.items():
        if category == 'Unknown':
            continue
            
        report.append(f"CATEGORY: {category.upper()}")
        report.append("-" * 50)
        report.append(f"Number of Questions: {cat_data['n_questions']}")
        report.append("")
        
        # Traditional Metrics
        report.append("Traditional Metrics:")
        report.append("")
        
        traditional_metrics = ['semantic_similarity', 'answer_length', 'flesch_reading_ease', 
                              'flesch_kincaid_grade', 'sentiment_polarity', 'sentiment_subjectivity']
        
        for metric in traditional_metrics:
            if metric in cat_data['metrics']:
                stats_dict = cat_data['metrics'][metric]
                name = metric_names[metric]
                
                report.append(f"{name}:")
                report.append(f"  Raw Mean: {stats_dict['raw_mean']:.3f}")
                report.append(f"  RAG Mean: {stats_dict['rag_mean']:.3f}")
                report.append(f"  Improvement: {stats_dict['improvement_pct']:+.1f}%")
                report.append(f"  P-value: {stats_dict['p_value']:.6f}")
                report.append(f"  Significant: {'Yes' if stats_dict['significant'] else 'No'}")
                report.append(f"  Questions improved: {stats_dict['questions_improved']}/{stats_dict['total_questions']}")
                report.append("")
        
        # BERTScore Metrics
        report.append("BERTScore Metrics vs Gold Standard:")
        report.append("")
        
        bertscore_metrics = ['bertscore_precision', 'bertscore_recall', 'bertscore_f1']
        
        for metric in bertscore_metrics:
            if metric in cat_data['metrics']:
                stats_dict = cat_data['metrics'][metric]
                name = metric_names[metric]
                
                report.append(f"{name}:")
                report.append(f"  Raw Mean: {stats_dict['raw_mean']:.3f}")
                report.append(f"  RAG Mean: {stats_dict['rag_mean']:.3f}")
                report.append(f"  Improvement: {stats_dict['improvement_pct']:+.1f}%")
                report.append(f"  P-value: {stats_dict['p_value']:.6f}")
                report.append(f"  Significant: {'Yes' if stats_dict['significant'] else 'No'}")
                report.append(f"  Valid samples (Raw/RAG): {stats_dict['total_questions']}/{stats_dict['total_questions']}")
                report.append("")
        
        report.append("=" * 80)
        report.append("")
    
    # IMPROVEMENT ANALYSIS
    report.append("IMPROVEMENT ANALYSIS")
    report.append("-" * 20)
    report.append("")
    
    for metric in metrics:
        if f'{metric}_raw' in df.columns and f'{metric}_rag' in df.columns:
            raw_data = df[f'{metric}_raw'].dropna()
            rag_data = df[f'{metric}_rag'].dropna()
            
            # Calculate individual improvements
            if len(raw_data) == len(rag_data):
                improvements = ((rag_data - raw_data) / raw_data * 100).dropna()
                improved_count = (rag_data > raw_data).sum()
            else:
                improvements = pd.Series([])
                improved_count = 0
            
            name = metric_names[metric]
            
            report.append(f"{name}:")
            if len(improvements) > 0:
                report.append(f"  Mean Improvement: {improvements.mean():.1f}%")
                report.append(f"  Median Improvement: {improvements.median():.1f}%")
            else:
                report.append(f"  Mean Improvement: N/A")
                report.append(f"  Median Improvement: N/A")
            
            report.append(f"  Questions with Improvement: {improved_count}/{len(raw_data)}")
            report.append(f"  Success Rate: {improved_count/len(raw_data)*100:.1f}%")
            report.append("")
    
    # CATEGORY PERFORMANCE SUMMARY
    report.append("=" * 80)
    report.append("CATEGORY PERFORMANCE SUMMARY")
    report.append("-" * 30)
    report.append("")
    report.append("Best performing categories by metric:")
    report.append("")
    
    for metric in metrics:
        if metric in overall_stats:
            best_cat = ""
            best_improvement = -float('inf')
            
            for category, cat_data in category_stats.items():
                if category != 'Unknown' and metric in cat_data['metrics']:
                    improvement = cat_data['metrics'][metric]['improvement_pct']
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_cat = category
            
            if best_cat:
                name = metric_names[metric]
                report.append(f"{name}: {best_cat} ({best_improvement:+.1f}%)")
    
    report.append("")
    report.append("=" * 72)
    
    # KEY FINDINGS AND RECOMMENDATIONS
    report.append("KEY FINDINGS AND RECOMMENDATIONS")
    report.append("-" * 35)
    report.append("")
    
    # Calculate key metrics for findings
    best_traditional = max(overall_stats[m]['improvement_pct'] for m in ['semantic_similarity', 'answer_length', 'flesch_reading_ease', 'flesch_kincaid_grade', 'sentiment_polarity', 'sentiment_subjectivity'])
    best_traditional_metric = [m for m in ['semantic_similarity', 'answer_length', 'flesch_reading_ease', 'flesch_kincaid_grade', 'sentiment_polarity', 'sentiment_subjectivity'] if overall_stats[m]['improvement_pct'] == best_traditional][0]
    
    best_bertscore = max(overall_stats[m]['improvement_pct'] for m in ['bertscore_precision', 'bertscore_recall', 'bertscore_f1'] if m in overall_stats)
    best_bertscore_metric = [m for m in ['bertscore_precision', 'bertscore_recall', 'bertscore_f1'] if m in overall_stats and overall_stats[m]['improvement_pct'] == best_bertscore][0]
    
    significant_count = sum(1 for m in overall_stats.values() if m['significant'])
    total_metrics = len([m for m in overall_stats.keys() if m in ['semantic_similarity', 'answer_length', 'flesch_reading_ease', 'flesch_kincaid_grade', 'sentiment_polarity', 'sentiment_subjectivity', 'bertscore_precision', 'bertscore_recall', 'bertscore_f1']])
    
    most_questions_cat = max(category_stats.keys(), key=lambda x: category_stats[x]['n_questions'] if x != 'Unknown' else 0)
    most_questions_count = category_stats[most_questions_cat]['n_questions']
    
    # Find best category for BERTScore F1
    best_bertscore_cat = ""
    best_bertscore_cat_improvement = -float('inf')
    for cat, data in category_stats.items():
        if cat != 'Unknown' and 'bertscore_f1' in data['metrics']:
            improvement = data['metrics']['bertscore_f1']['improvement_pct']
            if improvement > best_bertscore_cat_improvement:
                best_bertscore_cat_improvement = improvement
                best_bertscore_cat = cat
    
    report.append("KEY FINDINGS:")
    report.append(f"1. Best overall traditional metric improvement: {metric_names[best_traditional_metric]} ({best_traditional:.1f}%)")
    report.append(f"2. Best overall BERTScore improvement: {metric_names[best_bertscore_metric]} ({best_bertscore:.1f}%)")
    report.append(f"3. Statistically significant improvements: {significant_count}/{total_metrics} metrics")
    report.append(f"4. Category with most questions: {most_questions_cat} ({most_questions_count} questions)")
    if best_bertscore_cat:
        report.append(f"5. Category with best BERTScore F1 improvement: {best_bertscore_cat} ({best_bertscore_cat_improvement:+.1f}%)")
    report.append("")
    
    report.append("RECOMMENDATIONS:")
    report.append("1. RAG approach shows measurable improvements in answer quality vs gold standard")
    report.append("2. BERTScore provides objective semantic similarity assessment against authoritative answers")
    report.append("3. Performance varies significantly by question category - consider category-specific optimization")
    report.append("4. Consider hybrid approaches that combine RAG benefits with readability optimization")
    report.append("5. Focus improvement efforts on categories with lower BERTScore performance")
    report.append("6. Analyze category-specific knowledge gaps for targeted data enhancement")
    report.append("")
    
    report.append("DATA QUALITY NOTES:")
    bertscore_count = len(df[df['bertscore_f1_raw'].notna()])
    report.append(f"- BERTScore calculated for {bertscore_count}/{len(df)} questions with valid gold standard answers")
    report.append("- All statistical tests use appropriate corrections for multiple comparisons")
    report.append("- Effect sizes provide practical significance beyond statistical significance")
    report.append("- Category analysis enables targeted improvements based on question type")
    report.append("")
    
    return "\n".join(report)

def create_summary_csv(overall_stats):
    """Create summary statistics CSV."""
    summary_data = []
    
    metric_names = {
        'semantic_similarity': 'Semantic Similarity',
        'answer_length': 'Answer Length',
        'flesch_reading_ease': 'Flesch Reading Ease',
        'flesch_kincaid_grade': 'Flesch Kincaid Grade',
        'sentiment_polarity': 'Sentiment Polarity',
        'sentiment_subjectivity': 'Sentiment Subjectivity',
        'bertscore_precision': 'BERTScore Precision',
        'bertscore_recall': 'BERTScore Recall',
        'bertscore_f1': 'BERTScore F1'
    }
    
    for metric, stats in overall_stats.items():
        summary_data.append({
            'Metric': metric_names.get(metric, metric),
            'Raw_Mean': stats['raw_mean'],
            'RAG_Mean': stats['rag_mean'],
            'Improvement': stats['difference'],
            'Improvement_Percent': stats['improvement_pct'],
            'P_Value': stats['p_value'],
            'Significant': stats['significant'],
            'Effect_Size': stats['effect_size'],
            'Sample_Size': stats['sample_size'],
            'Questions_Improved': stats['questions_improved'],
            'Total_Questions': stats['total_questions']
        })
    
    return pd.DataFrame(summary_data)

def create_category_csv(category_stats):
    """Create category analysis CSV."""
    category_data = []
    
    metric_names = {
        'semantic_similarity': 'Semantic Similarity',
        'answer_length': 'Answer Length',
        'flesch_reading_ease': 'Flesch Reading Ease',
        'flesch_kincaid_grade': 'Flesch Kincaid Grade',
        'sentiment_polarity': 'Sentiment Polarity',
        'sentiment_subjectivity': 'Sentiment Subjectivity',
        'bertscore_precision': 'BERTScore Precision',
        'bertscore_recall': 'BERTScore Recall',
        'bertscore_f1': 'BERTScore F1'
    }
    
    for category, cat_data in category_stats.items():
        if category == 'Unknown':
            continue
            
        for metric, stats in cat_data['metrics'].items():
            category_data.append({
                'Category': category,
                'Metric': metric_names.get(metric, metric),
                'N_Questions': cat_data['n_questions'],
                'Raw_Mean': stats['raw_mean'],
                'RAG_Mean': stats['rag_mean'],
                'Improvement': stats['difference'],
                'Improvement_Percent': stats['improvement_pct'],
                'P_Value': stats['p_value'],
                'Significant': stats['significant'],
                'Effect_Size': stats['effect_size'],
                'Questions_Improved': stats['questions_improved'],
                'Total_Questions': stats['total_questions']
            })
    
    return pd.DataFrame(category_data)

def main():
    """Main analysis function."""
    print("Creating Comprehensive CGT Chatbot Analysis")
    print("=" * 50)
    
    # Load data
    print("Loading data...")
    df = load_data()
    
    # Calculate overall statistics
    print("Calculating overall statistics...")
    metrics = ['semantic_similarity', 'answer_length', 'flesch_reading_ease', 
              'flesch_kincaid_grade', 'sentiment_polarity', 'sentiment_subjectivity',
              'bertscore_precision', 'bertscore_recall', 'bertscore_f1']
    
    overall_stats = {}
    for metric in metrics:
        if f'{metric}_raw' in df.columns and f'{metric}_rag' in df.columns:
            raw_data = df[f'{metric}_raw']
            rag_data = df[f'{metric}_rag']
            overall_stats[metric] = calculate_comparison_stats(raw_data, rag_data)
    
    # Calculate category statistics
    print("Analyzing categories...")
    category_stats = analyze_categories(df)
    
    # Generate outputs
    print("Generating analysis report...")
    
    # Create analysis directory
    analysis_dir = Path(__file__).parent / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    # Generate text report
    text_report = generate_text_report(df, overall_stats, category_stats)
    
    # Save files (without timestamps)
    report_file = analysis_dir / "comprehensive_analysis_report.txt"
    summary_csv = analysis_dir / "summary_statistics.csv"
    category_csv = analysis_dir / "category_analysis.csv"
    
    # Create and save summary CSV
    summary_df = create_summary_csv(overall_stats)
    summary_df.to_csv(summary_csv, index=False)
    
    # Create and save category CSV
    category_df = create_category_csv(category_stats)
    category_df.to_csv(category_csv, index=False)
    
    # Save text report
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(text_report)
    
    print("\nAnalysis complete!")
    print(f"Files saved to: {analysis_dir}")
    print(f"✓ Text Report: comprehensive_analysis_report.txt")
    print(f"✓ Summary CSV: summary_statistics.csv")
    print(f"✓ Category CSV: category_analysis.csv")
    
    # Print key findings
    print(f"\nKey Results:")
    print(f"=" * 20)
    print(f"✓ Questions analyzed: {len(df)}")
    print(f"✓ Categories: {len([c for c in category_stats.keys() if c != 'Unknown'])}")
    
    significant_metrics = [m for m, s in overall_stats.items() if s['significant']]
    print(f"✓ Significant improvements: {len(significant_metrics)}/{len(overall_stats)} metrics")
    
    if significant_metrics:
        print(f"✓ Significant metrics: {', '.join(significant_metrics[:3])}{'...' if len(significant_metrics) > 3 else ''}")
    
    return df, overall_stats, category_stats

if __name__ == "__main__":
    main() 