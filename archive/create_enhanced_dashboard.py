import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# Set style for clean visualization
plt.style.use('default')
sns.set_palette("husl")

def load_comparison_data():
    """Load the enhanced comparison CSV with all metrics."""
    base_dir = Path(__file__).parent
    comparison_file = base_dir / "qa_outputs" / "questions_answers_comparison.csv"
    
    if not comparison_file.exists():
        raise FileNotFoundError(f"Comparison file not found: {comparison_file}")
    
    df = pd.read_csv(comparison_file)
    print(f"Loaded comparison data with {len(df)} questions and {len(df.columns)} columns")
    return df

def calculate_comprehensive_stats(df):
    """Calculate comprehensive statistics for all metrics."""
    # Define metrics to analyze
    core_metrics = ['semantic_similarity', 'flesch_reading_ease', 'answer_length', 'word_count']
    advanced_metrics = ['bertscore_precision', 'bertscore_recall', 'bertscore_f1']
    quality_metrics = ['char_count', 'sentence_count', 'avg_sentence_length']
    
    results = {}
    
    # Overall statistics for core metrics
    for metric in core_metrics:
        raw_col = f'{metric}_raw'
        rag_col = f'{metric}_rag'
        
        if raw_col in df.columns and rag_col in df.columns:
            raw_values = df[raw_col].values
            rag_values = df[rag_col].values
            
            raw_mean = np.mean(raw_values)
            rag_mean = np.mean(rag_values)
            improvement_pct = ((rag_mean - raw_mean) / raw_mean) * 100 if raw_mean != 0 else 0
            
            t_stat, p_value = stats.ttest_rel(rag_values, raw_values)
            
            results[metric] = {
                'raw_mean': raw_mean,
                'rag_mean': rag_mean,
                'improvement_pct': improvement_pct,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'raw_std': np.std(raw_values),
                'rag_std': np.std(rag_values)
            }
    
    # BERTScore statistics (only for questions with gold standard)
    valid_bert_df = df[df['has_gold_standard_raw'] == True]
    if len(valid_bert_df) > 0:
        for bert_metric in advanced_metrics:
            raw_col = f'{bert_metric}_raw'
            rag_col = f'{bert_metric}_rag'
            
            if raw_col in valid_bert_df.columns and rag_col in valid_bert_df.columns:
                raw_values = valid_bert_df[raw_col].values
                rag_values = valid_bert_df[rag_col].values
                
                raw_mean = np.mean(raw_values)
                rag_mean = np.mean(rag_values)
                improvement_pct = ((rag_mean - raw_mean) / raw_mean) * 100 if raw_mean != 0 else 0
                
                t_stat, p_value = stats.ttest_rel(rag_values, raw_values)
                
                results[bert_metric] = {
                    'raw_mean': raw_mean,
                    'rag_mean': rag_mean,
                    'improvement_pct': improvement_pct,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'raw_std': np.std(raw_values),
                    'rag_std': np.std(rag_values),
                    'total_valid': len(valid_bert_df)
                }
    
    # Category-specific analysis
    category_results = {}
    for category in df['Category'].unique():
        category_df = df[df['Category'] == category]
        if len(category_df) == 0:
            continue
            
        category_stats = {}
        
        # Core metrics for category
        for metric in core_metrics:
            raw_col = f'{metric}_raw'
            rag_col = f'{metric}_rag'
            
            if raw_col in category_df.columns and rag_col in category_df.columns:
                raw_mean = category_df[raw_col].mean()
                rag_mean = category_df[rag_col].mean()
                improvement = ((rag_mean - raw_mean) / raw_mean) * 100 if raw_mean != 0 else 0
                category_stats[f'{metric}_improvement'] = improvement
        
        # BERTScore for category
        category_bert_df = category_df[category_df['has_gold_standard_raw'] == True]
        if len(category_bert_df) > 0:
            for bert_metric in advanced_metrics:
                raw_col = f'{bert_metric}_raw'
                rag_col = f'{bert_metric}_rag'
                
                if raw_col in category_bert_df.columns and rag_col in category_bert_df.columns:
                    raw_mean = category_bert_df[raw_col].mean()
                    rag_mean = category_bert_df[rag_col].mean()
                    improvement = ((rag_mean - raw_mean) / raw_mean) * 100 if raw_mean != 0 else 0
                    category_stats[f'{bert_metric}_improvement'] = improvement
        
        category_stats['n_questions'] = len(category_df)
        category_stats['n_with_gold'] = len(category_bert_df) if 'category_bert_df' in locals() else 0
        category_results[category] = category_stats
    
    return results, category_results

def create_comprehensive_dashboard(df, overall_stats, category_stats):
    """Create a comprehensive dashboard with all enhanced metrics."""
    
    # Create figure with enhanced layout - 3x3 grid
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.8], width_ratios=[1, 1, 1], 
                         hspace=0.3, wspace=0.25)
    
    fig.suptitle('Enhanced CGT Chatbot Analysis: Comprehensive Performance Evaluation\nRAG vs Raw with BERTScore vs Gold Standard', 
                 fontsize=22, fontweight='bold', y=0.95)
    
    # Color scheme - professional
    raw_color = '#E74C3C'  # Red
    rag_color = '#2ECC71'  # Green
    accent_color = '#3498DB'  # Blue
    neutral_color = '#95A5A6'  # Grey
    
    # 1. Overall Performance Comparison (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ['Semantic\nSimilarity', 'Reading\nEase', 'Answer\nLength', 'BERTScore\nF1']
    raw_vals = [
        overall_stats['semantic_similarity']['raw_mean'], 
        overall_stats['flesch_reading_ease']['raw_mean'],
        overall_stats['answer_length']['raw_mean'],
        overall_stats['bertscore_f1']['raw_mean'] if 'bertscore_f1' in overall_stats else 0
    ]
    rag_vals = [
        overall_stats['semantic_similarity']['rag_mean'], 
        overall_stats['flesch_reading_ease']['rag_mean'],
        overall_stats['answer_length']['rag_mean'],
        overall_stats['bertscore_f1']['rag_mean'] if 'bertscore_f1' in overall_stats else 0
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, raw_vals, width, label='Raw Chatbot', 
                    color=raw_color, alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax1.bar(x + width/2, rag_vals, width, label='RAG Chatbot', 
                    color=rag_color, alpha=0.8, edgecolor='black', linewidth=1)
    
    ax1.set_title('Overall Performance Comparison', fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=10)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add improvement percentages
    improvements = [
        overall_stats['semantic_similarity']['improvement_pct'],
        overall_stats['flesch_reading_ease']['improvement_pct'],
        overall_stats['answer_length']['improvement_pct'],
        overall_stats['bertscore_f1']['improvement_pct'] if 'bertscore_f1' in overall_stats else 0
    ]
    
    for i, (bar1, bar2, imp) in enumerate(zip(bars1, bars2, improvements)):
        height = max(bar1.get_height(), bar2.get_height())
        color = rag_color if imp > 0 else raw_color
        ax1.text(i, height * 1.05, f'{imp:+.1f}%', 
                ha='center', va='bottom', fontweight='bold', fontsize=9, color=color)
    
    # 2. BERTScore Detailed Analysis (Top Center)
    ax2 = fig.add_subplot(gs[0, 1])
    
    valid_bert_df = df[df['has_gold_standard_raw'] == True]
    if len(valid_bert_df) > 0:
        bert_metrics = ['Precision', 'Recall', 'F1']
        raw_bert_vals = [
            overall_stats['bertscore_precision']['raw_mean'],
            overall_stats['bertscore_recall']['raw_mean'],
            overall_stats['bertscore_f1']['raw_mean']
        ]
        rag_bert_vals = [
            overall_stats['bertscore_precision']['rag_mean'],
            overall_stats['bertscore_recall']['rag_mean'],
            overall_stats['bertscore_f1']['rag_mean']
        ]
        
        x_bert = np.arange(len(bert_metrics))
        bars1_bert = ax2.bar(x_bert - width/2, raw_bert_vals, width, label='Raw vs Gold', 
                            color=raw_color, alpha=0.8, edgecolor='black', linewidth=1)
        bars2_bert = ax2.bar(x_bert + width/2, rag_bert_vals, width, label='RAG vs Gold', 
                            color=rag_color, alpha=0.8, edgecolor='black', linewidth=1)
        
        ax2.set_title('BERTScore vs Gold Standard', fontsize=14, fontweight='bold', pad=15)
        ax2.set_ylabel('BERTScore', fontsize=12, fontweight='bold')
        ax2.set_xticks(x_bert)
        ax2.set_xticklabels(bert_metrics, fontsize=11)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, 1)
    
    # 3. Category Performance Heatmap (Top Right)
    ax3 = fig.add_subplot(gs[0, 2])
    
    categories = list(category_stats.keys())
    if categories:
        # Create heatmap data for key metrics
        heatmap_metrics = ['semantic_similarity_improvement', 'flesch_reading_ease_improvement', 'bertscore_f1_improvement']
        heatmap_data = []
        
        for category in categories:
            row = []
            for metric in heatmap_metrics:
                value = category_stats[category].get(metric, 0)
                row.append(value)
            heatmap_data.append(row)
        
        heatmap_array = np.array(heatmap_data)
        
        im = ax3.imshow(heatmap_array, cmap='RdYlGn', aspect='auto', vmin=-25, vmax=25)
        
        # Set ticks and labels
        ax3.set_xticks(range(len(heatmap_metrics)))
        ax3.set_xticklabels(['Semantic\nSimilarity', 'Reading\nEase', 'BERTScore\nF1'], fontsize=10)
        ax3.set_yticks(range(len(categories)))
        ax3.set_yticklabels([cat.replace(' ', '\n') for cat in categories], fontsize=9)
        
        # Add text annotations
        for i in range(len(categories)):
            for j in range(len(heatmap_metrics)):
                value = heatmap_array[i, j]
                ax3.text(j, i, f'{value:+.1f}%', ha='center', va='center', 
                        fontweight='bold', fontsize=8,
                        color='white' if abs(value) > 12 else 'black')
        
        ax3.set_title('Category Improvement Heatmap', fontsize=14, fontweight='bold', pad=15)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3, shrink=0.6)
        cbar.set_label('Improvement %', fontsize=10)
    
    # 4. Distribution Comparison (Middle Left)
    ax4 = fig.add_subplot(gs[1, 0])
    
    semantic_raw = df['semantic_similarity_raw'].values
    semantic_rag = df['semantic_similarity_rag'].values
    
    ax4.hist(semantic_raw, bins=15, alpha=0.6, color=raw_color, 
            edgecolor='black', linewidth=1, label='Raw', density=True)
    ax4.hist(semantic_rag, bins=15, alpha=0.6, color=rag_color, 
            edgecolor='black', linewidth=1, label='RAG', density=True)
    
    ax4.axvline(np.mean(semantic_raw), color=raw_color, linestyle='--', linewidth=2)
    ax4.axvline(np.mean(semantic_rag), color=rag_color, linestyle='--', linewidth=2)
    
    ax4.set_title('Semantic Similarity Distribution', fontsize=14, fontweight='bold', pad=15)
    ax4.set_xlabel('Semantic Similarity Score', fontsize=12)
    ax4.set_ylabel('Density', fontsize=12)
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    # 5. Quality Metrics Comparison (Middle Center)
    ax5 = fig.add_subplot(gs[1, 1])
    
    quality_metrics = ['Word\nCount', 'Sentence\nCount', 'Avg Sentence\nLength']
    raw_quality = [
        df['word_count_raw'].mean(),
        df['sentence_count_raw'].mean(),
        df['avg_sentence_length_raw'].mean()
    ]
    rag_quality = [
        df['word_count_rag'].mean(),
        df['sentence_count_rag'].mean(),
        df['avg_sentence_length_rag'].mean()
    ]
    
    x_qual = np.arange(len(quality_metrics))
    bars1_qual = ax5.bar(x_qual - width/2, raw_quality, width, label='Raw', 
                        color=raw_color, alpha=0.8, edgecolor='black', linewidth=1)
    bars2_qual = ax5.bar(x_qual + width/2, rag_quality, width, label='RAG', 
                        color=rag_color, alpha=0.8, edgecolor='black', linewidth=1)
    
    ax5.set_title('Answer Quality Metrics', fontsize=14, fontweight='bold', pad=15)
    ax5.set_ylabel('Average Count/Length', fontsize=12)
    ax5.set_xticks(x_qual)
    ax5.set_xticklabels(quality_metrics, fontsize=10)
    ax5.legend(fontsize=11)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Gold Standard Coverage (Middle Right)
    ax6 = fig.add_subplot(gs[1, 2])
    
    total_questions = len(df)
    questions_with_gold = df['has_gold_standard_raw'].sum()
    questions_without_gold = total_questions - questions_with_gold
    
    sizes = [questions_with_gold, questions_without_gold]
    labels = [f'With Gold Standard\n({questions_with_gold})', f'Without Gold Standard\n({questions_without_gold})']
    colors = [rag_color, neutral_color]
    
    wedges, texts, autotexts = ax6.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                      startangle=90, textprops={'fontsize': 10})
    
    ax6.set_title('Gold Standard Coverage', fontsize=14, fontweight='bold', pad=15)
    
    # 7. Comprehensive Summary Table (Bottom - spans all columns)
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    # Create comprehensive summary table
    table_data = [
        ['Metric', 'Raw Mean ± SD', 'RAG Mean ± SD', 'Improvement', 'P-value', 'Significant', 'Sample Size'],
    ]
    
    # Add core metrics
    for metric in ['semantic_similarity', 'flesch_reading_ease', 'answer_length']:
        if metric in overall_stats:
            stats_data = overall_stats[metric]
            table_data.append([
                metric.replace('_', ' ').title(),
                f"{stats_data['raw_mean']:.3f} ± {stats_data['raw_std']:.3f}",
                f"{stats_data['rag_mean']:.3f} ± {stats_data['rag_std']:.3f}",
                f"{stats_data['improvement_pct']:+.1f}%",
                f"{stats_data['p_value']:.4f}",
                '✓' if stats_data['significant'] else '✗',
                str(len(df))
            ])
    
    # Add BERTScore metrics
    for metric in ['bertscore_precision', 'bertscore_recall', 'bertscore_f1']:
        if metric in overall_stats:
            stats_data = overall_stats[metric]
            table_data.append([
                metric.replace('_', ' ').title().replace('Bertscore', 'BERTScore'),
                f"{stats_data['raw_mean']:.3f} ± {stats_data['raw_std']:.3f}",
                f"{stats_data['rag_mean']:.3f} ± {stats_data['rag_std']:.3f}",
                f"{stats_data['improvement_pct']:+.1f}%",
                f"{stats_data['p_value']:.4f}",
                '✓' if stats_data['significant'] else '✗',
                str(stats_data['total_valid'])
            ])
    
    # Create table
    table = ax7.table(cellText=table_data[1:], colLabels=table_data[0], 
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style the table
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor(accent_color)
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code the cells
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if j == 3 and table_data[i][j] != 'N/A':  # Improvement column
                if '+' in table_data[i][j]:
                    table[(i, j)].set_facecolor('#D5F4E6')  # Light green
                elif '-' in table_data[i][j]:
                    table[(i, j)].set_facecolor('#FADBD8')  # Light red
            elif j == 5 and table_data[i][j] == '✓':  # Significant column
                table[(i, j)].set_facecolor('#D5F4E6')  # Light green
    
    ax7.set_title('Comprehensive Statistical Analysis Summary', fontsize=16, fontweight='bold', y=0.85)
    
    plt.tight_layout()
    
    # Save the comprehensive dashboard
    plots_dir = Path(__file__).parent / "analysis_results" / "dashboard"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    plt.savefig(plots_dir / "comprehensive_enhanced_analysis.png", 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(plots_dir / "comprehensive_enhanced_analysis.pdf", 
                bbox_inches='tight', facecolor='white')
    
    return fig

def main():
    """Main function to create the comprehensive enhanced dashboard."""
    print("Creating Comprehensive Enhanced CGT Analysis Dashboard")
    print("=" * 55)
    
    # Load comparison data
    df = load_comparison_data()
    
    print("Calculating comprehensive statistics...")
    overall_stats, category_stats = calculate_comprehensive_stats(df)
    
    print("Creating comprehensive dashboard...")
    fig = create_comprehensive_dashboard(df, overall_stats, category_stats)
    
    print("\nComprehensive Results Summary:")
    print("=" * 35)
    
    total_questions = len(df)
    questions_with_gold = df['has_gold_standard_raw'].sum()
    
    print(f"✓ Total Questions Analyzed: {total_questions}")
    print(f"✓ Questions with Gold Standard: {questions_with_gold}")
    print(f"✓ Gold Standard Coverage: {questions_with_gold/total_questions*100:.1f}%")
    print(f"✓ Categories Analyzed: {len(category_stats)}")
    
    print(f"\nKey Performance Improvements:")
    print(f"=" * 32)
    
    for metric in ['semantic_similarity', 'flesch_reading_ease', 'bertscore_f1']:
        if metric in overall_stats:
            improvement = overall_stats[metric]['improvement_pct']
            significant = "✓" if overall_stats[metric]['significant'] else "✗"
            print(f"✓ {metric.replace('_', ' ').title()}: {improvement:+.1f}% (Sig: {significant})")
    
    print(f"\nDashboard saved to: analysis_results/dashboard/")
    print(f"Files: comprehensive_enhanced_analysis.png & .pdf")
    
    plt.show()
    return fig

if __name__ == "__main__":
    main() 