import pandas as pd
import os
import csv
from answer_evaluator import AnswerEvaluator

def reevaluate_cleaned_answers(input_file: str, output_file: str = None):
    """
    Re-evaluate the cleaned RAG answers using the same evaluation metrics.
    
    Args:
        input_file (str): Path to the cleaned CSV file
        output_file (str): Path to save the re-evaluated results
    """
    # Initialize the evaluator
    evaluator = AnswerEvaluator()
    
    # Read the cleaned CSV file
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} rows from {input_file}")
    except FileNotFoundError:
        print(f"Error: File {input_file} not found!")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Check required columns
    if 'Question' not in df.columns or 'Answer' not in df.columns:
        print("Error: Required columns 'Question' and 'Answer' not found!")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Determine output file name
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_reevaluated.csv"
    
    print("Re-evaluating cleaned answers...")
    print(f"Output will be saved to: {output_file}")
    
    # Define CSV fieldnames including metrics
    fieldnames = ["Question", "Answer", 
                 "semantic_similarity", "answer_length", 
                 "flesch_reading_ease", "flesch_kincaid_grade",
                 "sentiment_polarity", "sentiment_subjectivity"]
    
    # Process each question-answer pair and re-evaluate
    with open(output_file, "w", newline="", encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for index, row in df.iterrows():
            question = row['Question']
            answer = row['Answer']
            
            print(f"Processing {index + 1}/{len(df)}: {question[:50]}...")
            
            try:
                # Evaluate the cleaned answer
                metrics = evaluator.evaluate_answer(question, answer)
                
                # Write to CSV
                row_data = {
                    "Question": question,
                    "Answer": answer,
                    **metrics
                }
                
                writer.writerow(row_data)
                csvfile.flush()
                
            except Exception as e:
                print(f"Error evaluating question {index + 1}: {str(e)}")
                # Write row with error metrics
                row_data = {
                    "Question": question,
                    "Answer": answer,
                    "semantic_similarity": 0.0,
                    "answer_length": len(answer) if isinstance(answer, str) else 0,
                    "flesch_reading_ease": 0.0,
                    "flesch_kincaid_grade": 0.0,
                    "sentiment_polarity": 0.0,
                    "sentiment_subjectivity": 0.0
                }
                writer.writerow(row_data)
                csvfile.flush()
                continue
    
    print(f"\nRe-evaluation complete! Results saved to {output_file}")
    
    # Show comparison summary
    print("\nComparison Summary:")
    print("-" * 40)
    
    # Load original results for comparison if they exist
    original_file = input_file.replace('_cleaned', '')
    if os.path.exists(original_file):
        try:
            original_df = pd.read_csv(original_file)
            new_df = pd.read_csv(output_file)
            
            metrics = ['semantic_similarity', 'answer_length', 'flesch_reading_ease', 
                      'flesch_kincaid_grade', 'sentiment_polarity', 'sentiment_subjectivity']
            
            print("Metric Comparison (Original vs Cleaned):")
            for metric in metrics:
                if metric in original_df.columns and metric in new_df.columns:
                    orig_mean = original_df[metric].mean()
                    new_mean = new_df[metric].mean()
                    change = new_mean - orig_mean
                    change_pct = (change / orig_mean * 100) if orig_mean != 0 else 0
                    
                    print(f"{metric:20}: {orig_mean:8.3f} → {new_mean:8.3f} "
                          f"(Δ {change:+7.3f}, {change_pct:+6.1f}%)")
            
        except Exception as e:
            print(f"Could not load original file for comparison: {e}")
    else:
        print(f"Original file {original_file} not found for comparison.")

def compare_with_original(cleaned_file: str, original_file: str = None):
    """
    Compare the re-evaluated cleaned results with the original results.
    
    Args:
        cleaned_file (str): Path to the re-evaluated cleaned results
        original_file (str): Path to the original results (optional)
    """
    if original_file is None:
        # Try to infer original file name
        original_file = cleaned_file.replace('_cleaned_reevaluated', '').replace('_reevaluated', '')
    
    if not os.path.exists(original_file):
        print(f"Original file {original_file} not found for detailed comparison.")
        return
    
    try:
        original_df = pd.read_csv(original_file)
        cleaned_df = pd.read_csv(cleaned_file)
        
        print(f"\nDetailed Comparison:")
        print(f"Original file: {original_file}")
        print(f"Cleaned file:  {cleaned_file}")
        print("-" * 60)
        
        # Merge dataframes for comparison
        merged = pd.merge(original_df, cleaned_df, on='Question', suffixes=('_orig', '_clean'))
        
        metrics = ['semantic_similarity', 'answer_length', 'flesch_reading_ease', 
                  'flesch_kincaid_grade', 'sentiment_polarity', 'sentiment_subjectivity']
        
        improvements = {}
        for metric in metrics:
            orig_col = f'{metric}_orig'
            clean_col = f'{metric}_clean'
            
            if orig_col in merged.columns and clean_col in merged.columns:
                diff = merged[clean_col] - merged[orig_col]
                improvements[metric] = {
                    'mean_change': diff.mean(),
                    'improved_count': (diff > 0).sum(),
                    'degraded_count': (diff < 0).sum(),
                    'unchanged_count': (diff == 0).sum(),
                    'total': len(diff)
                }
        
        print("\nImprovement Analysis:")
        print(f"{'Metric':<20} {'Mean Δ':<10} {'Improved':<10} {'Degraded':<10} {'Unchanged':<10}")
        print("-" * 70)
        
        for metric, stats in improvements.items():
            print(f"{metric:<20} {stats['mean_change']:+8.3f} "
                  f"{stats['improved_count']:>8} {stats['degraded_count']:>9} "
                  f"{stats['unchanged_count']:>9}")
        
    except Exception as e:
        print(f"Error in detailed comparison: {e}")

def main():
    """Main function to run the re-evaluation process."""
    # File paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, "qa_outputs", "questions_answers_rag_cleaned.csv")
    output_file = os.path.join(base_dir, "qa_outputs", "questions_answers_rag_cleaned_reevaluated.csv")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Cleaned input file not found at {input_file}")
        print("Please run the cleaning script first: python clean_rag_results.py")
        return
    
    print("RAG Results Re-evaluation Script")
    print("=" * 35)
    print(f"Input file:  {input_file}")
    print(f"Output file: {output_file}")
    print()
    
    # Re-evaluate the cleaned answers
    reevaluate_cleaned_answers(input_file, output_file)
    
    # Perform detailed comparison
    original_file = os.path.join(base_dir, "qa_outputs", "questions_answers_rag.csv")
    compare_with_original(output_file, original_file)
    
    print(f"\nRe-evaluation complete!")
    print(f"You can now use {output_file} for analysis or comparison.")
    print(f"To compare with raw results, update your analysis script to use:")
    print(f"  - Raw results: qa_outputs/questions_answers_raw.csv")
    print(f"  - Cleaned RAG results: {output_file}")

if __name__ == "__main__":
    main() 