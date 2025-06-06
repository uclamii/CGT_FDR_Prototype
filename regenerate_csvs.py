import pandas as pd
import numpy as np
from pathlib import Path
import os
from enhanced_answer_evaluator import EnhancedAnswerEvaluator
import csv

def categorize_questions(questions):
    """Categorize questions into predefined categories."""
    QUESTION_CATEGORIES = {
        "Genetic Variant Interpretation": [
            "What does this genetic variant mean for me?",
            "Does this mean I will definitely have cancer?",
            "Does this genetic variant affect my cancer treatment?",
            "How might my genetic test results change over time?",
            "What is Lynch Syndrome?"
        ],
        
        "Inheritance Patterns": [
            "Is this variant something I inherited?",
            "Can only women can carry a BRCA inherited mutation?",
            "Can I give this to my kids?",
            "Can this variant skip a generation?",
            "What if I want to have children and have a hereditary cancer gene? What are my reproductive options?",
            "I have a BRCA pathogenic variant and I want to have children, what are my options?",
            "Why do some families with Lynch syndrome have more cases of cancer than others?"
        ],
        
        "Family Risk Assessment": [
            "Why should I share with family my genetic results?",
            "Who are my first-degree relatives?",
            "Should my family members get tested?",
            "Which of my relatives are at risk?",
            "Should I contact my male and female relatives?",
            "What if a family member doesn't want to get tested?",
            "How can I get my kids tested?",
            "At what age should my children get tested?",
            "Why would my relatives want to know if they have this? What can they do about it?",
            "I don't talk to my family/parents/sister/brother. How can I share this with them?",
            "Who do my family members call to have genetic testing?"
        ],
        
        "Gene-Specific Recommendations": [
            "What are the recommendations for my family members if I have a mutation in (specify gene: MSH2, MSH1, MSH6, PMS2, EPCAM/MSH2, BRCA1, BRCA2)?",
            "What types of cancers am I at risk for?",
            "What screening tests do you recommend?",
            "What steps can I take to manage my cancer risk if I have Lynch syndrome? (not specific to variant)",
            "What are the Risks and Benefits of Risk-Reducing Surgeries for Lynch Syndrome?",
            "What is my cancer risk if I have MSH2 or EPCAM- associated Lynch syndrome?",
            "What is my cancer risk if I have PMS2 Lynch syndrome?",
            "What is my cancer risk if I have MSH1 Lynch syndrome?",
            "What is my cancer risk if I have MSH6 Lynch syndrome?",
            "What is my cancer risk if I have BRCA2 Hereditary Breast and Ovarian Cancer syndrome?",
            "What is my cancer risk if I have BRCA1 Hereditary Breast and Ovarian Cancer syndrome?",
            "What are the surveillance and preventions I can take to reduce my risk of cancer or detecting cancer early if I have a EPCAM/MSH2 mutation?",
            "What are the surveillance and preventions I can take to reduce my risk of cancer or detecting cancer early if I have an MSH2 mutation?",
            "What are the surveillance and preventions I can take to reduce my risk of cancer or detecting cancer early if I have a BRCA mutation?"
        ],
        
        "Support and Resources": [
            "Is genetic testing for my family members covered by insurance?",
            "Will this affect my health insurance?",
            "People who test positive for a genetic mutation are they at risk of losing their health insurance?",
            "Does GINA cover life or disability insurance?",
            "Will my insurance cover testing for my parents/brother/sister?",
            "My [relative] doesn't have insurance. What should they do?",
            "How can I cope with this diagnosis?",
            "What if I feel overwhelmed?",
            "Is new research being done on my condition?",
            "How can I help others with my condition?",
            "Where can I find a genetic counselor?",
            "What other resources are available to help me?"
        ]
    }
    
    question_to_category = {}
    for category, question_list in QUESTION_CATEGORIES.items():
        for question in question_list:
            question_to_category[question] = category
    return question_to_category

def load_existing_csvs():
    """Load existing CSV files."""
    base_dir = Path(__file__).parent
    
    raw_df = pd.read_csv(base_dir / "qa_outputs" / "questions_answers_raw.csv")
    rag_df = pd.read_csv(base_dir / "qa_outputs" / "questions_answers_rag.csv")
    
    return raw_df, rag_df

def regenerate_enhanced_csvs():
    """Regenerate CSV files with enhanced metrics and proper formatting."""
    print("Regenerating Enhanced CSV Files with Comprehensive Metrics")
    print("=" * 60)
    
    # Initialize enhanced evaluator
    evaluator = EnhancedAnswerEvaluator()
    
    # Load existing data
    print("Loading existing CSV files...")
    raw_df, rag_df = load_existing_csvs()
    
    # Define enhanced fieldnames
    enhanced_fieldnames = [
        "Question", "Answer", "Category",
        "semantic_similarity", "answer_length", "word_count", "char_count",
        "flesch_reading_ease", "flesch_kincaid_grade", "sentence_count", "avg_sentence_length",
        "sentiment_polarity", "sentiment_subjectivity",
        "bertscore_precision", "bertscore_recall", "bertscore_f1", "has_gold_standard"
    ]
    
    # Get question categories
    all_questions = raw_df['Question'].tolist()
    question_to_category = categorize_questions(all_questions)
    
    # Process Raw CSV with enhanced metrics
    print("\nProcessing Raw CSV with enhanced metrics...")
    enhanced_raw_path = Path(__file__).parent / "qa_outputs" / "questions_answers_raw_enhanced.csv"
    
    with open(enhanced_raw_path, "w", newline="", encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=enhanced_fieldnames)
        writer.writeheader()
        
        for _, row in raw_df.iterrows():
            question = row['Question']
            answer = row['Answer']
            
            print(f"  Processing Raw: {question[:50]}...")
            
            # Calculate enhanced metrics
            enhanced_metrics = evaluator.evaluate_answer(question, answer)
            
            # Prepare row data
            row_data = {
                "Question": question,
                "Answer": answer,
                "Category": question_to_category.get(question, "Unknown"),
                **enhanced_metrics
            }
            
            writer.writerow(row_data)
    
    # Process RAG CSV with enhanced metrics
    print("\nProcessing RAG CSV with enhanced metrics...")
    enhanced_rag_path = Path(__file__).parent / "qa_outputs" / "questions_answers_rag_enhanced.csv"
    
    with open(enhanced_rag_path, "w", newline="", encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=enhanced_fieldnames)
        writer.writeheader()
        
        for _, row in rag_df.iterrows():
            question = row['Question']
            answer = row['Answer']
            
            print(f"  Processing RAG: {question[:50]}...")
            
            # Calculate enhanced metrics
            enhanced_metrics = evaluator.evaluate_answer(question, answer)
            
            # Prepare row data
            row_data = {
                "Question": question,
                "Answer": answer,
                "Category": question_to_category.get(question, "Unknown"),
                **enhanced_metrics
            }
            
            writer.writerow(row_data)
    
    # Create merged comparison CSV with proper suffixes for dashboard
    print("\nCreating merged comparison CSV for dashboard...")
    merged_path = Path(__file__).parent / "qa_outputs" / "questions_answers_comparison.csv"
    
    # Load the enhanced CSVs
    enhanced_raw_df = pd.read_csv(enhanced_raw_path)
    enhanced_rag_df = pd.read_csv(enhanced_rag_path)
    
    # Merge on Question with proper suffixes
    merged_df = pd.merge(enhanced_raw_df, enhanced_rag_df, on='Question', suffixes=('_raw', '_rag'))
    
    # Keep only one Category column (they should be the same)
    if 'Category_raw' in merged_df.columns:
        merged_df['Category'] = merged_df['Category_raw']
        merged_df.drop(['Category_raw', 'Category_rag'], axis=1, inplace=True)
    
    # Save merged comparison
    merged_df.to_csv(merged_path, index=False)
    
    print(f"\nEnhanced CSV files generated:")
    print(f"✓ Raw Enhanced: {enhanced_raw_path}")
    print(f"✓ RAG Enhanced: {enhanced_rag_path}")
    print(f"✓ Merged Comparison: {merged_path}")
    
    # Generate summary statistics
    print(f"\nSummary Statistics:")
    print(f"=" * 30)
    
    total_questions = len(merged_df)
    questions_with_gold_standard = merged_df['has_gold_standard_raw'].sum()
    
    print(f"Total Questions: {total_questions}")
    print(f"Questions with Gold Standard: {questions_with_gold_standard}")
    print(f"Gold Standard Coverage: {questions_with_gold_standard/total_questions*100:.1f}%")
    
    # Category breakdown
    print(f"\nCategory Breakdown:")
    category_counts = merged_df['Category'].value_counts()
    for category, count in category_counts.items():
        print(f"  {category}: {count} questions")
    
    # Quick performance comparison
    print(f"\nQuick Performance Comparison:")
    print(f"=" * 35)
    
    # Semantic similarity
    raw_sem_mean = merged_df['semantic_similarity_raw'].mean()
    rag_sem_mean = merged_df['semantic_similarity_rag'].mean()
    sem_improvement = ((rag_sem_mean - raw_sem_mean) / raw_sem_mean) * 100
    print(f"Semantic Similarity - Raw: {raw_sem_mean:.3f}, RAG: {rag_sem_mean:.3f} ({sem_improvement:+.1f}%)")
    
    # Reading ease
    raw_read_mean = merged_df['flesch_reading_ease_raw'].mean()
    rag_read_mean = merged_df['flesch_reading_ease_rag'].mean()
    read_improvement = ((rag_read_mean - raw_read_mean) / raw_read_mean) * 100
    print(f"Reading Ease - Raw: {raw_read_mean:.1f}, RAG: {rag_read_mean:.1f} ({read_improvement:+.1f}%)")
    
    # BERTScore (only for questions with gold standard)
    valid_bert_df = merged_df[merged_df['has_gold_standard_raw'] == True]
    if len(valid_bert_df) > 0:
        raw_bert_mean = valid_bert_df['bertscore_f1_raw'].mean()
        rag_bert_mean = valid_bert_df['bertscore_f1_rag'].mean()
        bert_improvement = ((rag_bert_mean - raw_bert_mean) / raw_bert_mean) * 100
        print(f"BERTScore F1 vs Gold - Raw: {raw_bert_mean:.3f}, RAG: {rag_bert_mean:.3f} ({bert_improvement:+.1f}%)")
        print(f"BERTScore based on {len(valid_bert_df)} questions with gold standard")
    
    return merged_path

if __name__ == "__main__":
    enhanced_csv_path = regenerate_enhanced_csvs()
    print(f"\n✅ Enhanced CSV generation complete!")
    print(f"Main output file: {enhanced_csv_path}") 