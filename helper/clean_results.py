import pandas as pd
import re
import os

def clean_answer_text(answer: str) -> str:
    """
    Clean the answer text by removing 'assistant' from the beginning and all asterisks.
    
    Args:
        answer (str): The original answer text
        
    Returns:
        str: The cleaned answer text
    """
    if pd.isna(answer) or not isinstance(answer, str):
        return answer
    
    # Remove 'assistant' from the beginning (case insensitive)
    # This handles variations like "assistant", "Assistant", "assistantTo", etc.
    cleaned = re.sub(r'^assistant\s*', '', answer, flags=re.IGNORECASE)
    
    # Remove all asterisks
    cleaned = cleaned.replace('*', '')
    
    # Clean up any extra whitespace that might be left
    cleaned = cleaned.strip()
    
    # Remove any leading punctuation that might be left after removing 'assistant'
    cleaned = re.sub(r'^[^\w\s]+', '', cleaned).strip()
    
    return cleaned

def clean_rag_csv(input_file: str, output_file: str = None) -> None:
    """
    Clean the RAG results CSV file by processing the Answer column.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to the output CSV file (optional)
    """
    # Read the CSV file
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} rows from {input_file}")
    except FileNotFoundError:
        print(f"Error: File {input_file} not found!")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Check if 'Answer' column exists
    if 'Answer' not in df.columns:
        print("Error: 'Answer' column not found in the CSV file!")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Clean the answers
    print("Cleaning answers...")
    original_answers = df['Answer'].copy()
    df['Answer'] = df['Answer'].apply(clean_answer_text)
    
    # Count how many answers were modified
    modified_count = 0
    for i, (original, cleaned) in enumerate(zip(original_answers, df['Answer'])):
        if original != cleaned:
            modified_count += 1
    
    print(f"Modified {modified_count} out of {len(df)} answers")
    
    # Determine output file name
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_cleaned.csv"
    
    # Save the cleaned data
    try:
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
    except Exception as e:
        print(f"Error saving file: {e}")
        return
    
    # Show some examples of changes
    print("\nExamples of changes made:")
    print("-" * 50)
    
    example_count = 0
    for i, (original, cleaned) in enumerate(zip(original_answers, df['Answer'])):
        if original != cleaned and example_count < 3:
            print(f"Question {i+1}:")
            print(f"Original: {original[:100]}...")
            print(f"Cleaned:  {cleaned[:100]}...")
            print()
            example_count += 1
    
    if example_count == 0:
        print("No changes were made to the answers.")

def main():
    """Main function to run the cleaning process."""
    # File paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, "qa_outputs", "questions_answers_rag.csv")
    output_file = os.path.join(base_dir, "qa_outputs", "questions_answers_rag_cleaned.csv")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        print("Please make sure the RAG results file exists.")
        return
    
    print("RAG Results Cleaning Script")
    print("=" * 30)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print()
    
    # Clean the file
    clean_rag_csv(input_file, output_file)
    
    print("\nCleaning complete!")
    print(f"You can now use {output_file} for analysis.")

if __name__ == "__main__":
    main() 