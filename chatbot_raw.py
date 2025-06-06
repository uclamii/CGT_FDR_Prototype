import os
import csv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from enhanced_answer_evaluator import EnhancedAnswerEvaluator

# ---- File Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QUESTIONS_PATH = os.path.join(BASE_DIR, "questions.txt")
OUTPUT_CSV = os.path.join(BASE_DIR, "qa_outputs/questions_answers_raw_enhanced.csv")
MODEL_NAME = "microsoft/phi-4"

# ---- Device Setup ----
if torch.backends.mps.is_available():
    device = torch.device("mps")
    dtype = torch.float16
elif torch.cuda.is_available():
    device = torch.device("cuda")
    dtype = torch.float16
else:
    device = torch.device("cpu")
    dtype = torch.float32

print(f"Using device: {device}, dtype: {dtype}")

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

def generate_answer(question: str, max_retries: int = 3) -> str:
    """Generate an answer using the model."""
    for attempt in range(max_retries):
        try:
            # Create a simpler, more direct prompt
            prompt = f"""Question: {question}
Answer:"""
            print(f"\nSending prompt to model: {prompt}")
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            print("Input shape:", inputs["input_ids"].shape)
            
            # Generate with adjusted parameters
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.9,
                    top_k=50,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    length_penalty=1.0,
                    early_stopping=True
                )
            
            # Decode
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("Raw generated text:", generated_text)
            
            # Extract answer
            if generated_text.startswith(prompt):
                answer = generated_text[len(prompt):].strip()
            else:
                answer = generated_text.strip()
            
            # Check if we got a valid answer
            if answer and len(answer) > 20:  # Basic validation
                print("Extracted answer:", answer)
                return answer
            else:
                print(f"Attempt {attempt + 1}: Empty or too short answer, retrying...")
                continue
                
        except Exception as e:
            print(f"Error in generate_answer (attempt {attempt + 1}): {str(e)}")
            if attempt == max_retries - 1:
                return f"Error: Failed to generate answer after {max_retries} attempts"
            continue
    
    return "Error: Failed to generate a valid answer"

# ---- Load Model and Tokenizer ----
print("Loading model and tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Load model with appropriate settings for MPS
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )
    
    # Move model to device
    model = model.to(device)
    model.eval()
    print("Model and tokenizer loaded successfully")
    
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

# ---- Main Processing ----
def main():
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    # Initialize enhanced evaluator
    evaluator = EnhancedAnswerEvaluator()
    
    # Read questions
    with open(QUESTIONS_PATH, "r") as f:
        questions = [line.strip() for line in f if line.strip()]
    
    # Get question categories
    question_to_category = categorize_questions(questions)
    
    # Define CSV fieldnames including enhanced metrics
    fieldnames = [
        "Question", "Answer", "Category",
        "semantic_similarity", "answer_length", "word_count", "char_count",
        "flesch_reading_ease", "flesch_kincaid_grade", "sentence_count", "avg_sentence_length",
        "sentiment_polarity", "sentiment_subjectivity",
        "bertscore_precision", "bertscore_recall", "bertscore_f1", "has_gold_standard"
    ]
    
    # Process questions and save answers
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for q in questions:
            print(f"\nProcessing question: {q}")
            answer = generate_answer(q)
            
            # Only proceed if we got a valid answer
            if answer and not answer.startswith("Error:"):
                print(f"Final answer: {answer}\n{'-'*100}")
                
                # Evaluate answer with enhanced metrics
                metrics = evaluator.evaluate_answer(q, answer)
                
                # Combine answer and metrics for CSV
                row_data = {
                    "Question": q,
                    "Answer": answer,
                    "Category": question_to_category.get(q, "Unknown"),
                    **metrics
                }
                
                writer.writerow(row_data)
                csvfile.flush()
            else:
                print(f"Skipping question due to invalid answer: {answer}")
                # Write row with empty answer and default metrics
                row_data = {
                    "Question": q,
                    "Answer": "",
                    "Category": question_to_category.get(q, "Unknown"),
                    "semantic_similarity": 0.0,
                    "answer_length": 0,
                    "word_count": 0,
                    "char_count": 0,
                    "flesch_reading_ease": 0.0,
                    "flesch_kincaid_grade": 0.0,
                    "sentence_count": 0,
                    "avg_sentence_length": 0,
                    "sentiment_polarity": 0.0,
                    "sentiment_subjectivity": 0.0,
                    "bertscore_precision": 0.0,
                    "bertscore_recall": 0.0,
                    "bertscore_f1": 0.0,
                    "has_gold_standard": False
                }
                writer.writerow(row_data)
                csvfile.flush()
    
    print(f"\nDone! Enhanced answers and metrics saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main() 