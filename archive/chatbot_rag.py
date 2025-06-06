import os
import csv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List, Dict, Tuple
import glob
from answer_evaluator import AnswerEvaluator

# ---- File Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GUIDELINES_DIR = os.path.join(BASE_DIR, "guidelines")
QUESTIONS_PATH = os.path.join(BASE_DIR, "questions.txt")
OUTPUT_CSV = os.path.join(BASE_DIR, "qa_outputs/questions_answers_rag.csv")
MODEL_NAME = "microsoft/phi-4"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

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

def load_documents() -> List[Dict]:
    """Load and process all PDF documents from the guidelines directory."""
    documents = []
    pdf_files = glob.glob(os.path.join(GUIDELINES_DIR, "*.pdf"))
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_path in pdf_files:
        try:
            print(f"Loading: {os.path.basename(pdf_path)}")
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            documents.extend(docs)
            print(f"  - Loaded {len(docs)} pages")
        except Exception as e:
            print(f"Error loading {pdf_path}: {e}")
    
    print(f"Total documents loaded: {len(documents)}")
    return documents

def create_vector_store():
    """Create a vector store from the loaded documents."""
    print("Processing documents and creating vector store...")
    documents = load_documents()
    
    if not documents:
        raise ValueError("No documents loaded! Please check the guidelines directory.")
    
    # Split documents into chunks - adjusted for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Increased from 500
        chunk_overlap=200,  # Increased from 100
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks")
    
    # Create vector store using Chroma
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},  # Use CPU for embeddings to save GPU memory
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=os.path.join(BASE_DIR, "chroma_db")
    )
    return vector_store

def extract_answer_from_output(output: str, prompt: str) -> str:
    """Extract the actual answer from the model output."""
    # Remove the prompt from the output if it's included
    if output.startswith(prompt):
        answer = output[len(prompt):].strip()
    else:
        # Try to find where the answer starts
        answer_markers = ["Answer:", "Response:", "Based on", "According to", "The answer is"]
        for marker in answer_markers:
            if marker in output:
                answer = output.split(marker, 1)[-1].strip()
                break
        else:
            answer = output.strip()
    
    # Clean up the answer
    answer = answer.replace("</s>", "").replace("<|endoftext|>", "").strip()
    
    # Remove any remaining prompt fragments
    if "Question:" in answer:
        answer = answer.split("Question:")[0].strip()
    
    return answer

def generate_answer(question: str, max_retries: int = 2) -> Tuple[str, Dict]:
    """Generate an answer using RAG approach with improved prompting."""
    try:
        # Step 1: Retrieve relevant documents
        print("\nSearching for relevant information in guidelines...")
        relevant_docs = vector_store.similarity_search(question, k=5)  # Increased back to 5
        
        if not relevant_docs:
            print("No relevant documents found!")
            context = ""
        else:
            context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(relevant_docs)])
            print(f"Found {len(relevant_docs)} relevant document chunks")
        
        # Improved RAG prompt
        rag_prompt = f"""You are a helpful medical assistant. Use the following guidelines to answer the patient's question. 
If the guidelines contain relevant information, use it to provide a comprehensive answer.
If the guidelines don't contain enough information, you may supplement with general medical knowledge, but prioritize the guidelines.

Guidelines:
{context}

Patient Question: {question}

Provide a clear, detailed answer that directly addresses the question. Your answer should be informative and helpful.

Answer: """
        
        print("\nGenerating answer...")
        inputs = tokenizer(rag_prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Add attention mask
        attention_mask = inputs.get('attention_mask', None)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=attention_mask,
                max_new_tokens=300,  # Increased from 200
                min_new_tokens=50,   # Added minimum length
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,  # Reduced from 1.2
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                early_stopping=False  # Changed to False to encourage longer answers
            )
        
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = extract_answer_from_output(full_output, rag_prompt)
        
        # Check if we got a meaningful answer
        if answer and len(answer) > 30:  # Increased minimum length check
            print(f"Generated answer (length: {len(answer)} chars)")
            metrics = evaluator.evaluate_answer(question, answer)
            return answer, metrics
        else:
            print(f"Answer too short or empty (length: {len(answer)}), trying with more focused prompt...")
            
            # Try a more focused prompt
            focused_prompt = f"""Based on medical guidelines, please answer this question: {question}

Important: Provide a detailed, informative answer of at least 2-3 sentences. Do not give generic advice about consulting healthcare providers unless specifically relevant to the question.

Answer: """
            
            inputs = tokenizer(focused_prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    min_new_tokens=50,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    early_stopping=False
                )
            
            full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = extract_answer_from_output(full_output, focused_prompt)
            
            if answer and len(answer) > 30:
                print(f"Generated focused answer (length: {len(answer)} chars)")
                metrics = evaluator.evaluate_answer(question, answer)
                return answer, metrics
        
        # Last resort - but make it more informative
        print("Still having trouble generating a good answer, providing informative fallback...")
        fallback_answer = f"Based on the available medical guidelines, {question.lower()} This is an important topic that may require personalized medical advice. While general guidelines exist, individual circumstances can vary significantly. I recommend discussing your specific situation with a healthcare provider who can consider your complete medical history and provide tailored guidance."
        metrics = evaluator.evaluate_answer(question, fallback_answer)
        return fallback_answer, metrics
            
    except Exception as e:
        print(f"Error in generate_answer: {str(e)}")
        import traceback
        traceback.print_exc()
        
        error_answer = f"I encountered an error while processing your question about {question.lower()} Please try rephrasing your question or consult with a healthcare provider for personalized guidance."
        metrics = evaluator.evaluate_answer(question, error_answer)
        return error_answer, metrics

# ---- Load Model and Tokenizer ----
print("Loading model and tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with memory optimizations
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    
    # Set model to eval mode
    model.eval()
    print("Model and tokenizer loaded successfully")
    
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

# ---- Initialize Vector Store and Evaluator ----
print("Initializing vector store...")
vector_store = create_vector_store()
evaluator = AnswerEvaluator()

# ---- Main Processing ----
def main():
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    # Read questions
    with open(QUESTIONS_PATH, "r") as f:
        questions = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(questions)} questions")
    
    # Define CSV fieldnames including metrics
    fieldnames = ["Question", "Answer", 
                 "semantic_similarity", "answer_length", 
                 "flesch_reading_ease", "flesch_kincaid_grade",
                 "sentiment_polarity", "sentiment_subjectivity"]
    
    # Process questions and save answers
    with open(OUTPUT_CSV, "w", newline="", encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, q in enumerate(questions, 1):
            print(f"\n{'='*60}")
            print(f"Processing question {i}/{len(questions)}: {q}")
            print(f"{'='*60}")
            
            try:
                answer, metrics = generate_answer(q)
                
                print(f"\nGenerated Answer: {answer[:100]}..." if len(answer) > 100 else f"\nGenerated Answer: {answer}")
                
                # Write to CSV
                row_data = {
                    "Question": q,
                    "Answer": answer,
                    **metrics
                }
                
                writer.writerow(row_data)
                csvfile.flush()
                
                # Clear memory after each question
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                
            except Exception as e:
                print(f"Error processing question: {q}\nError: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # Write error row
                row_data = {
                    "Question": q,
                    "Answer": f"Error processing question: {str(e)}",
                    "semantic_similarity": 0.0,
                    "answer_length": 0,
                    "flesch_reading_ease": 0.0,
                    "flesch_kincaid_grade": 0.0,
                    "sentiment_polarity": 0.0,
                    "sentiment_subjectivity": 0.0
                }
                writer.writerow(row_data)
                csvfile.flush()
                continue
    
    print(f"\n{'='*60}")
    print(f"Done! Answers and metrics saved to {OUTPUT_CSV}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()