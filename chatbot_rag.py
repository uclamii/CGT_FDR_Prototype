import os
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

class RAGChatbot:
    def __init__(self):
        # Initialize the LLM
        print("Loading LLM model...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        ).to(device)
        self.model.eval()

        # Initialize the embedding model
        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        # Process documents and create vector store
        self.vector_store = self._create_vector_store()
        
        # Initialize answer evaluator
        self.evaluator = AnswerEvaluator()

    def _load_documents(self) -> List[Dict]:
        """Load and process all PDF documents from the guidelines directory."""
        documents = []
        pdf_files = glob.glob(os.path.join(GUIDELINES_DIR, "*.pdf"))
        
        for pdf_path in pdf_files:
            try:
                loader = PyPDFLoader(pdf_path)
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {pdf_path}: {e}")
        
        return documents

    def _create_vector_store(self):
        """Create a vector store from the loaded documents."""
        print("Processing documents and creating vector store...")
        documents = self._load_documents()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create vector store using Chroma
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=os.path.join(BASE_DIR, "chroma_db")
        )
        return vector_store

    def _clean_response(self, response: str) -> str:
        """Clean up the response by removing unwanted formatting and repetitions."""
        # Remove any question-answer pairs that might have been generated
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            if not line.startswith('Question:') and not line.startswith('Answer:'):
                cleaned_lines.append(line)
        
        # Join lines and clean up extra whitespace
        cleaned = ' '.join(cleaned_lines)
        cleaned = ' '.join(cleaned.split())  # Remove extra whitespace
        
        # Remove bullet points and other formatting
        cleaned = cleaned.replace('•', '').replace('·', '')
        
        return cleaned.strip()

    def _generate_with_context(self, question: str, context: str) -> str:
        """Generate an answer using the provided context."""
        prompt = f"""You are a medical assistant helping patients understand their genetic test results and cancer risks. 
        Use the following context from medical guidelines to answer the patient's question.
        Provide a clear, concise, and accurate response based ONLY on the given context.
        If the answer cannot be found in the context, say "I cannot find specific information about this in the guidelines."
        Do not include any question-answer pairs in your response.
        Do not repeat yourself.
        Keep the response focused and to the point.

        Context:
        {context}

        Patient's Question: {question}

        Your Response:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.9,
                top_p=0.9,
                top_k=50,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                early_stopping=True
            )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._clean_response(answer[len(prompt):].strip())

    def _generate_general_answer(self, question: str) -> str:
        """Generate an answer using the LLM's general knowledge."""
        prompt = f"""You are a medical assistant helping patients understand their genetic test results and cancer risks.
        Answer the following question based on your general medical knowledge.
        Be clear, concise, and accurate. If you're not certain about something, say so.
        Do not include any question-answer pairs in your response.
        Do not repeat yourself.
        Keep the response focused and to the point.

        Patient's Question: {question}

        Your Response:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.9,
                top_p=0.9,
                top_k=50,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                early_stopping=True
            )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._clean_response(answer[len(prompt):].strip())

    def generate_answer(self, question: str) -> Tuple[str, Dict]:
        """Generate an answer using RAG approach, falling back to general knowledge if needed."""
        try:
            # Retrieve relevant documents
            relevant_docs = self.vector_store.similarity_search(question, k=3)
            context = "\n".join([doc.page_content for doc in relevant_docs])
            
            # First try to answer using the context
            answer = self._generate_with_context(question, context)
            
            # If the answer indicates no information was found, try general knowledge
            if "cannot find specific information" in answer.lower():
                print("No specific information found in guidelines, using general knowledge...")
                general_answer = self._generate_general_answer(question)
                answer = f"{answer}\n\nHowever, based on general medical knowledge: {general_answer}"
            
            # Evaluate the answer
            metrics = self.evaluator.evaluate_answer(question, answer)
            
            return answer, metrics
            
        except Exception as e:
            print(f"Error generating answer for question:\n{question}\n{e}")
            error_answer = "I apologize, but I encountered an error while generating the answer. Please try again."
            metrics = self.evaluator.evaluate_answer(question, error_answer)
            return error_answer, metrics

def main():
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    # Initialize the chatbot
    chatbot = RAGChatbot()
    
    # Read questions
    with open(QUESTIONS_PATH, "r") as f:
        questions = [line.strip() for line in f if line.strip()]
    
    # Define CSV fieldnames including metrics
    fieldnames = ["Question", "Answer", 
                 "semantic_similarity", "answer_length", 
                 "flesch_reading_ease", "flesch_kincaid_grade",
                 "sentiment_polarity", "sentiment_subjectivity"]
    
    # Process questions and save answers
    import csv
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for q in questions:
            print(f"Question: {q}")
            answer, metrics = chatbot.generate_answer(q)
            print(f"Answer: {answer}\n{'-'*100}")
            
            # Combine answer and metrics for CSV
            row_data = {
                "Question": q,
                "Answer": answer,
                **metrics
            }
            
            writer.writerow(row_data)
            csvfile.flush()
    
    print(f"Done! Answers and metrics saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
