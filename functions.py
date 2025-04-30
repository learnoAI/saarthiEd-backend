import os
import json
import logging
import time
import concurrent.futures
import google.generativeai as genai
import PyPDF2
from dotenv import load_dotenv
from threading import Lock

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('pdf_processing.log'),
        logging.StreamHandler()
    ]
)

load_dotenv()

class RateLimiter:
    def __init__(self, max_requests=30, per_seconds=60):
        self.max_requests = max_requests
        self.per_seconds = per_seconds
        self.request_times = []
        self.lock = Lock()

    def wait(self):
        current_time = time.time()
        
        with self.lock:
            # Remove timestamps outside the current window
            self.request_times = [
                t for t in self.request_times 
                if current_time - t < self.per_seconds
            ]
            
            # If we've reached max requests, wait
            while len(self.request_times) >= self.max_requests:
                oldest = self.request_times[0]
                wait_time = self.per_seconds - (current_time - oldest)
                if wait_time > 0:
                    logging.info(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                    current_time = time.time()
                
                # Clean up old timestamps
                self.request_times = [
                    t for t in self.request_times 
                    if current_time - t < self.per_seconds
                ]
            
            # Record this request
            self.request_times.append(current_time)

# Global rate limiter
RATE_LIMITER = RateLimiter(max_requests=30, per_seconds=60)

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file with error handling.
    """
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        logging.error(f"Error reading {pdf_path}: {e}")
        return ''

def configure_generative_ai():
    """
    Configure Google's Generative AI with error handling and rate limiting.
    """
    try:
        # Apply rate limiting
        RATE_LIMITER.wait()
        
        api_key = os.getenv('GOOGLE_API_KEY')
        
        if not api_key:
            raise ValueError("No API key found. Please set GOOGLE_API_KEY environment variable.")
        
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-2.0-flash-lite')
    
    except Exception as e:
        logging.error(f"Gemini AI Configuration Error: {e}")
        return None

def extract_questions_and_answers(model, question_text, answer_text):
    """
    Use Gemini to extract questions and corresponding answers with rate limiting.
    """
    if not model:
        logging.error("Generative AI model not configured")
        return []

    # Apply rate limiting before API call
    RATE_LIMITER.wait()

    prompt = f"""
    Carefully extract questions and their corresponding answers from the following texts:

    Questions Text:
    {question_text}

    Answers Text:
    {answer_text}

    Guidelines:
    - Return a JSON-formatted list of dictionaries
    - Each dictionary must have 'question' and 'answer' keys
    - If no answer found for a question, use an empty string
    - If no questions are found, return an empty list
    - Ensure high accuracy in matching questions and answers
    - Limit response to top 20 questions if many are present
    """
    
    try:
        # Ensure rate limiting is applied
        RATE_LIMITER.wait()
        
        response = model.generate_content(prompt)
        
        # Multiple parsing attempts
        parsing_attempts = [
            lambda: json.loads(response.text),
            lambda: json.loads(response.text.strip('```json\n').strip('```')),
            lambda: json.loads(response.text.replace('\n', ''))
        ]
        
        for attempt in parsing_attempts:
            try:
                return attempt()
            except json.JSONDecodeError:
                continue
        
        return []
    
    except Exception as e:
        logging.error(f"Error extracting Q&A: {e}")
        return []

def process_single_pdf(dataset_path, range_folder, pdf, existing_data):
    """
    Process a single PDF file with comprehensive checks and rate limiting
    """
    # Skip already processed files
    pdf_id = pdf.split('.')[0].replace('answer', '')
    
    # Check if already processed in existing data
    if range_folder in existing_data and pdf_id in existing_data[range_folder]:
        logging.info(f"Skipping already processed file: {pdf}")
        return None
    
    # Ensure it's a question PDF
    if pdf.endswith('answer.pdf'):
        return None
    
    # Construct full paths
    question_path = os.path.join(dataset_path, range_folder, pdf)
    answer_path = os.path.join(dataset_path, range_folder, f"{pdf_id}answer.pdf")
    
    # Extract texts
    question_text = extract_text_from_pdf(question_path)
    answer_text = extract_text_from_pdf(answer_path) if os.path.exists(answer_path) else ''
    
    # Configure AI with global rate limiter
    try:
        model = configure_generative_ai()
        
        # Extract Q&A with global rate limiting
        qa_data = extract_questions_and_answers(model, question_text, answer_text)
    except Exception as e:
        logging.error(f"Processing failed for {pdf_id}: {e}")
        return None
    
    logging.info(f"Processed {pdf_id} with {len(qa_data)} questions")
    
    return {
        'range_folder': range_folder,
        'pdf_id': pdf_id,
        'qa_data': qa_data
    }

def process_dataset_folder(dataset_path, max_workers=3):  # Reduced workers to respect rate limit
    """
    Process dataset folder with controlled parallel processing and incremental updates
    """
    # Load existing data to avoid reprocessing
    try:
        with open('dataset_questions_answers.json', 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = {}
    
    # Prepare for parallel processing
    pdf_jobs = []
    for range_folder in os.listdir(dataset_path):
        range_path = os.path.join(dataset_path, range_folder)
        
        if not os.path.isdir(range_path):
            continue
        
        pdfs = [f for f in os.listdir(range_path) if f.endswith('.pdf')]
        
        for pdf in pdfs:
            pdf_jobs.append((dataset_path, range_folder, pdf, existing_data))
    
    # Parallel processing with reduced workers
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pdf = {
            executor.submit(process_single_pdf, *job): job 
            for job in pdf_jobs
        }
        
        for future in concurrent.futures.as_completed(future_to_pdf):
            try:
                result = future.result()
                if result:
                    results.append(result)
                    
                    # Incremental update to JSON
                    if result['range_folder'] not in existing_data:
                        existing_data[result['range_folder']] = {}
                    
                    existing_data[result['range_folder']][result['pdf_id']] = result['qa_data']
                    
                    # Save after each successful processing
                    with open('dataset_questions_answers.json', 'w', encoding='utf-8') as f:
                        json.dump(existing_data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logging.error(f"Error in processing: {e}")
    
    logging.info(f"Processed {len(results)} PDFs")
    return existing_data

def main():
    dataset_path = 'Dataset'  # Replace with your actual dataset path
    
    logging.info("Starting PDF processing")
    
    try:
        result = process_dataset_folder(dataset_path)
        logging.info("Processing complete. Final JSON saved.")
    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")

# if __name__ == '__main__':
#     main()