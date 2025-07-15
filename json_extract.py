import os
import json
import logging
import time
import re
import google.generativeai as genai
import PyPDF2
from dotenv import load_dotenv
from threading import Lock
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('book_extraction.log'),
        logging.StreamHandler()
    ]
)

load_dotenv()

class RateLimiter:
    def __init__(self, max_requests=50, per_seconds=60):
        self.max_requests = max_requests
        self.per_seconds = per_seconds
        self.request_times = []
        self.lock = Lock()

    def wait(self):
        current_time = time.time()
        
        with self.lock:
            self.request_times = [
                t for t in self.request_times 
                if current_time - t < self.per_seconds
            ]
            
            while len(self.request_times) >= self.max_requests:
                oldest = self.request_times[0]
                wait_time = self.per_seconds - (current_time - oldest)
                if wait_time > 0:
                    logging.info(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                    current_time = time.time()
                
                self.request_times = [
                    t for t in self.request_times 
                    if current_time - t < self.per_seconds
                ]
            
            self.request_times.append(current_time)

# Initialize rate limiter - increased limits for better API key
RATE_LIMITER = RateLimiter(max_requests=45, per_seconds=60)

# Thread-local storage for models
thread_local = threading.local()

# Global lock for data operations
data_lock = Lock()

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text() + '\n'
            return text
    except Exception as e:
        logging.error(f"Error reading {pdf_path}: {e}")
        return ''

def configure_gemini():
    """Configure Gemini AI model with thread-local storage"""
    try:
        if not hasattr(thread_local, 'model'):
            api_key = os.getenv('GOOGLE_API_KEY')
            genai.configure(api_key=api_key)
            thread_local.model = genai.GenerativeModel('gemini-2.5-flash')
        return thread_local.model
    except Exception as e:
        logging.error(f"Gemini AI Configuration Error: {e}")
        return None

def extract_worksheets_from_text(model, pdf_text, book_number):
    """Use Gemini to extract worksheet data from PDF text"""
    if not model:
        logging.error("Generative AI model not configured")
        return {}

    # Split text into smaller chunks to avoid timeouts
    text_chunks = []
    chunk_size = 4000  # Smaller chunks to avoid token limits
    
    for i in range(0, len(pdf_text), chunk_size):
        chunk = pdf_text[i:i + chunk_size]
        text_chunks.append(chunk)
    
    all_worksheets = {}
    
    for i, chunk in enumerate(text_chunks[:3]):  # Process first 3 chunks only
        try:
            RATE_LIMITER.wait()
            
            prompt = f"""
            You are analyzing text from Book {book_number} answer PDF (chunk {i+1}).
            
            Extract worksheet answers from this text. Look for sections with headings like "Answer - 130", "Answer - 131", etc.
            
            For each worksheet section:
            1. Extract the worksheet number from the heading (e.g., "Answer - 130" → worksheet "130")
            2. Extract all numbered answers below it (1, 2, 3, etc.)
            3. Store each answer exactly as written
            
            Return JSON format:
            {{
                "worksheets": {{
                    "130": ["answer1", "answer2", "answer3"],
                    "131": ["answer1", "answer2"]
                }}
            }}
            
            Text chunk:
            {chunk}
            
            Return ONLY valid JSON, no extra text.
            """
            
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean up response
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            try:
                chunk_data = json.loads(response_text.strip())
                if "worksheets" in chunk_data:
                    all_worksheets.update(chunk_data["worksheets"])
            except json.JSONDecodeError:
                logging.warning(f"Failed to parse JSON for book {book_number}, chunk {i+1}")
                continue
                
            # Add delay between chunks
            time.sleep(1)
            
        except KeyboardInterrupt:
            logging.info("Process interrupted by user")
            break
        except Exception as e:
            logging.warning(f"Error processing chunk {i+1} for book {book_number}: {e}")
            continue
    
    if all_worksheets:
        return {
            "worksheets": all_worksheets,
            "book_number": book_number,
            "extracted_at": datetime.now().isoformat()
        }
    else:
        return {}

def get_book_number(filename):
    """Extract book number from filename"""
    # Handle different filename patterns
    if 'Book' in filename:
        # Extract number after 'Book'
        match = re.search(r'Book(\d+)', filename)
        if match:
            return match.group(1)
    return None

def load_existing_data():
    """Load existing book worksheet data if it exists"""
    try:
        with open('Results/book_worksheets.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"books": {}, "last_updated": None}

def save_data(data):
    """Save data with thread safety"""
    try:
        # Ensure Results directory exists
        os.makedirs('Results', exist_ok=True)
        
        # Add metadata
        data["last_updated"] = datetime.now().isoformat()
        data["total_books_processed"] = len(data.get("books", {}))
        
        # Use a lock for thread-safe writing
        with data_lock:
            with open('Results/book_worksheets.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Data saved to Results/book_worksheets.json")
        return True
    except Exception as e:
        logging.error(f"Error saving data: {e}")
        return False

def process_book_pdf(pdf_path, existing_data):
    """Process a single book PDF"""
    filename = os.path.basename(pdf_path)
    book_number = get_book_number(filename)
    
    if not book_number:
        logging.warning(f"Could not extract book number from {filename}")
        return None
    
    # Skip if already processed
    if book_number in existing_data.get("books", {}):
        logging.info(f"Skipping already processed Book {book_number}")
        return None
    
    logging.info(f"Processing Book {book_number} from {filename}")
    
    # Extract text from PDF
    pdf_text = extract_text_from_pdf(pdf_path)
    if not pdf_text.strip():
        logging.warning(f"No text extracted from {filename}")
        return None
    
    # Configure Gemini
    model = configure_gemini()
    if not model:
        logging.error(f"Failed to configure Gemini for Book {book_number}")
        return None
    
    # Extract worksheet data
    worksheet_data = extract_worksheets_from_text(model, pdf_text, book_number)
    
    if not worksheet_data or "worksheets" not in worksheet_data:
        logging.warning(f"No worksheet data extracted from Book {book_number}")
        return None
    
    # Count extracted worksheets
    num_worksheets = len(worksheet_data.get("worksheets", {}))
    logging.info(f"Extracted {num_worksheets} worksheets from Book {book_number}")
    
    return {
        "book_number": book_number,
        "data": worksheet_data
    }

def main():
    """Main function to process all book PDFs with parallel processing"""
    books_folder = "All book answer"
    
    if not os.path.exists(books_folder):
        logging.error(f"Folder '{books_folder}' not found")
        return
    
    # Load existing data
    existing_data = load_existing_data()
    
    # Get list of PDF files
    pdf_files = [f for f in os.listdir(books_folder) if f.endswith('.pdf')]
    pdf_files.sort()  # Process in order
    
    # Filter out already processed books
    pending_files = []
    for pdf_file in pdf_files:
        book_number = get_book_number(pdf_file)
        if book_number and book_number not in existing_data.get("books", {}):
            pending_files.append(pdf_file)
    
    logging.info(f"Found {len(pdf_files)} total PDF files")
    logging.info(f"Found {len(pending_files)} pending PDF files to process")
    logging.info(f"Already processed: {len(existing_data.get('books', {}))} books")
    
    if not pending_files:
        logging.info("All books have been processed!")
        return
    
    processed_count = 0
    error_count = 0
    
    # Use ThreadPoolExecutor for parallel processing
    max_workers = 3  # Adjust based on your API limits and system capacity
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks
        future_to_file = {
            executor.submit(process_book_pdf, os.path.join(books_folder, pdf_file), existing_data): pdf_file
            for pdf_file in pending_files
        }
        
        try:
            for future in as_completed(future_to_file):
                pdf_file = future_to_file[future]
                
                try:
                    result = future.result()
                    
                    if result:
                        book_number = result["book_number"]
                        
                        # Add to existing data in a thread-safe manner
                        with data_lock:
                            if "books" not in existing_data:
                                existing_data["books"] = {}
                            
                            existing_data["books"][book_number] = result["data"]
                        
                        # Save data incrementally
                        if save_data(existing_data):
                            processed_count += 1
                            logging.info(f"Successfully processed and saved Book {book_number}")
                        else:
                            error_count += 1
                            logging.error(f"Failed to save data for Book {book_number}")
                    else:
                        error_count += 1
                        logging.warning(f"No data extracted from {pdf_file}")
                        
                except Exception as e:
                    error_count += 1
                    logging.error(f"Error processing {pdf_file}: {e}")
        
        except KeyboardInterrupt:
            logging.info("Process interrupted by user")
            executor.shutdown(wait=False)
    
    # Final summary
    logging.info(f"Processing complete!")
    logging.info(f"Successfully processed: {processed_count} books")
    logging.info(f"Errors encountered: {error_count} books")
    logging.info(f"Total books in dataset: {len(existing_data.get('books', {}))}")
    
    # Generate summary report
    if existing_data.get("books"):
        total_worksheets = 0
        for book_data in existing_data["books"].values():
            if isinstance(book_data, dict) and "worksheets" in book_data:
                total_worksheets += len(book_data["worksheets"])
        
        logging.info(f"Total worksheets extracted across all books: {total_worksheets}")

if __name__ == "__main__":
    main()
