import os
import json
import re
from conns import s3_client, gemini_client, collection
from google.genai import types
from datetime import datetime, timedelta
from PIL import Image
import io

S3_BUCKET_NAME = "learno-pdf-document"

def upload_to_s3(file_path):
    try:
        file_name = os.path.basename(file_path)
        s3_key = f"worksheets-{file_name}"
        
        with open(file_path, 'rb') as file_data:
            s3_client.upload_fileobj(
                file_data, 
                S3_BUCKET_NAME, 
                s3_key,
                ExtraArgs={'ACL': 'public-read'}
            )
        
        s3_url = f"https://{S3_BUCKET_NAME}.s3.ap-south-1.amazonaws.com/{s3_key}"
        return s3_url
    except Exception as e:
        print(f"Error uploading to S3: {str(e)}")
        return None

def use_gemini_for_ocr(image_bytes_list):
    try:
        if not isinstance(image_bytes_list, list):
            image_bytes_list = [image_bytes_list]
        
        processed_images = []
        for image_bytes in image_bytes_list:
            image = Image.open(io.BytesIO(image_bytes))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='JPEG', quality=95)
            processed_images.append(img_buffer.getvalue())
        
        ocr_prompt = """
            Extract all questions and their corresponding student answers from these worksheet images. 
            Return the data in JSON format where each question is keyed as "q1", "q2", etc.

            IMPORTANT INSTRUCTIONS:
            1. Extract the student's answers EXACTLY as written - do not correct or modify them
            2. If a question is unanswered, use an empty string "" for the answer
            3. Include both the question text and the student's answer
            4. Number questions sequentially starting from q1
            5. These may be multiple images from the same worksheet - consolidate all questions and answers
            6. For duplicate questions across images, include all versions with unique keys (q1, q2, etc.)

            Return format:
            {
            "q1": {
                "question": "<question_text>",
                "answer": "<student_answer_exactly_as_written>"
            },
            "q2": {
                "question": "<question_text>", 
                "answer": "<student_answer_exactly_as_written>"
            }
            }
            """
        
        contents = [ocr_prompt]
        for image_data in processed_images:
            contents.append(types.Part.from_bytes(data=image_data, mime_type='image/jpeg'))
            
        print(f"Sending {len(processed_images)} images to Gemini OCR in a single request...")
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash-preview-05-20',
            contents=contents
        )
        
        response_text = response.text.strip()
        print(response_text)
        
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        result = json.loads(response_text)
        print(f"Gemini OCR extracted {len(result)} questions")
        return result
        
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {str(e)}")
        print(f"Raw response: {response.text}")
        return {"error": f"Failed to parse OCR response as JSON: {str(e)}"}
    except Exception as e:
        print(f"Gemini OCR error: {str(e)}")
        return {"error": f"Gemini OCR error: {str(e)}"}

def load_book_worksheets_data():
    """Load the book worksheets JSON data from file."""
    try:
        book_worksheets_path = os.path.join(os.path.dirname(__file__), 'Results', 'book_worksheets.json')
        with open(book_worksheets_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading book worksheets data: {str(e)}")
        return None

def compare_answers_with_book_data(student_answers, worksheet_name, book_worksheets_data):
    """Compare student answers with correct answers from book_worksheets.json."""
    try:
        # Extract book number and worksheet number from worksheet_name
        import re
        
        book_number = None
        worksheet_number = None
        
        # Try different parsing strategies
        worksheet_name_clean = worksheet_name.strip()
        
        # Strategy 1: Just a number (assume it's worksheet ID)
        if worksheet_name_clean.isdigit():
            worksheet_number = worksheet_name_clean
            # Find which book contains this worksheet
            for book_id, book_data in book_worksheets_data.get('books', {}).items():
                if worksheet_number in book_data.get('worksheets', {}):
                    book_number = book_id
                    break
        
        # Strategy 2: Format like "Book10-Worksheet370" or "Book 10 Worksheet 370"
        elif 'book' in worksheet_name_clean.lower() and 'worksheet' in worksheet_name_clean.lower():
            # Extract numbers from the string
            numbers = re.findall(r'\d+', worksheet_name_clean)
            if len(numbers) >= 2:
                book_number = numbers[0]
                worksheet_number = numbers[1]
            elif len(numbers) == 1:
                # Could be "Worksheet370" format
                worksheet_number = numbers[0]
                # Find which book contains this worksheet
                for book_id, book_data in book_worksheets_data.get('books', {}).items():
                    if worksheet_number in book_data.get('worksheets', {}):
                        book_number = book_id
                        break
        
        # Strategy 3: Format like "10-370" or "10_370"
        elif re.match(r'^\d+[-_]\d+$', worksheet_name_clean):
            numbers = re.findall(r'\d+', worksheet_name_clean)
            if len(numbers) == 2:
                book_number = numbers[0]
                worksheet_number = numbers[1]
        
        # Strategy 4: Try to extract just worksheet number from formats like "Worksheet370"
        elif 'worksheet' in worksheet_name_clean.lower():
            numbers = re.findall(r'\d+', worksheet_name_clean)
            if numbers:
                worksheet_number = numbers[0]
                # Find which book contains this worksheet
                for book_id, book_data in book_worksheets_data.get('books', {}).items():
                    if worksheet_number in book_data.get('worksheets', {}):
                        book_number = book_id
                        break
        
        # Strategy 5: Try to find any numbers and treat the largest as worksheet ID
        else:
            numbers = re.findall(r'\d+', worksheet_name_clean)
            if numbers:
                # Try the largest number as worksheet ID
                for num in sorted(numbers, key=int, reverse=True):
                    for book_id, book_data in book_worksheets_data.get('books', {}).items():
                        if num in book_data.get('worksheets', {}):
                            book_number = book_id
                            worksheet_number = num
                            break
                    if book_number:
                        break

        print(f"Parsed worksheet: Book {book_number}, Worksheet {worksheet_number}")
        
        if not book_number or not worksheet_number:
            print(f"Could not extract book/worksheet numbers from: {worksheet_name}")
            return None, None
        
        # Get correct answers from book data
        books_data = book_worksheets_data.get('books', {})
        if book_number not in books_data:
            print(f"Book {book_number} not found in worksheets data")
            return None, None
        
        book_data = books_data[book_number]
        worksheets = book_data.get('worksheets', {})
        
        if worksheet_number not in worksheets:
            print(f"Worksheet {worksheet_number} not found in book {book_number}")
            return None, None
        
        correct_answers = worksheets[worksheet_number]
        print(f"Found {len(correct_answers)} correct answers for Book {book_number}, Worksheet {worksheet_number}")
        
        return correct_answers, len(correct_answers)
        
    except Exception as e:
        print(f"Error comparing answers: {str(e)}")
        return None, None

def grade_student_answers(student_answers, correct_answers, total_questions):
    """Grade student answers against correct answers."""
    question_scores = []
    wrong_questions = []
    correct_questions = []
    unanswered_questions = []
    total_correct = 0
    
    # Determine points per question (total 40 points)
    points_per_question = 40 / total_questions if total_questions > 0 else 0
    
    for i, (question_id, question_data) in enumerate(student_answers.items()):
        question_number = i + 1
        student_answer = question_data.get('answer', '').strip()
        question_text = question_data.get('question', '')
        
        # Get correct answer if available
        correct_answer = None
        if i < len(correct_answers):
            correct_answer = str(correct_answers[i]).strip()
        
        # Determine if answer is correct
        is_correct = False
        if correct_answer and student_answer:
            # Normalize answers for comparison (remove spaces, convert to lowercase)
            normalized_student = student_answer.lower().replace(' ', '')
            normalized_correct = correct_answer.lower().replace(' ', '')
            
            # Check if answers match (exact or close match)
            if normalized_student == normalized_correct:
                is_correct = True
            # For numeric answers, try to compare as numbers
            elif normalized_student.replace('.', '').replace('-', '').isdigit() and \
                 normalized_correct.replace('.', '').replace('-', '').isdigit():
                try:
                    if float(normalized_student) == float(normalized_correct):
                        is_correct = True
                except:
                    pass
        
        points_earned = points_per_question if is_correct else 0
        if is_correct:
            total_correct += 1
        
        question_score = {
            "question_number": question_number,
            "question": question_text,
            "student_answer": student_answer,
            "correct_answer": correct_answer,
            "points_earned": round(points_earned, 2),
            "max_points": round(points_per_question, 2),
            "is_correct": is_correct
        }
        
        question_scores.append(question_score)
        
        # Categorize questions
        if not student_answer:
            unanswered_questions.append(question_score)
        elif is_correct:
            correct_questions.append(question_score)
        else:
            wrong_questions.append(question_score)
    
    overall_score = round(total_correct * points_per_question, 2)
    grade_percentage = round((total_correct / total_questions * 100), 2) if total_questions > 0 else 0
    
    # Generate encouraging feedback
    if grade_percentage >= 90:
        feedback = "Excellent work! You demonstrated a strong understanding of the concepts."
    elif grade_percentage >= 75:
        feedback = "Good job! You're doing well and showing good progress."
    elif grade_percentage >= 60:
        feedback = "Keep practicing! You're on the right track and improving."
    else:
        feedback = "Don't give up! Every mistake is a learning opportunity. Keep working hard!"
    
    return {
        "overall_score": overall_score,
        "total_possible": 40,
        "question_scores": question_scores,
        "wrong_questions": wrong_questions,
        "correct_questions": correct_questions,
        "unanswered_questions": unanswered_questions,
        "grade_percentage": grade_percentage,
        "overall_feedback": feedback,
        "total_questions": total_questions,
        "correct_answers": total_correct,
        "wrong_answers": len(wrong_questions),
        "unanswered": len(unanswered_questions)
    }

def use_gemini_for_grading_without_answers(ocr_result):
    """Use Gemini to grade student answers when correct answers are not available."""
    try:
        # Format the extracted questions and answers for Gemini grading
        questions_text = ""
        for i, (question_id, question_data) in enumerate(ocr_result.items()):
            question_number = i + 1
            question_text = question_data.get('question', '')
            student_answer = question_data.get('answer', '')
            
            questions_text += f"Question {question_number}: {question_text}\n"
            questions_text += f"Student Answer: {student_answer}\n\n"
        
        grading_prompt = f"""
        You are an expert teacher grading student worksheets. Below are the questions and student answers extracted from a worksheet.
        
        Please grade each answer and provide a score out of 40 total points (distribute points evenly among all questions).
        
        For each question, evaluate:
        1. Correctness of the answer
        2. Effort shown by the student
        3. Partial credit for partially correct answers
        
        {questions_text}
        
        IMPORTANT: Return your response in the following JSON format:
        {{
            "total_questions": <number_of_questions>,
            "overall_score": <total_score_out_of_40>,
            "grade_percentage": <percentage_score>,
            "question_scores": [
                {{
                    "question_number": 1,
                    "question": "<question_text>",
                    "student_answer": "<student_answer>",
                    "correct_answer": "<expected_answer_or_explanation>",
                    "points_earned": <points_for_this_question>,
                    "max_points": <max_points_for_this_question>,
                    "is_correct": <true_or_false>,
                    "feedback": "<brief_explanation_of_grading>"
                }}
            ],
            "correct_answers": <number_of_correct_answers>,
            "wrong_answers": <number_of_wrong_answers>,
            "unanswered": <number_of_unanswered_questions>,
            "overall_feedback": "<encouraging_feedback_for_student>"
        }}
        
        Be fair but encouraging in your grading. Give partial credit where appropriate.
        """
        
        print("Sending questions to Gemini for grading...")
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash-preview-05-20',
            contents=[grading_prompt]
        )
        
        response_text = response.text.strip()
        print("Received grading response from Gemini")
        
        # Clean up response
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        grading_result = json.loads(response_text)
        
        # Process the grading result to match our expected format
        question_scores = grading_result.get("question_scores", [])
        wrong_questions = []
        correct_questions = []
        unanswered_questions = []
        
        for score in question_scores:
            if not score.get("student_answer", "").strip():
                unanswered_questions.append(score)
            elif score.get("is_correct", False):
                correct_questions.append(score)
            else:
                wrong_questions.append(score)
        
        # Ensure we have all required fields
        final_result = {
            "overall_score": grading_result.get("overall_score", 0),
            "total_possible": 40,
            "question_scores": question_scores,
            "wrong_questions": wrong_questions,
            "correct_questions": correct_questions,
            "unanswered_questions": unanswered_questions,
            "grade_percentage": grading_result.get("grade_percentage", 0),
            "overall_feedback": grading_result.get("overall_feedback", "Keep up the good work!"),
            "total_questions": grading_result.get("total_questions", len(ocr_result)),
            "correct_answers": grading_result.get("correct_answers", len(correct_questions)),
            "wrong_answers": grading_result.get("wrong_answers", len(wrong_questions)),
            "unanswered": grading_result.get("unanswered", len(unanswered_questions)),
            "note": "Graded by Gemini AI - correct answers not available in database"
        }
        
        print(f"Gemini grading completed - Score: {final_result.get('overall_score', 0)}/40")
        return final_result
        
    except json.JSONDecodeError as e:
        print(f"JSON decode error in Gemini grading: {str(e)}")
        print(f"Raw response: {response.text}")
        return {"error": f"Failed to parse Gemini grading response as JSON: {str(e)}"}
    except Exception as e:
        print(f"Error in Gemini grading: {str(e)}")
        return {"error": f"Gemini grading error: {str(e)}"}

def use_gemini_for_direct_grading(image_bytes_list, worksheet_name):
    """Use Gemini for OCR extraction and then compare with book data for grading."""
    try:
        # Step 1: Use Gemini for OCR extraction
        print("Step 1: Extracting text using Gemini OCR...")
        ocr_result = use_gemini_for_ocr(image_bytes_list)
        
        if "error" in ocr_result:
            return ocr_result
        
        # Step 2: Load book worksheets data
        print("Step 2: Loading book worksheets data...")
        book_worksheets_data = load_book_worksheets_data()
        
        if not book_worksheets_data:
            return {"error": "Failed to load book worksheets data"}
        
        # Step 3: Compare with correct answers
        print("Step 3: Comparing with correct answers...")
        correct_answers, total_questions = compare_answers_with_book_data(
            ocr_result, worksheet_name, book_worksheets_data
        )
        
        if correct_answers is None:
            # Fallback: if we can't find the worksheet, use Gemini for grading
            print("Worksheet not found in database, using Gemini for grading...")
            return use_gemini_for_grading_without_answers(ocr_result)
        
        # Step 4: Grade the answers
        print("Step 4: Grading the answers...")
        grading_result = grade_student_answers(ocr_result, correct_answers, total_questions)
        
        print(f"Grading completed - Score: {grading_result.get('overall_score', 0)}/40")
        return grading_result
        
    except Exception as e:
        print(f"Error in grading process: {str(e)}")
        return {"error": f"Grading process error: {str(e)}"}

def save_results_to_mongo(token_no, worksheet_name, grading_result, s3_url, filename):
    try:
        s3_urls = s3_url.split(';')
        filenames = filename.split(', ')
        
        document = {
            "token_no": token_no,
            "worksheet_name": worksheet_name,
            "filename": filename,
            "s3_url": s3_url,
            "s3_urls": s3_urls,
            "image_count": len(s3_urls),
            "filenames": filenames,
            "overall_score": grading_result.get("overall_score", 0),
            "total_possible": 40,
            "grade_percentage": grading_result.get("grade_percentage", 0),
            "question_scores": grading_result.get("question_scores", []),
            "wrong_questions": grading_result.get("wrong_questions", []),
            "correct_questions": grading_result.get("correct_questions", []),
            "unanswered_questions": grading_result.get("unanswered_questions", []),
            "overall_feedback": grading_result.get("overall_feedback", ""),
            "total_questions": grading_result.get("total_questions", 0),
            "correct_answers": grading_result.get("correct_answers", 0),
            "wrong_answers": grading_result.get("wrong_answers", 0),
            "unanswered": grading_result.get("unanswered", 0),
            "grading_method": "gemini-ai-grading" if "Graded by Gemini AI" in grading_result.get("note", "") else "gemini-ocr-with-book-comparison",
            "has_answer_key": "note" not in grading_result or "Graded by Gemini AI" not in grading_result.get("note", ""),  # True if we found answers in book_worksheets.json
            "timestamp": (datetime.utcnow() + timedelta(hours=5, minutes=30)).isoformat(),
            "processed_with": "gemini-2.5-flash-ai-grading" if "Graded by Gemini AI" in grading_result.get("note", "") else "gemini-2.5-flash-ocr-plus-book-comparison"
        }
        
        result = collection.insert_one(document)
        print(f"Saved to mongo: {result.inserted_id}")
        return str(result.inserted_id)
        
    except Exception as e:
        print(f"MongoDB save error: {str(e)}")
        return None

def extract_entries_from_response(response_data):
    entries = []
    
    if isinstance(response_data, dict):
        for key, value in response_data.items():
            if (not key.startswith('_') and 
                isinstance(value, dict) and 
                'question' in value and 
                'answer' in value):
                
                entries.append({
                    'question_id': key,
                    'question': value['question'],
                    'answer': value['answer']
                })
    
    print(f"Extracted {len(entries)} entries from OCR response")
    return entries

def deduplicate_student_entries(all_entries):
    question_groups = {}
    for entry in all_entries:
        norm_question = ' '.join(entry['question'].lower().strip().split())
        
        if norm_question not in question_groups:
            question_groups[norm_question] = []
        
        question_groups[norm_question].append(entry)
    
    deduplicated = []
    for _, entries in question_groups.items():
        if len(entries) == 1:
            deduplicated.append(entries[0])
        else:
            best_entry = entries[0]
            for entry in entries[1:]:
                if not best_entry['answer'].strip() and entry['answer'].strip():
                    best_entry = entry
                elif (best_entry['answer'].strip() and entry['answer'].strip() and 
                      len(entry['answer']) > len(best_entry['answer'])):
                    best_entry = entry
            
            deduplicated.append(best_entry)
    
    return deduplicated

