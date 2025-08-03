import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from conns import s3_client, gemini_client, collection
from google.genai import types
from datetime import datetime, timedelta
from PIL import Image
import io
from pydantic import BaseModel, Field
from functools import lru_cache
import concurrent.futures
from pathlib import Path

S3_BUCKET_NAME = "learno-pdf-document"

class ExtractedQuestion(BaseModel):
    question_number: int = Field(description="The unique identifier for the question")
    question: str = Field(description="The entire text of the question without the student's answer. Do not confuse questions on different columns.")
    student_answer: str = Field(description="The student's answer to the question - the exact answer as written by the student and not necessarily the correct answer")

class ExtractedQuestions(BaseModel):
    questions: List[ExtractedQuestion] = Field(description="The list of questions and their corresponding student answers")

def upload_to_s3(file_path: str) -> Optional[str]:
    """Upload file to S3 with optimized buffer size"""
    try:
        file_name = Path(file_path).name
        s3_key = f"worksheets-{file_name}"
        
        # Use larger buffer for better I/O performance
        with open(file_path, 'rb', buffering=8192) as file_data:
            s3_client.upload_fileobj(
                file_data, 
                S3_BUCKET_NAME, 
                s3_key,
                ExtraArgs={'ACL': 'public-read'}
            )
        
        return f"https://{S3_BUCKET_NAME}.s3.ap-south-1.amazonaws.com/{s3_key}"
    except Exception as e:
        print(f"Error uploading to S3: {str(e)}")
        return None

def _process_image_for_ocr(image_bytes: bytes) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as image:
        # Skip conversion if already RGB
        if image.mode == 'RGB':
            return image_bytes
        
        # Convert and optimize
        rgb_image = image.convert('RGB')
        img_buffer = io.BytesIO()
        # Use optimize flag for smaller file size
        rgb_image.save(img_buffer, format='JPEG', quality=95, optimize=True)
        return img_buffer.getvalue()

def use_gemini_for_ocr(image_bytes_list: List[bytes], worksheet_name: str = None) -> Dict[str, Any]:
    try:
        if not isinstance(image_bytes_list, list):
            image_bytes_list = [image_bytes_list]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            processed_images = list(executor.map(_process_image_for_ocr, image_bytes_list))
        
        ocr_prompt = None
        if worksheet_name:
            worksheet_number = None
            if worksheet_name.strip().isdigit():
                worksheet_number = worksheet_name.strip()
            else:
                import re
                numbers = re.findall(r'\d+', worksheet_name)
                if numbers:
                    worksheet_number = numbers[-1]
            
            if worksheet_number:
                prompt_file_path = Path(__file__).parent / 'context' / 'prompts' / f'{worksheet_number}.txt'
                if prompt_file_path.exists():
                    try:
                        with prompt_file_path.open('r', encoding='utf-8') as f:
                            ocr_prompt = f.read()
                        print(f"Using custom OCR prompt for worksheet {worksheet_number}")
                    except Exception as e:
                        print(f"Error loading custom prompt for worksheet {worksheet_number}: {str(e)}")
        
        if not ocr_prompt:
            ocr_prompt = """Extract all questions and their corresponding student answers from these worksheet images. 

<Rules>
1. When giving the student's answer, give exactly what they wrote. DO NOT INTERPRET. Remember that the student's future depends on the correctness of your interpretation.
2. If a question is unanswered, use an empty string "" for the answer.
3. There may be two reasonable interpretations for a student's answer. For instance, students may write something which could be interpreted as 1 or 7. Give the one which is most likely.
4. There are a total of 40 questions in the worksheet.
5. Return the questions in the order of question number.
6. Some sheets have multiple columns of questions. Don't confuse questions on different columns.
7. Include the entire question text in the question field - do not miss anything. Do not include the student's answer in the question field.
</Rules>   

Respond in the following JSON format:
{format_instructions}

"""
        
        contents = [ocr_prompt]
        for image_data in processed_images:
            contents.append(types.Part.from_bytes(data=image_data, mime_type='image/jpeg'))
            
        print(f"Sending {len(processed_images)} images to Gemini OCR in a single request...")
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                response_schema=ExtractedQuestions,
                temperature=0.0
            )
            
        )
        
        result = response.parsed
        print(result)
        return result
        
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {str(e)}")
        print(f"Raw response: {response.text}")
        return {"error": f"Failed to parse OCR response as JSON: {str(e)}"}
    except Exception as e:
        print(f"Gemini OCR error: {str(e)}")
        return {"error": f"Gemini OCR error: {str(e)}"}

@lru_cache(maxsize=1)
def load_book_worksheets_data() -> Optional[Dict[str, Any]]:
    try:
        book_worksheets_path = Path(__file__).parent / 'Results' / 'book_worksheets.json'
        with book_worksheets_path.open('r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading book worksheets data: {str(e)}")
        return None

_WORKSHEET_NUMBER_PATTERN = re.compile(r'\d+')
_BOOK_WORKSHEET_PATTERN = re.compile(r'^\d+[-_]\d+$')

def compare_answers_with_book_data(student_answers: ExtractedQuestions, worksheet_name: str, book_worksheets_data: Dict[str, Any]) -> Tuple[Optional[List], Optional[int]]:
    try:
        book_number = None
        worksheet_number = None
        worksheet_name_clean = worksheet_name.strip()
        
        if worksheet_name_clean.isdigit():
            worksheet_number = worksheet_name_clean
            for book_id, book_data in book_worksheets_data.get('books', {}).items():
                if worksheet_number in book_data.get('worksheets', {}):
                    book_number = book_id
                    break
        elif 'book' in worksheet_name_clean.lower() and 'worksheet' in worksheet_name_clean.lower():
            numbers = _WORKSHEET_NUMBER_PATTERN.findall(worksheet_name_clean)
            if len(numbers) >= 2:
                book_number = numbers[0]
                worksheet_number = numbers[1]
            elif len(numbers) == 1:
                worksheet_number = numbers[0]
                for book_id, book_data in book_worksheets_data.get('books', {}).items():
                    if worksheet_number in book_data.get('worksheets', {}):
                        book_number = book_id
                        break
        elif _BOOK_WORKSHEET_PATTERN.match(worksheet_name_clean):
            numbers = _WORKSHEET_NUMBER_PATTERN.findall(worksheet_name_clean)
            if len(numbers) == 2:
                book_number = numbers[0]
                worksheet_number = numbers[1]
        elif 'worksheet' in worksheet_name_clean.lower():
            numbers = _WORKSHEET_NUMBER_PATTERN.findall(worksheet_name_clean)
            if numbers:
                worksheet_number = numbers[0]
                for book_id, book_data in book_worksheets_data.get('books', {}).items():
                    if worksheet_number in book_data.get('worksheets', {}):
                        book_number = book_id
                        break
        else:
            numbers = _WORKSHEET_NUMBER_PATTERN.findall(worksheet_name_clean)
            if numbers:
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

def grade_student_answers(student_answers: ExtractedQuestions, correct_answers: List[str], total_questions: int) -> Dict[str, Any]:
    question_scores = []
    wrong_questions = []
    correct_questions = []
    unanswered_questions = []
    total_correct = 0
    
    points_per_question = 40 / total_questions if total_questions > 0 else 0
    
    for i, question in enumerate(student_answers.questions):
        question_number = question.question_number
        student_answer = question.student_answer
        question_text = question.question
        
        correct_answer = None
        if i < len(correct_answers):
            correct_answer = str(correct_answers[i]).strip()
        
        # Optimized comparison logic
        is_correct = False
        if correct_answer and student_answer:
            # Normalize once
            normalized_student = student_answer.lower().replace(' ', '')
            normalized_correct = correct_answer.lower().replace(' ', '')
            
            # Quick string comparison first
            if normalized_student == normalized_correct:
                is_correct = True
            else:
                # Check numeric only if needed
                student_numeric = normalized_student.replace('.', '').replace('-', '')
                correct_numeric = normalized_correct.replace('.', '').replace('-', '')
                
                if student_numeric.isdigit() and correct_numeric.isdigit():
                    try:
                        is_correct = float(normalized_student) == float(normalized_correct)
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
        
        if not student_answer:
            unanswered_questions.append(question_score)
        elif is_correct:
            correct_questions.append(question_score)
        else:
            wrong_questions.append(question_score)
    
    overall_score = round(total_correct * points_per_question, 2)
    grade_percentage = round((total_correct / total_questions * 100), 2) if total_questions > 0 else 0
    
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

def use_gemini_for_grading_without_answers(ocr_result: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # Build questions text more efficiently
        questions_parts = []
        for i, (question_id, question_data) in enumerate(ocr_result.items()):
            question_number = i + 1
            question_text = question_data.get('question', '')
            student_answer = question_data.get('answer', '')
            
            questions_parts.append(f"Question {question_number}: {question_text}\nStudent Answer: {student_answer}\n")
        
        questions_text = '\n'.join(questions_parts)
        
        grading_prompt = f"""You are an expert teacher grading student worksheets. Below are the questions and student answers extracted from a worksheet.
        
Please grade each answer and provide a score out of 40 total points (distribute points evenly among all questions).
        
For each question, evaluate:
1. Correctness of the answer
2. Effort shown by the student
        
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

     
        """
        print("Sending questions to Gemini for grading...")
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[grading_prompt]
        )
        
        response_text = response.text.strip()
        print("Received grading response from Gemini")
        
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        grading_result = json.loads(response_text)
        
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

def use_gemini_for_direct_grading(image_bytes_list: List[bytes], worksheet_name: str) -> Dict[str, Any]:
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            ocr_future = executor.submit(use_gemini_for_ocr, image_bytes_list, worksheet_name)
            book_data_future = executor.submit(load_book_worksheets_data)
            
            ocr_result = ocr_future.result()
            
            if "error" in ocr_result:
                return ocr_result
            
            book_worksheets_data = book_data_future.result()
            
            if not book_worksheets_data:
                return {"error": "Failed to load book worksheets data"}
        
        correct_answers, total_questions = compare_answers_with_book_data(
            ocr_result, worksheet_name, book_worksheets_data
        )
        
        if correct_answers is None:
            return use_gemini_for_grading_without_answers(ocr_result)
        
        grading_result = grade_student_answers(ocr_result, correct_answers, total_questions)
        
        print(f"Grading completed - Score: {grading_result.get('overall_score', 0)}/40")
        return grading_result
        
    except Exception as e:
        print(f"Error in grading process: {str(e)}")
        return {"error": f"Grading process error: {str(e)}"}

def save_results_to_mongo(token_no: str, worksheet_name: str, grading_result: Dict[str, Any], s3_url: str, filename: str) -> Optional[str]:
    try:
        s3_urls = s3_url.split(';')
        filenames = filename.split(', ')
        is_ai_graded = "Graded by Gemini AI" in grading_result.get("note", "")
        
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
            "grading_method": "gemini-ai-grading" if is_ai_graded else "gemini-ocr-with-book-comparison",
            "has_answer_key": not is_ai_graded,
            "timestamp": (datetime.utcnow() + timedelta(hours=5, minutes=30)).isoformat(),
            "processed_with": "gemini-2.5-flash-ai-grading" if is_ai_graded else "gemini-2.5-flash-ocr-plus-book-comparison"
        }
        
        result = collection.insert_one(document)
        print(f"Saved to mongo: {result.inserted_id}")
        return str(result.inserted_id)
        
    except Exception as e:
        print(f"MongoDB save error: {str(e)}")
        return None

def extract_entries_from_response(response_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(response_data, dict):
        return []
    
    entries = [
        {
            'question_id': key,
            'question': value['question'],
            'answer': value['answer']
        }
        for key, value in response_data.items()
        if (not key.startswith('_') and 
            isinstance(value, dict) and 
            'question' in value and 
            'answer' in value)
    ]
    
    print(f"Extracted {len(entries)} entries from OCR response")
    return entries

def deduplicate_student_entries(all_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from collections import defaultdict
    
    question_groups = defaultdict(list)
    
    for entry in all_entries:
        norm_question = ' '.join(entry['question'].lower().strip().split())
        question_groups[norm_question].append(entry)
    
    deduplicated = []
    for entries in question_groups.values():
        if len(entries) == 1:
            deduplicated.append(entries[0])
        else:
            # Select best entry based on answer content
            best_entry = max(entries, key=lambda e: (
                bool(e['answer'].strip()),  # Prefer answered
                len(e['answer'])  # Then prefer longer answers
            ))
            deduplicated.append(best_entry)
    
    return deduplicated