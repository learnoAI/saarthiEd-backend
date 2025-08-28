import json
import re
from typing import List, Dict, Any, Optional
from conns import s3_client, gemini_client, collection, openai_client
from google.genai import types
from datetime import datetime, timedelta
from PIL import Image
import io
import concurrent.futures
from pathlib import Path
import base64
from schema import ExtractedQuestions, GradingResult

# Constants
S3_BUCKET_NAME = "learno-pdf-document"
S3_REGION = "ap-south-1"
PNG_QUALITY = 100
MAX_WORKER_THREADS = 4
TOTAL_POSSIBLE_POINTS = 40

def load_book_worksheets_answers() -> Dict[str, Any]:
    """Load the book worksheets answers from the JSON file."""
    try:
        book_worksheets_path = Path(__file__).parent / 'Results' / 'book_worksheets.json'
        with open(book_worksheets_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading book worksheets: {str(e)}")
        return {}

def find_worksheet_answers(worksheet_name: str, book_worksheets_data: Dict[str, Any]) -> Optional[List[str]]:
    """Find answers for a specific worksheet in the book data."""
    try:
        # Extract worksheet number from the name
        worksheet_number = None
        if worksheet_name.strip().isdigit():
            worksheet_number = worksheet_name.strip()
        else:
            number_matches = re.findall(r'\d+', worksheet_name)
            if number_matches:
                worksheet_number = number_matches[-1]
        
        if not worksheet_number:
            return None
        
        # Search through all books for this worksheet
        books = book_worksheets_data.get('books', {})
        for book_id, book_data in books.items():
            worksheets = book_data.get('worksheets', {})
            if worksheet_number in worksheets:
                answers = worksheets[worksheet_number]
                if answers:  # Only return if there are actual answers
                    print(f"Found answers for worksheet {worksheet_number} in book {book_id}: {len(answers)} answers")
                    return answers
        
        print(f"No answers found for worksheet {worksheet_number}")
        return None
        
    except Exception as e:
        print(f"Error finding worksheet answers: {str(e)}")
        return None

def upload_file_to_s3(file_path: str) -> Optional[str]:
    try:
        filename = Path(file_path).name
        s3_object_key = f"worksheets-{filename}"
        
        with open(file_path, 'rb', buffering=8192) as file_stream:
            s3_client.upload_fileobj(
                file_stream, 
                S3_BUCKET_NAME, 
                s3_object_key,
                ExtraArgs={'ACL': 'public-read'}
            )
        
        return f"https://{S3_BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/{s3_object_key}"
    except Exception as upload_error:
        print(f"Error uploading to S3: {str(upload_error)}")
        return None

def _convert_image_to_rgb(image_bytes: bytes) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as image:
        if image.mode == 'RGB':
            return image_bytes
        
        rgb_image = image.convert('RGB')
        image_buffer = io.BytesIO()
        rgb_image.save(image_buffer, format='PNG', quality=PNG_QUALITY, optimize=True, subsampling=0)
        return image_buffer.getvalue()

def extract_questions_with_gemini_ocr(image_bytes_list: List[bytes], worksheet_name: str = None) -> Dict[str, Any]:
    try:
        if not isinstance(image_bytes_list, list):
            image_bytes_list = [image_bytes_list]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKER_THREADS) as executor:
            processed_images = list(executor.map(_convert_image_to_rgb, image_bytes_list))
        
        custom_ocr_prompt = None
        if worksheet_name:
            extracted_worksheet_number = None
            if worksheet_name.strip().isdigit():
                extracted_worksheet_number = worksheet_name.strip()
            else:
                number_matches = re.findall(r'\d+', worksheet_name)
                if number_matches:
                    extracted_worksheet_number = number_matches[-1]
            
            if extracted_worksheet_number:
                custom_prompt_file_path = Path(__file__).parent / 'context' / 'prompts' / f'{extracted_worksheet_number}.txt'
                if custom_prompt_file_path.exists():
                    try:
                        with custom_prompt_file_path.open('r', encoding='utf-8') as prompt_file:
                            custom_ocr_prompt = prompt_file.read()
                        print(f"Using custom OCR prompt for worksheet {extracted_worksheet_number}")
                    except Exception as prompt_error:
                        print(f"Error loading custom prompt for worksheet {extracted_worksheet_number}: {str(prompt_error)}")
        
        if not custom_ocr_prompt:
            custom_ocr_prompt = """Extract all questions and their corresponding student answers from these worksheet images. 

            <Rules>
            1. When giving the student's answer, give exactly what they wrote. DO NOT INTERPRET. Remember that the student's future depends on the correctness of your interpretation.
            2. If a question is unanswered, use an empty string "" for the answer.
            3. There may be two reasonable interpretations for a student's answer. For instance, students may write something which could be interpreted as 1 or 7. GIVE THE ONE WHICH IS MORE LIKELY.
            4. There can be a total of 40 questions in the worksheet (NOT ALWAYS CAN BE ANYWHERE BETWEEN 8-40 QUESTIONS)
            5. Return the questions in the order of question number.
            6. Some sheets have multiple columns of questions. THESE ARE INDIVIDUAL QUESTIONS WHERE STUDENT HAS TO FILL IN THE ANSWERS. EXTRACT THEM PROPERLY
            7. Include the entire question text in the question field - do not miss anything. Do not include the student's answer in the question field.
            8. CAREFULLY SEE WHICH IS THE STUDENT's ANSWER AS THEY ARE USING A PENCIL TO MARK THEIR ANSWERS.
            </Rules>

            """

        # Use Gemini for OCR
        gemini_content_parts = [custom_ocr_prompt]
        for processed_image_data in processed_images:
            gemini_content_parts.append(types.Part.from_bytes(data=processed_image_data, mime_type='image/png'))
            
        print(f"Sending {len(processed_images)} images to Gemini OCR in a single request...")
        gemini_response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=gemini_content_parts,
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                response_schema=ExtractedQuestions,
                temperature=0.1,
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        
        extraction_result = gemini_response.parsed
        print(extraction_result)
        return extraction_result

        # Alternative OpenAI implementation
        # openai_content_parts = [{"type": "input_text", "text": custom_ocr_prompt}]
        
        # for processed_image_data in processed_images:
        #     base64_encoded_image = base64.b64encode(processed_image_data).decode('utf-8')
        #     openai_content_parts.append({
        #         "type": "input_image",
        #         "image_url": f"data:image/png;base64,{base64_encoded_image}"
        #     })
        
        # print(f"Sending {len(processed_images)} images to OpenAI OCR in a single request...")
        # openai_response = openai_client.responses.parse(
        #     model="gpt-5-nano",
        #     input=[
        #         {
        #             "role": "user",
        #             "content": openai_content_parts
        #         }
        #     ],
        #     text_format=ExtractedQuestions,
        #     text={
        #         "verbosity": "high"
        #     }
        # )

        # extraction_result = openai_response.output_parsed
        # print(extraction_result)
        # return extraction_result
        
    except json.JSONDecodeError as json_error:
        print(f"JSON decode error: {str(json_error)}")
        return {"error": f"Failed to parse OCR response as JSON: {str(json_error)}"}
    except Exception as ocr_error:
        print(f"OCR error: {str(ocr_error)}")
        return {"error": f"OCR error: {str(ocr_error)}"}

def grade_questions_with_gemini_ai(extracted_questions: ExtractedQuestions) -> Dict[str, Any]:
    try:
        formatted_question_parts = []
        for question in extracted_questions.questions:
            question_num = question.question_number
            question_content = question.question
            student_response = question.student_answer
            
            formatted_question_parts.append(f"Question {question_num}: {question_content}\nStudent Answer: {student_response}\n")
        
        formatted_questions_text = '\n'.join(formatted_question_parts)
        
        ai_grading_prompt = f"""You are an expert teacher grading student worksheets. Below are the questions and student answers extracted from a worksheet.
                    
            Please grade each answer and provide a score out of {TOTAL_POSSIBLE_POINTS} total points (distribute points evenly among all questions).
            <Rules>
                1. NO PARTIAL GRADING OF A QUESTION
                2. If a question is unanswered, it should receive 0 points.
                3. If a question is answered incorrectly, it should receive no points.
                4. There can be a total of {TOTAL_POSSIBLE_POINTS} questions in the worksheet (NOT ALWAYS CAN BE ANYWHERE BETWEEN 8-40 QUESTIONS).
                5. NO GRADES IN DECIMALS
            </Rules>
            For each question, evaluate:
            1. Correctness of the answer
            2. Effort shown by the student
                    
            {formatted_questions_text}
                    
            IMPORTANT: Return your response in the following JSON format:
            {{
                "total_questions": <number_of_questions>,
                "overall_score": <total_score_out_of_{TOTAL_POSSIBLE_POINTS}>,
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
                "overall_feedback": "<encouraging_feedback_for_student_in_1_line>"
            }}

        """

        print("Sending questions to Gemini for grading...")
        grading_response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[ai_grading_prompt],
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                response_schema=GradingResult,
                temperature=0.1,
                # thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        grading_response_text = grading_response.text

        # print('sending questions to openai for grading')
        # grading_response = openai_client.responses.parse(
        #     model="gpt-5-nano",
        #     input=ai_grading_prompt,
        #     text_format=GradingResult,
        #     text={
        #         "verbosity": "high"
        #     },
        # )
        # grading_response_text = grading_response.output_text

        parsed_grading_result = json.loads(grading_response_text)
        print(parsed_grading_result)
        individual_question_scores = parsed_grading_result.get("question_scores", [])
        incorrect_questions = []
        correct_questions = []
        blank_questions = []
        
        for question_score in individual_question_scores:
            if not question_score.get("student_answer", "").strip():
                blank_questions.append(question_score)
            elif question_score.get("is_correct", False):
                correct_questions.append(question_score)
            else:
                incorrect_questions.append(question_score)
        
        comprehensive_grading_result = {
            "overall_score": parsed_grading_result.get("overall_score", 0),
            "total_possible": TOTAL_POSSIBLE_POINTS,
            "question_scores": individual_question_scores,
            "wrong_questions": incorrect_questions,
            "correct_questions": correct_questions,
            "unanswered_questions": blank_questions,
            "grade_percentage": parsed_grading_result.get("grade_percentage", 0),
            "overall_feedback": parsed_grading_result.get("overall_feedback", "Keep up the good work!"),
            "total_questions": parsed_grading_result.get("total_questions", len(extracted_questions.questions)),
            "correct_answers": parsed_grading_result.get("correct_answers", len(correct_questions)),
            "wrong_answers": parsed_grading_result.get("wrong_answers", len(incorrect_questions)),
            "unanswered": parsed_grading_result.get("unanswered", len(blank_questions)),
            "note": "Graded by Gemini AI - correct answers not available in database",
            "reason_why": parsed_grading_result.get("reason_why", "No specific reason provided")
        }
        
        return comprehensive_grading_result
        
    except json.JSONDecodeError as json_decode_error:
        print(f"JSON decode error in Gemini grading: {str(json_decode_error)}")
        return {"error": f"Failed to parse Gemini grading response as JSON: {str(json_decode_error)}"}
    except Exception as grading_error:
        print(f"Error in Gemini grading: {str(grading_error)}")
        return {"error": f"Gemini grading error: {str(grading_error)}"}

def grade_questions_with_book_answers(extracted_questions: ExtractedQuestions, book_answers: List[str]) -> Dict[str, Any]:
    try:
        formatted_question_parts = []
        for i, question in enumerate(extracted_questions.questions):
            question_num = question.question_number
            question_content = question.question
            student_response = question.student_answer
            
            correct_answer = book_answers[i] if i < len(book_answers) else "Answer not available"
            
            formatted_question_parts.append(
                f"Question {question_num}: {question_content}\n"
                f"Student Answer: {student_response}\n"
                f"Correct Answer: {correct_answer}\n"
            )
        
        formatted_questions_text = '\n'.join(formatted_question_parts)
        
        ai_grading_prompt = f"""You are an expert teacher grading student worksheets. Below are the questions, student answers, and correct answers from the answer key.
                    
            Please grade each answer and provide a score out of {TOTAL_POSSIBLE_POINTS} total points (distribute points evenly among all questions).
            <Rules>
                1. NO PARTIAL GRADING OF A QUESTION
                2. If a question is unanswered, it should receive 0 points.
                3. If a question is answered incorrectly, it should receive no points.
                4. There can be a total of {TOTAL_POSSIBLE_POINTS} questions in the worksheet (NOT ALWAYS CAN BE ANYWHERE BETWEEN 8-40 QUESTIONS).
                5. NO GRADES IN DECIMALS
                6. Compare the student answer with the provided correct answer exactly.
                7. Consider minor variations in formatting or spacing, but the core answer must match.
            </Rules>
            
            For each question, evaluate:
            1. Exact match with the correct answer
            2. Accept reasonable variations (e.g., "20" vs "20.0" or slight spacing differences)
                    
            {formatted_questions_text}
                    
            IMPORTANT: Return your response in the following JSON format:
            {{
                "total_questions": <number_of_questions>,
                "overall_score": <total_score_out_of_{TOTAL_POSSIBLE_POINTS}>,
                "grade_percentage": <percentage_score>,
                "question_scores": [
                    {{
                        "question_number": 1,
                        "question": "<question_text>",
                        "student_answer": "<student_answer>",
                        "correct_answer": "<correct_answer_from_book>",
                        "points_earned": <points_for_this_question>,
                        "max_points": <max_points_for_this_question>,
                        "is_correct": <true_or_false>,
                        "feedback": "<brief_explanation_of_grading>"
                    }}
                ],
                "correct_answers": <number_of_correct_answers>,
                "wrong_answers": <number_of_wrong_answers>,
                "unanswered": <number_of_unanswered_questions>,
                "overall_feedback": "<encouraging_feedback_for_student_in_1_line>"
            }}

        """

        print("Sending questions to Gemini for grading with book answers...")
        grading_response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[ai_grading_prompt],
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                response_schema=GradingResult,
                temperature=0.1,
                # thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        grading_response_text = grading_response.text

        parsed_grading_result = json.loads(grading_response_text)
        print(parsed_grading_result)
        individual_question_scores = parsed_grading_result.get("question_scores", [])
        incorrect_questions = []
        correct_questions = []
        blank_questions = []
        
        for question_score in individual_question_scores:
            if not question_score.get("student_answer", "").strip():
                blank_questions.append(question_score)
            elif question_score.get("is_correct", False):
                correct_questions.append(question_score)
            else:
                incorrect_questions.append(question_score)
        
        comprehensive_grading_result = {
            "overall_score": parsed_grading_result.get("overall_score", 0),
            "total_possible": TOTAL_POSSIBLE_POINTS,
            "question_scores": individual_question_scores,
            "wrong_questions": incorrect_questions,
            "correct_questions": correct_questions,
            "unanswered_questions": blank_questions,
            "grade_percentage": parsed_grading_result.get("grade_percentage", 0),
            "overall_feedback": parsed_grading_result.get("overall_feedback", "Keep up the good work!"),
            "total_questions": parsed_grading_result.get("total_questions", len(extracted_questions.questions)),
            "correct_answers": parsed_grading_result.get("correct_answers", len(correct_questions)),
            "wrong_answers": parsed_grading_result.get("wrong_answers", len(incorrect_questions)),
            "unanswered": parsed_grading_result.get("unanswered", len(blank_questions)),
            "note": "Graded with book answer key",
            "reason_why": "Answer key available in book worksheets database"
        }
        
        return comprehensive_grading_result
        
    except json.JSONDecodeError as json_decode_error:
        print(f"JSON decode error in book answer grading: {str(json_decode_error)}")
        return {"error": f"Failed to parse book answer grading response as JSON: {str(json_decode_error)}"}
    except Exception as grading_error:
        print(f"Error in book answer grading: {str(grading_error)}")
        return {"error": f"Book answer grading error: {str(grading_error)}"}

def process_worksheet_with_gemini_direct_grading(image_bytes_list: List[bytes], worksheet_name: str) -> Dict[str, Any]:
    try:
        extraction_result = extract_questions_with_gemini_ocr(image_bytes_list, worksheet_name)
        
        if "error" in extraction_result:
            return extraction_result
        
        book_worksheets_data = load_book_worksheets_answers()
        
        book_answers = find_worksheet_answers(worksheet_name, book_worksheets_data)
        
        if book_answers:
            print(f"Using book answers for grading worksheet {worksheet_name}")
            comprehensive_grading_result = grade_questions_with_book_answers(extraction_result, book_answers)
        else:
            print(f"No book answers found for worksheet {worksheet_name}, using direct AI grading")
            comprehensive_grading_result = grade_questions_with_gemini_ai(extraction_result)
        
        print(f"Grading completed - Score: {comprehensive_grading_result.get('overall_score', 0)}/{TOTAL_POSSIBLE_POINTS}")
        return comprehensive_grading_result
        
    except Exception as processing_error:
        print(f"Error in grading process: {str(processing_error)}")
        return {"error": f"Grading process error: {str(processing_error)}"}

def save_worksheet_results_to_mongodb(student_token_number: str, worksheet_identifier: str, grading_results: Dict[str, Any], s3_file_url: str, original_filename: str) -> Optional[str]:
    try:
        parsed_s3_urls = s3_file_url.split(';')
        parsed_filenames = original_filename.split(', ')
        
        grading_note = grading_results.get("note", "")
        is_book_answer_graded = "book answer key" in grading_note.lower()
        is_ai_graded_worksheet = "Graded by Gemini AI" in grading_note
        
        if is_book_answer_graded:
            grading_method = "gemini-with-book-answers"
            processed_with = "gemini-2.5-flash-with-book-answers"
            has_answer_key = True
        elif is_ai_graded_worksheet:
            grading_method = "gemini-ai-grading"
            processed_with = "gemini-2.5-flash"
            has_answer_key = False
        else:
            grading_method = "gemini-ocr-with-book-comparison"
            processed_with = "gemini-2.5-flash-plus-book-comparison"
            has_answer_key = True
        
        mongodb_document = {
            "token_no": student_token_number,
            "worksheet_name": worksheet_identifier,
            "filename": original_filename,
            "s3_urls": parsed_s3_urls,
            "image_count": len(parsed_s3_urls),
            "filenames": parsed_filenames,
            "overall_score": grading_results.get("overall_score", 0),
            "total_possible": TOTAL_POSSIBLE_POINTS,
            "grade_percentage": grading_results.get("grade_percentage", 0),
            "question_scores": grading_results.get("question_scores", []),
            "wrong_questions": grading_results.get("wrong_questions", []),
            "correct_questions": grading_results.get("correct_questions", []),
            "unanswered_questions": grading_results.get("unanswered_questions", []),
            "overall_feedback": grading_results.get("overall_feedback", ""),
            "total_questions": grading_results.get("total_questions", 0),
            "correct_answers": grading_results.get("correct_answers", 0),
            "wrong_answers": grading_results.get("wrong_answers", 0),
            "unanswered": grading_results.get("unanswered", 0),
            "grading_method": grading_method,
            "has_answer_key": has_answer_key,
            "timestamp": (datetime.utcnow() + timedelta(hours=5, minutes=30)).isoformat(),
            "processed_with": processed_with,
            "reason_why": grading_results.get("reason_why", "")
        }
        
        insertion_result = collection.insert_one(mongodb_document)
        print(f"Saved to MongoDB: {insertion_result.inserted_id}")
        return str(insertion_result.inserted_id)
        
    except Exception as mongodb_save_error:
        print(f"MongoDB save error: {str(mongodb_save_error)}")
        return None
