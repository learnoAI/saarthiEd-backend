import os
import json
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

def use_gemini_for_scoring(student_entries, expected_answers, worksheet_name):
    try:
        student_data = []
        for entry in student_entries:
            student_data.append({
                "question_id": entry.get("question_id", ""),
                "question": entry.get("question", ""),
                "student_answer": entry.get("answer", "").strip()
            })
        
        expected_data = []
        for i, expected in enumerate(expected_answers):
            expected_data.append({
                "question_number": i + 1,
                "question": expected.get("question", ""),
                "expected_answer": expected.get("answer", "")
            })
        multi_image_info = "The student entries below may have been extracted from multiple images of the same worksheet."
        
        scoring_prompt = f"""
                You are an expert teacher grading student worksheets. Compare each student answer with the expected answer and provide detailed scoring.

                Worksheet: {worksheet_name}

                {multi_image_info}

                SCORING GUIDELINES:
                1. Exact matches get 1 point
                2. Mathematically equivalent answers get 1 point (e.g., 0.5 = 1/2, 2/4 = 0.5)
                3. Minor spelling/formatting differences should be handled gracefully
                4. Case-insensitive matching for text answers
                5. Completely wrong answers get 0 points
                6. Empty/unanswered questions get 0 points
                7. Deduplicate any repeated questions from multiple images - use the best answer if a student answered the same question multiple times

                Student Data:
                {json.dumps(student_data, indent=2)}

                Expected Answers:
                {json.dumps(expected_data, indent=2)}

                Return a JSON response with this EXACT structure:
                {{
                "overall_score": <total_points_earned>,
                "total_possible": {len(expected_answers)},
                "question_scores": [
                    {{
                    "question_number": 1,
                    "student_answer": "<student's answer>",
                    "expected_answer": "<expected answer>",
                    "points_earned": <0 or 1>,
                    "is_correct": true/false,
                    }}
                ]
                }}
                """
        
        response = gemini_client.models.generate_content(
            model='gemini-2.0-flash',
            contents=scoring_prompt
        )
        response_text = response.text.strip()
        
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        result = json.loads(response_text)
        print(f"Scoring Completed")
        return result
        
    except json.JSONDecodeError as e:
        print(f"Scoring JSON decode error: {str(e)}")
        print(f"Raw scoring response: {response.text}")
        return {"error": f"Failed to parse scoring response as JSON: {str(e)}"}
    except Exception as e:
        print(f"Gemini scoring error: {str(e)}")
        return {"error": f"Gemini scoring error: {str(e)}"}

def save_results_to_mongo(token_no, worksheet_name, scoring_result, s3_url, filename):
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
            "overall_score": scoring_result.get("overall_score", 0),
            "total_possible": scoring_result.get("total_possible", 40),
            "question_scores": scoring_result.get("question_scores", []),
            "timestamp": (datetime.utcnow()+ timedelta(hours=5, minutes=30)).isoformat(),
            "processed_with": "gemini-2.5-flash"
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

