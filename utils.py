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

def use_gemini_for_direct_grading(image_bytes_list, worksheet_name):
    import io
    import json
    from PIL import Image
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

        grading_prompt = f"""
            You are a lenient and encouraging teacher grading student worksheets. Analyze the worksheet images and provide fair grading.

            Worksheet: {worksheet_name}

            INSTRUCTIONS:
            1. Extract all questions and student answers from the images
            2. These images may be parts of the same worksheet - treat them as one complete worksheet
            3. Grade each question based on the correctness of the student's answer
            4. Provide a final score out of 40 points total
            5. Be LENIENT and ENCOURAGING in your grading - give students the benefit of the doubt
            6. NO PARTIAL MARKING - each question is either correct (full points) or incorrect (0 points)
            7. For handwriting exercises, accept answers that are reasonably legible even if not perfect
            8. For math problems, accept answers that show the correct result even if methodology is slightly unclear
            9. Focus on effort and understanding rather than perfect execution

            GRADING CRITERIA:
            - Correct or reasonably correct answers: Full points for that question
            - Clearly incorrect or completely empty answers: 0 points
            - When in doubt, give the student the benefit and award full points
            - Be encouraging and supportive in your overall feedback
            - Do not penalize minor handwriting imperfections or small errors

            Return a JSON response with this EXACT structure:
            {{
                "overall_score": <total_points_earned_out_of_40>,
                "total_possible": 40,
                "question_scores": [
                    {{
                        "question_number": 1,
                        "question": "<question_text>",
                        "student_answer": "<student's answer>",
                        "points_earned": <points_for_this_question>,
                        "max_points": <maximum_points_for_this_question>,
                        "is_correct": true/false
                    }}
                ],
                "grade_percentage": <percentage_score>,
                "overall_feedback": "<brief encouraging feedback on the overall worksheet performance>"
            }}
            """
        from conns import gemini_client
        from google.genai import types
        contents = [grading_prompt]
        for image_data in processed_images:
            contents.append(types.Part.from_bytes(data=image_data, mime_type='image/jpeg'))
        print(f"Sending {len(processed_images)} images to Gemini for direct grading...")
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash-preview-05-20',
            contents=contents
        )
        response_text = response.text.strip()
        print(f"Gemini grading response received: {response_text[:200]}...")
        # Remove code block formatting if present
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        try:
            result = json.loads(response_text)
            print(f"Direct grading completed - Score: {result.get('overall_score', 0)}/40")
            return result
        except json.JSONDecodeError as e:
            print(f"Grading JSON decode error: {str(e)}")
            print(f"Raw grading response: {response_text}")
            return {"error": f"Failed to parse grading response as JSON: {str(e)}"}
    except Exception as e:
        print(f"Gemini direct grading error: {str(e)}")
        return {"error": f"Gemini direct grading error: {str(e)}"}

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
            "overall_feedback": grading_result.get("overall_feedback", ""),
            "timestamp": (datetime.utcnow()+ timedelta(hours=5, minutes=30)).isoformat(),
            "processed_with": "gemini-2.0-flash-direct-grading"
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

