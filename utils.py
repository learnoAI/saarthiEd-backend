import base64
import os
import json
import re
from conns import s3_client, gemini_client, collection
from google.genai import types
from datetime import datetime
from bson import ObjectId
from PIL import Image
import io

S3_BUCKET_NAME = "learno-pdf-document"

def encode_image(image_path):
    """Read image file as bytes"""
    with open(image_path, "rb") as image_file:
        return image_file.read()

def upload_to_s3(file_path):
    """Upload file to S3 and return public URL"""
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
    """
    Use Gemini 2.5 Flash for OCR to extract questions and answers from worksheet images
    
    This function can process multiple worksheet images at once and extracts questions and answers.
    Accepts either a single image_bytes or a list of image_bytes.
    """
    try:
        # Check if we have a single image or multiple images
        if not isinstance(image_bytes_list, list):
            image_bytes_list = [image_bytes_list]
        
        # Process each image into proper format
        processed_images = []
        for image_bytes in image_bytes_list:
            # Convert bytes to PIL Image for better processing
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save to bytes buffer
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='JPEG', quality=95)
            processed_images.append(img_buffer.getvalue())
        
        # Prepare the prompt for OCR
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
        
        # Prepare content array with prompt and all images
        contents = [ocr_prompt]
        for image_data in processed_images:
            contents.append(types.Part.from_bytes(data=image_data, mime_type='image/jpeg'))
            
        print(f"Sending {len(processed_images)} images to Gemini OCR in a single request...")
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash-preview-05-20',
            contents=contents
        )
        
        response_text = response.text.strip()
        
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        # Parse JSON
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
    """
    Use Gemini 2.5 Flash to intelligently score student answers against expected answers
    
    This function can handle student entries combined from multiple images of the same worksheet.
    It will consolidate all extracted questions and answers before scoring them against the expected answers.
    """
    try:
        # Prepare data for scoring
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
          # Add information about the number of entries and possible multi-image source
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
                    "points_possible": 1,
                    "is_correct": true/false,
                    "feedback": "<brief explanation>"
                    }}
                ]
                }}
                """
        
        # Generate scoring using Gemini
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash-preview-05-20',
            contents=scoring_prompt
        )
        response_text = response.text.strip()
        
        # Clean up response
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        result = json.loads(response_text)
        print(f"Gemini scoring completed. Score: {result.get('overall_score', 0)}/{result.get('total_possible', len(expected_answers))}")
        return result
        
    except json.JSONDecodeError as e:
        print(f"Scoring JSON decode error: {str(e)}")
        print(f"Raw scoring response: {response.text}")
        return {"error": f"Failed to parse scoring response as JSON: {str(e)}"}
    except Exception as e:
        print(f"Gemini scoring error: {str(e)}")
        return {"error": f"Gemini scoring error: {str(e)}"}

def save_results_to_mongo(token_no, worksheet_name, scoring_result, s3_url, filename):
    """
    Save the scoring results to MongoDB and return the document ID
    
    For multi-image worksheets, s3_url contains semicolon-separated URLs and
    filename contains comma-separated filenames.
    """
    try:        # Parse multiple S3 URLs if present
        s3_urls = s3_url.split(';')
        filenames = filename.split(', ')
        
        document = {
            "token_no": token_no,
            "worksheet_name": worksheet_name,
            "filename": filename,
            "s3_url": s3_url,
            "s3_urls": s3_urls,
            "image_count": len(s3_urls),
            "is_multi_image": len(s3_urls) > 1,
            "filenames": filenames,
            "overall_score": scoring_result.get("overall_score", 0),
            "total_possible": scoring_result.get("total_possible", 40),
            "question_scores": scoring_result.get("question_scores", []),
            "timestamp": datetime.utcnow(),
            "processed_with": "gemini-2.5-flash"
        }
        
        # Insert into MongoDB
        result = collection.insert_one(document)
        print(f"Saved results to MongoDB with ID: {result.inserted_id}")
        return str(result.inserted_id)
        
    except Exception as e:
        print(f"MongoDB save error: {str(e)}")
        return None

def extract_entries_from_response(response_data):
    """
    Extract entries from Gemini OCR response and convert to expected format
    """
    entries = []
    
    if isinstance(response_data, dict):
        for key, value in response_data.items():
            # Skip non-question keys and ensure we have the right structure
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
    """
    Deduplicate student entries from multiple images
    
    When processing multiple images of the same worksheet, we might get
    duplicate questions. This function keeps the best answer for each question.
    
    A 'better' answer is one that is non-empty when compared to an empty answer.
    If both answers are non-empty, we keep the longer one as it likely contains more information.
    """
    # Group entries by question text (normalized)
    question_groups = {}
    for entry in all_entries:
        # Normalize question text for comparison (lowercase, trim, remove extra spaces)
        norm_question = ' '.join(entry['question'].lower().strip().split())
        
        if norm_question not in question_groups:
            question_groups[norm_question] = []
        
        question_groups[norm_question].append(entry)
    
    # For each group, select the best entry
    deduplicated = []
    for question, entries in question_groups.items():
        if len(entries) == 1:
            # Only one entry for this question
            deduplicated.append(entries[0])
        else:
            # Multiple entries - choose the best answer
            best_entry = entries[0]
            for entry in entries[1:]:
                # If current best is empty but this one isn't
                if not best_entry['answer'].strip() and entry['answer'].strip():
                    best_entry = entry
                # Both have answers, prefer the longer one
                elif (best_entry['answer'].strip() and entry['answer'].strip() and 
                      len(entry['answer']) > len(best_entry['answer'])):
                    best_entry = entry
            
            deduplicated.append(best_entry)
    
    print(f"Deduplicated {len(all_entries)} entries to {len(deduplicated)} unique questions")
    return deduplicated

# For backward compatibility - use Gemini instead of Groq
def use_groq(image_bytes):
    """
    Backward compatibility function - now uses Gemini instead of Groq
    """
    return use_gemini_for_ocr(image_bytes)