from groq import Groq
import google.generativeai as genai
import base64
import os
from dotenv import load_dotenv
import json
# import PIL.Image
from io import BytesIO
import re
import os.path
from datetime import datetime
import time
from collections import deque
from conns import collection, s3_client
load_dotenv()

S3_BUCKET_NAME = "learno-pdf-document"

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
gemini_client = genai.GenerativeModel('gemini-2.0-flash-lite')

# RPM handler for Gemini API
class RPMHandler:
    def __init__(self, rpm_limit=30):
        self.rpm_limit = rpm_limit
        self.request_timestamps = deque()
    
    def wait_if_needed(self):
        current_time = time.time()
        
        # Remove timestamps older than 60 seconds
        while self.request_timestamps and current_time - self.request_timestamps[0] > 60:
            self.request_timestamps.popleft()
        
        # If we've reached the limit, wait until we can make another request
        if len(self.request_timestamps) >= self.rpm_limit:
            wait_time = 60 - (current_time - self.request_timestamps[0])
            if wait_time > 0:
                print(f"Rate limit reached, waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time)
        
        # Add current request timestamp
        self.request_timestamps.append(time.time())

# Initialize RPM handler
gemini_rpm_handler = RPMHandler(rpm_limit=30)

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return image_file.read()

def upload_to_s3(file_path):
    try:
        file_name = os.path.basename(file_path)
        s3_key = f"worksheets-{file_name}"
        
        # Upload with public-read ACL
        with open(file_path, 'rb') as file_data:
            s3_client.upload_fileobj(
                file_data, 
                S3_BUCKET_NAME, 
                s3_key,
                ExtraArgs={'ACL': 'public-read'}
            )
        
        # Generate S3 URL
        s3_url = f"https://{S3_BUCKET_NAME}.s3.ap-south-1.amazonaws.com/{s3_key}"
        return s3_url
    except Exception as e:
        print(f"Error uploading to S3: {str(e)}")
        return None

def fix_json(json_str):
    json_match = re.search(r'(\{.*\})', json_str, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    json_str = json_str.replace("'", '"')
    
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    json_str = re.sub(r'([{,])\s*([a-zA-Z0-9_]+):', r'\1"\2":', json_str)
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {"error": "Couldn't fix JSON", "raw_content": json_str}

def use_groq(image_bytes):
    try:
        base64_str = base64.b64encode(image_bytes).decode('utf-8')
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {   "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a helpful assistant that specializes in extracting student answers from images without correcting their answers. You will be provided with an image of a student's answer sheet, and your task is to extract the text from the image and return it in a specific JSON format. DO NOT TRY TO FIX THE ANSWERS."},
                        {"type": "text", "text": "Please extract the text from the image and return it in a Single Line JSON format. DO NOT RETURN ANYTHING ELSE JUST THE SINGLE LINE JSON."},
                    ],
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Return me the contents of this image. I am processing a document where students have to write the answer to the question. I need to extract the answer from the image along with the respective question in proper SINGLE LINE json format. DO not mess with the order of the questions. Follow this JSON format: {'q1':{'question':'<question>', 'answer':'<answer>'}, 'q2':{'question':'<question>', 'answer':'<answer>'}, ...}. Please do not add any extra text or explanation. Just return the JSON IN A SINGLE LINE NOTHING ELSE."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_str}",
                            },
                        },
                    ],
                }
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
        )
        try:
            return json.loads(chat_completion.choices[0].message.content)
        except json.JSONDecodeError as e:
            fixed_json = fix_json(chat_completion.choices[0].message.content)
            if "error" not in fixed_json:
                return fixed_json
            return {"error": f"JSON decode error: {str(e)}", "raw_content": chat_completion.choices[0].message.content}
    except Exception as e:
        return {"error": f"Groq API error: {str(e)}"}

# def use_gemini(image_bytes):
#     try:
#         # Apply rate limiting
#         gemini_rpm_handler.wait_if_needed()
        
#         pil_image = PIL.Image.open(BytesIO(image_bytes))
        
#         response = gemini_client.generate_content(
#                 contents=[
#                 "You are an AI assistant tasked with extracting student answers from exam sheets. Your role is to accurately transcribe both questions and answers exactly as written, without making any corrections. Extract the content and format it as a JSON object with the following requirements:\n\n1. Format: {'q1':{'question':'<exact question text>', 'answer':'<student's exact answer>'}, 'q2':{'question':'<exact question text>', 'answer':'<student's exact answer>'}, ...}\n2. Maintain the original question order\n3. Preserve all spelling, grammar, and punctuation exactly as written\n4. Include all visible text from the image\n\nReturn ONLY the SINGLE LINE JSON object in a single line with no additional text or explanations.",
#                 pil_image
#             ]
#         )
        
#         try:
#             return json.loads(response.text)
#         except json.JSONDecodeError as e:
#             fixed_json = fix_json(response.text)
#             if "error" not in fixed_json:
#                 return fixed_json
#             return {"error": f"JSON decode error: {str(e)}", "raw_content": response.text}
#     except Exception as e:
#         return {"error": f"Gemini API error: {str(e)}"}

def extract_entries_from_response(response_data):
    entries = []
    
    # Filter out metadata fields and process only question/answer pairs
    for key, value in response_data.items():
        if not key.startswith('_') and isinstance(value, dict) and 'question' in value and 'answer' in value:
            entries.append({
                'question_id': key,
                'question': value['question'],
                'answer': value['answer']
            })
    
    return entries

# def main(images):
#     gr_responses = []
#     gm_responses = []
#     errors = []
#     mongo_documents = []
    
#     for i, image_path in enumerate(images):
#         try:
#             # Extract filename without path and extension for worksheet name
#             filename = os.path.basename(image_path)
#             worksheet_name = os.path.splitext(filename)[0]
            
#             # Upload image to S3 and get the URL
#             s3_url = upload_to_s3(image_path)
#             if not s3_url:
#                 raise Exception(f"Failed to upload image to S3: {image_path}")
            
#             image_bytes = encode_image(image_path)
            
            # Process with Groq
            # gr_response = use_groq(image_bytes)
            # gr_responses.append(gr_response)
            
            # Save worksheet to MongoDB with proper structure
            # if "error" not in gr_response:
            #     entries = extract_entries_from_response(gr_response)
                
            #     worksheet_doc = {
            #         "name": worksheet_name,
            #         "entries": entries,
            #         "processor": "groq",
            #         "model": "llama-4-scout-17b-16e-instruct",
            #         "processed_at": datetime.now(),
            #         "source_image": s3_url
            #     }
                
            #     mongo_documents.append(worksheet_doc)
            
            # Process with Gemini
    #         gm_response = use_gemini(image_bytes)
    #         gm_responses.append(gm_response)

    #         # Save Gemini worksheet to MongoDB with proper structure
    #         if "error" not in gm_response:
    #             entries = extract_entries_from_response(gm_response)
                
    #             worksheet_doc = {
    #                 "name": worksheet_name,
    #                 "entries": entries,
    #                 "processor": "gemini",
    #                 "model": "gemini-2.0-flash-lite",
    #                 "processed_at": datetime.now(),
    #                 "source_image": s3_url
    #             }
                
    #             mongo_documents.append(worksheet_doc)

    #         # Check for errors
    #         if isinstance(gm_response, dict) and "error" in gm_response:
    #             errors.append({"image": image_path, "error": gm_response["error"], "index": i})
                
    #     except Exception as e:
    #         errors.append({"image": image_path, "error": str(e), "index": i})
    
    # # Save worksheets to MongoDB
    # if mongo_documents:
    #     collection.insert_many(mongo_documents)
    #     print(f"Inserted {len(mongo_documents)} worksheet documents into MongoDB")
    
    # # Save raw responses to files for reference
    # # with open("llama4_scout_response.json", "w") as f:
    # #     json.dump(gr_responses, f, indent=4)
        
    # with open("gemini2_flash_lite_response.json", "w") as f:
    #     json.dump(gm_responses, f, indent=4)
        
    # if errors:
    #     with open("json_decode_errors.json", "w") as f:
    #         json.dump(errors, f, indent=4)
    #     print(f"Encountered {len(errors)} errors. See json_decode_errors.json for details.")
    
    # print(f"Processed {len(images)} images")

# sw = os.listdir("sw")
# sw= [f"sw/{f}" for f in sw]

# main(sw)