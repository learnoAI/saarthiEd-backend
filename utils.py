import base64
import os
import json
import re
import os.path
from conns import s3_client, groq_client

S3_BUCKET_NAME = "learno-pdf-document"

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return image_file.read()

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

def extract_entries_from_response(response_data):
    entries = []
    
    for key, value in response_data.items():
        if not key.startswith('_') and isinstance(value, dict) and 'question' in value and 'answer' in value:
            entries.append({
                'question_id': key,
                'question': value['question'],
                'answer': value['answer']
            })
    
    return entries