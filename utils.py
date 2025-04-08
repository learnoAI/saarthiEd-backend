from groq import Groq
import google.generativeai as genai
import base64
import os
from dotenv import load_dotenv
import json
import PIL.Image
from io import BytesIO
import re

load_dotenv()

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
gemini_client = genai.GenerativeModel('gemini-2.0-flash-lite')

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return image_file.read()

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

def use_gemini(image_bytes):
    try:
        pil_image = PIL.Image.open(BytesIO(image_bytes))
        
        response = gemini_client.generate_content(
                contents=[
                "You are an AI assistant tasked with extracting student answers from exam sheets. Your role is to accurately transcribe both questions and answers exactly as written, without making any corrections. Extract the content and format it as a JSON object with the following requirements:\n\n1. Format: {'q1':{'question':'<exact question text>', 'answer':'<student's exact answer>'}, 'q2':{'question':'<exact question text>', 'answer':'<student's exact answer>'}, ...}\n2. Maintain the original question order\n3. Preserve all spelling, grammar, and punctuation exactly as written\n4. Include all visible text from the image\n\nReturn ONLY the SINGLE LINE JSON object in a single line with no additional text or explanations.",
                pil_image
            ]
        )
        
        try:
            return json.loads(response.text)
        except json.JSONDecodeError as e:
            fixed_json = fix_json(response.text)
            if "error" not in fixed_json:
                return fixed_json
            return {"error": f"JSON decode error: {str(e)}", "raw_content": response.text}
    except Exception as e:
        return {"error": f"Gemini API error: {str(e)}"}

def main(images):
    gr = []
    gm = []
    errors = []
    
    for i, image in enumerate(images):
        try:
            image_bytes = encode_image(image)
            # gr_response = use_groq(image_bytes)
            # gr.append(gr_response)
            gm_response = use_gemini(image_bytes)
            gm.append(gm_response)

            if isinstance(gm_response, dict) and "error" in gm_response:
                errors.append({"image": image, "error": gm_response["error"], "index": i})
                
        except Exception as e:
            errors.append({"image": image, "error": str(e), "index": i})
    
    # with open("llama4_scout_response.json", "w") as f:
    #     json.dump(gr, f, indent=4)
    with open("gemini2_flash_lite_response.json", "w") as f:
        json.dump(gm, f, indent=4)
        
    if errors:
        with open("json_decode_errors.json", "w") as f:
            json.dump(errors, f, indent=4)
        print(f"Encountered {len(errors)} errors. See json_decode_errors.json for details.")

sw = ['sw/1000123982.jpg','sw/WhatsApp Image 2025-03-11 at 11.25.16 AM (3).jpeg',"sw/1000123926.jpg","sw/1000123927.jpg","sw/1000123928.jpg","sw/1000123929.jpg","sw/1000123930.jpg","sw/1000123931.jpg","sw/1000123932.jpg","sw/1000123933.jpg","sw/1000123934.jpg"]
main(sw)