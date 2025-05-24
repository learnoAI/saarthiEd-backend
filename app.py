from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import tempfile
import shutil
from utils import (use_gemini_for_ocr, use_gemini_for_scoring, save_results_to_mongo, 
                  upload_to_s3, extract_entries_from_response, deduplicate_student_entries)
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import uvicorn 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=5)

with open('Results/dataset_questions_answers.json', 'r') as f:
    EXPECTED_ANSWERS = json.load(f)

@app.get("/")
async def root():
    return {"message": "SaarthiEd Python API is running with Gemini 2.5 Flash"}

@app.get("/healthcheck")
async def healthcheck():
    return {"message": "ok", "model": "gemini-2.5-flash-preview-05-20"}

def find_expected_answers(worksheet_name):
    worksheet_range = None
    for range_key in EXPECTED_ANSWERS.keys():
        if worksheet_name in EXPECTED_ANSWERS[range_key]:
            worksheet_range = range_key
            print(f"Found worksheet {worksheet_name} in range {range_key}")
            break
    
    if worksheet_range is None:
        return None, f"No expected answers found for worksheet {worksheet_name}"
    
    expected_answers = EXPECTED_ANSWERS[worksheet_range].get(worksheet_name, [])
    
    if not expected_answers:
        return None, f"No expected answers found for worksheet {worksheet_name} in range {worksheet_range}"
    
    return expected_answers, None

async def process_student_worksheet(token_no, worksheet_name, files):
    temp_paths = []
    s3_urls = []
    all_student_entries = []
    combined_filenames = []
    
    try:
        all_image_bytes = []
        
        for file in files:
            combined_filenames.append(file.filename)
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp:
                shutil.copyfileobj(file.file, temp)
                temp_path = temp.name
                temp_paths.append(temp_path)
            
            s3_url = await asyncio.get_event_loop().run_in_executor(
                executor, upload_to_s3, temp_path
            )
            
            if not s3_url:
                raise Exception(f"Failed to upload image to S3: {file.filename}")
            
            s3_urls.append(s3_url)
            
            with open(temp_path, "rb") as image_file:
                image_bytes = image_file.read()
                all_image_bytes.append(image_bytes)
        
        ocr_response = await asyncio.get_event_loop().run_in_executor(
            executor, use_gemini_for_ocr, all_image_bytes
        )
        
        if "error" in ocr_response:
            for path in temp_paths:
                try:
                    os.unlink(path)
                except:
                    pass
            return {"filename": ", ".join(combined_filenames), "error": ocr_response["error"], "success": False}
        
        student_entries = extract_entries_from_response(ocr_response)
        all_student_entries.extend(student_entries)
        
        if not all_student_entries:
            for path in temp_paths:
                try:
                    os.unlink(path)
                except:
                    pass
            return {"filename": ", ".join(combined_filenames), "error": "No questions extracted from images", "success": False}
        
        if len(files) > 1:
            print(f"Deduplicating {len(all_student_entries)} entries from {len(files)} images")
            all_student_entries = deduplicate_student_entries(all_student_entries)
        
        expected_answers, error_msg = find_expected_answers(worksheet_name)
        
        if expected_answers is None:
            for path in temp_paths:
                try:
                    os.unlink(path)
                except:
                    pass
            return {"filename": ", ".join(combined_filenames), "error": error_msg, "success": False}
        
        scoring_result = await asyncio.get_event_loop().run_in_executor(
            executor, use_gemini_for_scoring, all_student_entries, expected_answers, worksheet_name
        )
        
        if "error" in scoring_result:
            for path in temp_paths:
                try:
                    os.unlink(path)
                except:
                    pass
            return {"filename": ", ".join(combined_filenames), "error": scoring_result["error"], "success": False}
        
        mongodb_id = await asyncio.get_event_loop().run_in_executor(
            executor, save_results_to_mongo, token_no, worksheet_name, scoring_result, ";".join(s3_urls), ", ".join(combined_filenames)
        )
        
        for path in temp_paths:
            try:
                os.unlink(path)
            except:
                pass
        
        result = {
            "success": True,
            "token_no": token_no,
            "worksheet_name": worksheet_name,
            "mongodb_id": mongodb_id,
            "grade": (scoring_result.get("overall_score", 0)/scoring_result.get("total_possible")*40),
        }
        
        return result
    
    except Exception as e:
        for path in temp_paths:
            try:
                os.unlink(path)
            except: 
                pass
        
        error_info = {
            "success": False,
            "error": str(e)
        }
        
        if combined_filenames:
            error_info["image_filenames"] = combined_filenames
            
        return error_info

@app.post("/process-worksheets")
async def process_worksheets(token_no: str, worksheet_name: str, files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded")
    
    if not token_no:
        raise HTTPException(status_code=400, detail="token_no is required")
    
    if not worksheet_name:
        raise HTTPException(status_code=400, detail="worksheet_name is required")
    
    print(f"Processing {len(files)} images for worksheet {worksheet_name}, token {token_no}")
    
    result = await process_student_worksheet(token_no, worksheet_name, files)
     
    return result

if __name__ == "__main__":
    uvicorn.run("app:app", port=8080, reload=True)
