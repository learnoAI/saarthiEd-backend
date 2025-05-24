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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=5)

def load_expected_answers():
    """Load expected answers from JSON file"""
    try:
        with open('Results/dataset_questions_answers.json', 'r') as f:
            data = json.load(f)
            print(f"Successfully loaded answers for {len(data)} worksheet ranges")
            return data
    except Exception as e:
        print(f"ERROR loading expected answers: {str(e)}")
        return {}

EXPECTED_ANSWERS = load_expected_answers()

@app.get("/")
async def root():
    return {"message": "SaarthiEd API is running with Gemini 2.5 Flash"}

@app.get("/healthcheck")
async def healthcheck():
    return {"message": "ok", "model": "gemini-2.0-flash"}

def find_expected_answers(worksheet_name):
    """Find expected answers for a given worksheet name"""
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
    """Process multiple images of the same worksheet using Gemini 2.5 Flash"""
    temp_paths = []
    s3_urls = []
    all_student_entries = []
    combined_filenames = []
    
    try:
        # First, process all files and collect image bytes
        all_image_bytes = []
        
        for file in files:
            combined_filenames.append(file.filename)
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp:
                shutil.copyfileobj(file.file, temp)
                temp_path = temp.name
                temp_paths.append(temp_path)
            
            # Upload to S3
            s3_url = await asyncio.get_event_loop().run_in_executor(
                executor, upload_to_s3, temp_path
            )
            
            if not s3_url:
                raise Exception(f"Failed to upload image to S3: {file.filename}")
            
            s3_urls.append(s3_url)
            
            # Read image bytes
            with open(temp_path, "rb") as image_file:
                image_bytes = image_file.read()
                all_image_bytes.append(image_bytes)
        
        # Use Gemini for OCR with all images at once
        print(f"Processing {len(files)} images with Gemini OCR in a single batch...")
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
        
        # Extract entries from OCR response
        student_entries = extract_entries_from_response(ocr_response)
        all_student_entries.extend(student_entries)
        
        if not all_student_entries:
            for path in temp_paths:
                try:
                    os.unlink(path)
                except:
                    pass
            return {"filename": ", ".join(combined_filenames), "error": "No questions extracted from images", "success": False}
        
        # Deduplicate entries if we have multiple images
        if len(files) > 1:
            print(f"Deduplicating {len(all_student_entries)} entries from {len(files)} images")
            all_student_entries = deduplicate_student_entries(all_student_entries)
        
        # Find expected answers
        expected_answers, error_msg = find_expected_answers(worksheet_name)
        
        if expected_answers is None:
            for path in temp_paths:
                try:
                    os.unlink(path)
                except:
                    pass
            return {"filename": ", ".join(combined_filenames), "error": error_msg, "success": False}
        
        # Use Gemini for intelligent scoring
        print(f"Scoring combined worksheet with {len(files)} images using Gemini AI...")
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
        
        # Save results to MongoDB with all S3 URLs
        mongodb_id = await asyncio.get_event_loop().run_in_executor(
            executor, save_results_to_mongo, token_no, worksheet_name, scoring_result, ";".join(s3_urls), ", ".join(combined_filenames)
        )
        
        # Clean up temp files
        for path in temp_paths:
            try:
                os.unlink(path)
            except:
                pass
        
        # Prepare response
        result = {
            "filename": ", ".join(combined_filenames),
            "worksheet_name": worksheet_name,
            "token_no": token_no,
            "s3_urls": s3_urls,
            "mongodb_id": mongodb_id,
            "overall_score": (scoring_result.get("overall_score", 0)/scoring_result.get("total_possible")*40),
            "total_possible": scoring_result.get("total_possible", 40),
            "entries_count": len(all_student_entries),
            "images_count": len(files),
            "is_multi_image": len(files) > 1,
            "image_filenames": combined_filenames,
            "question_scores": scoring_result.get("question_scores", []),
            "success": True,
            "processed_with": "gemini-2.5-flash"
        }
        
        # Add diagnostics for zero scores
        if scoring_result.get("overall_score", 0) == 0:
            incorrect_answers = []
            unanswered = 0
            
            for score_entry in scoring_result.get("question_scores", []):
                if not score_entry.get("is_correct", False):
                    if score_entry.get("student_answer", "").strip() == "":
                        unanswered += 1
                    else:
                        incorrect_answers.append(
                            f"Q{score_entry.get('question_number', '?')}: answered '{score_entry.get('student_answer', '')}' instead of '{score_entry.get('expected_answer', '')}'"
                        )
            
            diagnostics = []
            if incorrect_answers:
                diagnostics.append(f"Incorrect answers: {', '.join(incorrect_answers[:5])}")
                if len(incorrect_answers) > 5:
                    diagnostics.append(f"...and {len(incorrect_answers)-5} more")
            
            if unanswered > 0:
                diagnostics.append(f"Unanswered questions: {unanswered}")
            
            if diagnostics:
                result["zero_marks_diagnostics"] = diagnostics
        
        return result
    
    except Exception as e:
        for path in temp_paths:
            try:
                os.unlink(path)
            except: 
                pass
        
        error_info = {
            "filename": ", ".join(combined_filenames) if combined_filenames else "multiple files", 
            "error": str(e), 
            "success": False,
            "images_count": len(files),
            "is_multi_image": len(files) > 1,
            "processed_image_count": len(all_student_entries)
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
    
    # Process all files for a single worksheet
    result = await process_student_worksheet(token_no, worksheet_name, files)
    
    # Check if processing was successful
    if result.get("success", False):
        processed = [result]
        errors = []
        # Clean up success flag from response
        result.pop("success", None)
    else:
        processed = []
        errors = [result]
    
    # Add multi-image processing info to the main response
    response = {
        "success": len(processed) > 0,
        "processed_count": len(processed),
        "error_count": len(errors),
        "processed": processed,
        "errors": errors,
        "model_used": "gemini-2.5-flash"
    }
    
    if processed and len(processed) > 0:
        response["total_images_processed"] = processed[0].get("images_count", 1)
        response["is_multi_image"] = processed[0].get("is_multi_image", False)
        
    return response

if __name__ == "__main__":
    import uvicorn 
    uvicorn.run("app:app", port=8080, reload=True)
