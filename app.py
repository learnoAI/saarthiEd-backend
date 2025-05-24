from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import tempfile
import shutil
from utils import use_gemini_for_ocr, use_gemini_for_scoring, save_results_to_mongo, upload_to_s3, extract_entries_from_response
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
    return {"message": "SaarthiEd API is running with Gemini 2.0 Flash"}

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

async def process_student_worksheet(token_no, worksheet_name, file):
    """Process a single student worksheet using Gemini 2.5 Flash"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp:
            shutil.copyfileobj(file.file, temp)
            temp_path = temp.name
        
        # Upload to S3
        s3_url = await asyncio.get_event_loop().run_in_executor(
            executor, upload_to_s3, temp_path
        )
        
        if not s3_url:
            raise Exception(f"Failed to upload image to S3: {file.filename}")
        
        # Read image bytes
        with open(temp_path, "rb") as image_file:
            image_bytes = image_file.read()
        
        # Use Gemini for OCR
        print(f"Processing {file.filename} with Gemini OCR...")
        ocr_response = await asyncio.get_event_loop().run_in_executor(
            executor, use_gemini_for_ocr, image_bytes
        )
        
        if "error" in ocr_response:
            os.unlink(temp_path)
            return {"filename": file.filename, "error": ocr_response["error"], "success": False}
        
        # Extract entries from OCR response
        student_entries = extract_entries_from_response(ocr_response)
        
        if not student_entries:
            os.unlink(temp_path)
            return {"filename": file.filename, "error": "No questions extracted from image", "success": False}
        
        # Find expected answers
        expected_answers, error_msg = find_expected_answers(worksheet_name)
        
        if expected_answers is None:
            os.unlink(temp_path)
            return {"filename": file.filename, "error": error_msg, "success": False}
        
        # Use Gemini for intelligent scoring
        print(f"Scoring {file.filename} with Gemini AI...")
        scoring_result = await asyncio.get_event_loop().run_in_executor(
            executor, use_gemini_for_scoring, student_entries, expected_answers, worksheet_name
        )
        
        if "error" in scoring_result:
            os.unlink(temp_path)
            return {"filename": file.filename, "error": scoring_result["error"], "success": False}
        
        # Save results to MongoDB
        mongodb_id = await asyncio.get_event_loop().run_in_executor(
            executor, save_results_to_mongo, token_no, worksheet_name, scoring_result, s3_url, file.filename
        )
        
        # Clean up temp file
        os.unlink(temp_path)
        
        # Prepare response
        result = {
            "filename": file.filename,
            "worksheet_name": worksheet_name,
            "token_no": token_no,
            "s3_url": s3_url,
            "mongodb_id": mongodb_id,
            "overall_score": (scoring_result.get("overall_score", 0)/scoring_result.get("total_possible")*40),
            "total_possible": scoring_result.get("total_possible", 40),
            "entries_count": len(student_entries),
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
        try:
            os.unlink(temp_path)
        except: 
            pass
        return {"filename": file.filename, "error": str(e), "success": False}

@app.post("/process-worksheets")
async def process_worksheets(token_no: str, worksheet_name: str, files: List[UploadFile] = File(...)):
    """Process multiple student worksheets using Gemini 2.5 Flash for OCR and scoring"""
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded")
    
    if not token_no:
        raise HTTPException(status_code=400, detail="token_no is required")
    
    if not worksheet_name:
        raise HTTPException(status_code=400, detail="worksheet_name is required")
    
    print(f"Processing {len(files)} worksheets for token {token_no}, worksheet {worksheet_name}")
    
    # Process all files in parallel
    tasks = [process_student_worksheet(token_no, worksheet_name, file) for file in files]
    results = await asyncio.gather(*tasks)
    
    # Separate successful and failed results
    processed = [r for r in results if r.get("success", False)]
    errors = [r for r in results if not r.get("success", False)]
    
    # Clean up success flag from response
    for r in processed: 
        r.pop("success", None)
    
    return {
        "success": len(processed) > 0,
        "processed_count": len(processed),
        "error_count": len(errors),
        "processed": processed,
        "errors": errors,
        "model_used": "gemini-2.5-flash"
    }

if __name__ == "__main__":
    import uvicorn 
    uvicorn.run("app:app", port=8080, reload=True)
