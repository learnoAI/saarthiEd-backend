from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import tempfile
import shutil
from utils import use_gemini_for_direct_grading, save_results_to_mongo, upload_to_s3
import asyncio
from concurrent.futures import ThreadPoolExecutor
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

@app.get("/")
async def root():
    return {"message": "Saarthi AI Score API is running"}

@app.get("/healthcheck")
async def healthcheck():
    return {"message": "ok"}

async def process_student_worksheet(token_no, worksheet_name, files):
    temp_paths = []
    s3_urls = []
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
        
        # Use Gemini for direct grading (combines OCR and grading)
        grading_result = await asyncio.get_event_loop().run_in_executor(
            executor, use_gemini_for_direct_grading, all_image_bytes, worksheet_name
        )
        
        if "error" in grading_result:
            for path in temp_paths:
                try:
                    os.unlink(path)
                except:
                    pass
            return {"filename": ", ".join(combined_filenames), "error": grading_result["error"], "success": False}
        
        mongodb_id = await asyncio.get_event_loop().run_in_executor(
            executor, save_results_to_mongo, token_no, worksheet_name, grading_result, ";".join(s3_urls), ", ".join(combined_filenames)
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
            "grade": grading_result.get("overall_score", 0),
            "total_possible": 40,
            "grade_percentage": grading_result.get("grade_percentage", 0),
            "total_questions": grading_result.get("total_questions", 0),
            "correct_answers": grading_result.get("correct_answers", 0),
            "wrong_answers": grading_result.get("wrong_answers", 0),
            "unanswered": grading_result.get("unanswered", 0),
            "question_scores": grading_result.get("question_scores", []),
            "wrong_questions": grading_result.get("wrong_questions", []),
            "correct_questions": grading_result.get("correct_questions", []),
            "unanswered_questions": grading_result.get("unanswered_questions", []),
            "overall_feedback": grading_result.get("overall_feedback", "")
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
