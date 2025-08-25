from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import os
import tempfile
from utils import process_worksheet_with_gemini_direct_grading, save_worksheet_results_to_mongodb, upload_file_to_s3
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uvicorn
from contextlib import asynccontextmanager
from conns import collection
from schema import getImages

executor = ThreadPoolExecutor(max_workers=min(10, (os.cpu_count() or 1) * 2))

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    executor.shutdown(wait=True)

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "Saarthi AI Score API is running"}

@app.get("/healthcheck")
async def healthcheck() -> Dict[str, str]:
    return {"message": "ok"}

async def process_student_worksheet(token_no: str, worksheet_name: str, files: List[UploadFile]) -> Dict[str, Any]:
    temp_paths = []
    s3_urls = []
    combined_filenames = []
    
    try:
        all_image_bytes = []
        
        async def process_single_file(file: UploadFile) -> tuple[str, str, bytes]:
            combined_filenames.append(file.filename)
            
            suffix = os.path.splitext(file.filename)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
                temp_path = temp.name
                content = await file.read()
                temp.write(content)
                temp_paths.append(temp_path)
            
            loop = asyncio.get_event_loop()
            s3_url = await loop.run_in_executor(executor, upload_file_to_s3, temp_path)
            
            if not s3_url:
                raise Exception(f"Failed to upload image to S3: {file.filename}")
            
            return temp_path, s3_url, content
        
        results = await asyncio.gather(*[process_single_file(file) for file in files])
        
        for _, s3_url, content in results:
            s3_urls.append(s3_url)
            all_image_bytes.append(content)
        
        loop = asyncio.get_event_loop()
        grading_result = await loop.run_in_executor(
            executor, process_worksheet_with_gemini_direct_grading, all_image_bytes, worksheet_name
        )
        
        if "error" in grading_result:
            return {"filename": ", ".join(combined_filenames), "error": grading_result["error"], "success": False}
        
        mongodb_id = await loop.run_in_executor(
            executor, save_worksheet_results_to_mongodb, token_no, worksheet_name, grading_result, ";".join(s3_urls), ", ".join(combined_filenames)
        )
        
        return {
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
    
    except Exception as e:
        error_info = {
            "success": False,
            "error": str(e)
        }
        
        if combined_filenames:
            error_info["image_filenames"] = combined_filenames
            
        return error_info
    finally:
        async def cleanup_temp_files():
            for path in temp_paths:
                try:
                    os.unlink(path)
                except:
                    pass
        
        asyncio.create_task(cleanup_temp_files())

@app.post("/process-worksheets")
async def process_worksheets(token_no: str, worksheet_name: str, files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded")
    
    if not token_no:
        raise HTTPException(status_code=400, detail="token_no is required")
    
    if not worksheet_name:
        raise HTTPException(status_code=400, detail="worksheet_name is required")
    
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"File {file.filename} has unsupported format. Allowed: {', '.join(allowed_extensions)}"
            )
    
    print(f"Processing {len(files)} images for worksheet {worksheet_name}, token {token_no}")
    
    return await process_student_worksheet(token_no, worksheet_name, files)

@app.post("/get-worksheet-images")
async def get_worksheet_images(req: getImages):
    doc = collection.find_one({"token_no": req.token_no, "worksheet_name": req.worksheet_name})
    return doc['s3_urls']

@app.get("/total-ai-graded")
async def total_ai_graded():
    count = collection.estimated_document_count()
    return {"total_ai_graded": count}

if __name__ == "__main__":
    uvicorn.run(
        "app:app", 
        port=8080, 
        reload=True,
        workers=2,
        loop="asyncio",
        access_log=False
    )
