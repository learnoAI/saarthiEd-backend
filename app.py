from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import os
import tempfile
from utils import process_worksheet_with_gemini_direct_grading, save_worksheet_results_to_mongodb, upload_file_to_s3, log_error
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uvicorn
from contextlib import asynccontextmanager
from conns import collection
from schema import getImages, gradeDetails, TimeRangeFilter
from datetime import datetime

executor = ThreadPoolExecutor(max_workers=min(20, (os.cpu_count() or 1) * 4))

# Request size limits
MAX_FILE_SIZE_MB = 10  # 10MB per file
MAX_TOTAL_SIZE_MB = 50  # 50MB total request size
MAX_FILES_PER_REQUEST = 10  # Maximum 10 files per request

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

async def validate_uploaded_files(files: List[UploadFile]) -> None:
    """
    Validate uploaded files before processing.
    Checks file count, individual sizes, and total payload size.

    Raises:
        HTTPException: If validation fails
    """
    # Check file count
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum {MAX_FILES_PER_REQUEST} files allowed, received {len(files)}"
        )

    # Check individual file sizes and types
    total_size = 0
    for file in files:
        # Get file size
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning

        file_size_mb = file_size / 1024 / 1024

        if file_size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=f"File {file.filename} exceeds maximum size of {MAX_FILE_SIZE_MB}MB (size: {file_size_mb:.2f}MB)"
            )

        total_size += file_size_mb

    # Check total payload size
    if total_size > MAX_TOTAL_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"Total payload size {total_size:.2f}MB exceeds maximum {MAX_TOTAL_SIZE_MB}MB"
        )

async def cleanup_temp_files(paths: List[str]):
    """
    Cleanup temporary files with proper error handling.
    Guaranteed to run even if processing fails.
    """
    if not paths:
        return

    cleanup_errors = []
    for path in paths:
        try:
            if os.path.exists(path):
                os.unlink(path)
                print(f"Cleaned up temp file: {path}")
        except Exception as e:
            cleanup_errors.append(f"{path}: {str(e)}")
            print(f"Failed to cleanup {path}: {str(e)}")

    if cleanup_errors:
        print(f"Cleanup completed with {len(cleanup_errors)} errors")
        # Log cleanup failures for monitoring
        log_error(
            "TEMP_FILE_CLEANUP_ERROR",
            f"Failed to cleanup {len(cleanup_errors)} temp files",
            {"failed_paths": cleanup_errors}
        )

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
        log_error("WORKSHEET_PROCESSING_ERROR", str(e), {
            "token_no": token_no,
            "worksheet_name": worksheet_name,
            "filenames": combined_filenames,
            "s3_urls": s3_urls
        })

        error_info = {
            "success": False,
            "error": str(e)
        }

        if combined_filenames:
            error_info["image_filenames"] = combined_filenames

        return error_info
    finally:
        # Await cleanup to ensure temp files are deleted
        await cleanup_temp_files(temp_paths)

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

    # Validate file sizes before processing
    await validate_uploaded_files(files)

    print(f"Processing {len(files)} images for worksheet {worksheet_name}, token {token_no}")
    
    return await process_student_worksheet(token_no, worksheet_name, files)

@app.post("/get-worksheet-images")
async def get_worksheet_images(req: getImages):
    loop = asyncio.get_event_loop()
    doc = await loop.run_in_executor(
        executor, 
        collection.find_one, 
        {"token_no": req.token_no, "worksheet_name": req.worksheet_name}
    )
    if doc and 's3_urls' in doc:
        return doc['s3_urls']
    else:
        raise HTTPException(status_code=404, detail="Worksheet not found")

@app.post("/total-ai-graded")
async def total_ai_graded(time_filter: TimeRangeFilter):
    loop = asyncio.get_event_loop()
    
    if time_filter.full:
        count = await loop.run_in_executor(executor, collection.estimated_document_count)
        return {"total_ai_graded": count}
    
    if not time_filter.start_time and not time_filter.end_time:
        raise HTTPException(
            status_code=400, 
            detail="Either set 'full' to true or provide 'start_time' and/or 'end_time'"
        )
    
    def parse_and_convert_date(date_str: str, is_end_of_day: bool = False) -> str:
        date_str = date_str.replace('Z', '+00:00')
        
        try:
            dt = datetime.fromisoformat(date_str)
        except ValueError:
            try:
                parts = date_str.split('-')
                if len(parts) == 3:
                    dt = datetime(int(parts[0]), int(parts[1]), int(parts[2]))
                else:
                    raise ValueError(f"Invalid date format: {date_str}")
            except (ValueError, IndexError):
                raise ValueError(f"Invalid date format: {date_str}. Use format like 2025-11-1 or 2025-11-01")
        
        if is_end_of_day:
            dt = dt.replace(hour=23, minute=59, second=59, microsecond=999999)
        else:
            dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        
        return dt.isoformat()
    
    query = {"timestamp": {}}
    
    try:
        if time_filter.start_time:
            query["timestamp"]["$gte"] = parse_and_convert_date(time_filter.start_time, is_end_of_day=False)
        
        if time_filter.end_time:
            query["timestamp"]["$lte"] = parse_and_convert_date(time_filter.end_time, is_end_of_day=True)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    count = await loop.run_in_executor(executor, lambda: collection.count_documents(query))
    
    return {"total_ai_graded": count}

@app.post("/student-grading-details")
async def get_student_gradind_details(req: gradeDetails):
    query = {"token_no": req.token_no, 
             "worksheet_name": req.worksheet_name,
             "question_scores": {
                    "$exists": True,
                    "$nin": [None, "NA"]
                }}
    
    # Only filter by overall_score if provided
    if req.overall_score is not None:
        query["overall_score"] = req.overall_score
    
    projection = {
        "token_no": 0,
        "worksheet_name": 0,
        "filename": 0,
        "s3_urls": 0,
        "s3_url": 0,
        "image_count": 0,
        "filenames": 0,
        "_id": 0,
        "grading_method": 0,
        "has_answer_key": 0,
        "timestamp": 0,
        "processed_with": 0,
        "reason_why": 0
    }
    
    loop = asyncio.get_event_loop()
    doc = await loop.run_in_executor(
        executor,
        lambda: collection.find_one(query, projection=projection)
    )
    
    # Fallback: if not found with overall_score, retry without it
    if doc is None and req.overall_score is not None:
        fallback_query = {k: v for k, v in query.items() if k != "overall_score"}
        doc = await loop.run_in_executor(
            executor,
            lambda: collection.find_one(fallback_query, projection=projection)
        )
    
    if doc is None:
        raise HTTPException(status_code=404, detail="Student grading details not found")
    
    return doc

if __name__ == "__main__":
    uvicorn.run(
        "app:app", 
        host=os.getenv("HOST", "127.0.0.1"),
        port=os.getenv("PORT", 8080),
        reload=True,
        workers=2,
        loop="asyncio",
        access_log=False,
        limit_concurrency=1000,
        limit_max_requests=1000
    )
