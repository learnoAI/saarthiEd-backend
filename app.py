from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import tempfile
import shutil
from utils import use_groq, extract_entries_from_response, upload_to_s3
from datetime import datetime
from conns import collection
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("saarthied-api")

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
    logger.info("Root endpoint called")
    return {"message": "SaarthiEd API is running"}

@app.get("/healthcheck")
async def healthcheck():
    logger.info("Health check endpoint called")
    return {"message": "ok"}

@app.on_event("startup")
async def startup_event():
    port = os.environ.get("PORT", 8080)
    logger.info(f"Starting SaarthiEd API on port {port}")
    # Add other startup checks here if needed
    try:
        # Test MongoDB connection at startup
        logger.info("Testing MongoDB connection...")
        # Add any startup validations or connections here
        logger.info("All startup checks passed")
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        # Don't raise exception here, let the app try to start anyway

async def process_stud_worksheets(token_no, worksheet_name, file):
    temp_path = None
    try:
        logger.info(f"Processing worksheet: {file.filename} for token: {token_no}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp:
            shutil.copyfileobj(file.file, temp)
            temp_path = temp.name
        
        logger.info(f"Uploading to S3: {file.filename}")
        s3_url = await asyncio.get_event_loop().run_in_executor(
            executor, upload_to_s3, temp_path
        )
        
        if not s3_url:
            raise Exception(f"Failed to upload image to S3: {file.filename}")
        
        logger.info(f"Processing with Groq: {file.filename}")
        with open(temp_path, "rb") as image_file:
            image_bytes = image_file.read()
        
        gr_response = await asyncio.get_event_loop().run_in_executor(
            executor, use_groq, image_bytes
        )
        
        if "error" in gr_response:
            logger.error(f"Groq processing error: {gr_response['error']}")
            os.unlink(temp_path)
            return {"filename": file.filename, "error": gr_response["error"], "success": False}
        
        entries = extract_entries_from_response(gr_response)
        
        worksheet_doc = {
            "name": worksheet_name,
            "token_no": token_no,
            "entries": entries,
            "processor": "groq",
            "model": "llama-4-scout-17b-16e-instruct",
            "processed_at": datetime.now(),
            "source_image": s3_url
        }
        
        logger.info(f"Saving to MongoDB: {file.filename}")
        await asyncio.get_event_loop().run_in_executor(
            executor, collection.insert_one, worksheet_doc
        )
        
        os.unlink(temp_path)
        
        logger.info(f"Successfully processed: {file.filename}")
        return {
            "filename": file.filename,
            "worksheet_name": worksheet_name,
            "s3_url": s3_url,
            "entries_count": len(entries),
            "success": True
        }
            
    except Exception as e:
        logger.error(f"Error processing {file.filename}: {str(e)}")
        if temp_path:
            try:
                os.unlink(temp_path)
            except Exception as clean_err:
                logger.error(f"Error cleaning up temp file: {str(clean_err)}")
        return {"filename": file.filename, "error": str(e), "success": False}

@app.post("/process-worksheets")
async def process_worksheets(token_no: str, worksheet_name: str, files: List[UploadFile] = File(...)):
    logger.info(f"Process worksheets request: token={token_no}, worksheet={worksheet_name}, files={len(files)}")
    
    if not files:
        logger.warning("No files were uploaded")
        raise HTTPException(status_code=400, detail="No files were uploaded")
    
    if not token_no:
        logger.warning("token_no is required")
        raise HTTPException(status_code=400, detail="token_no is required")
    
    if not worksheet_name:
        logger.warning("worksheet_name is required")
        raise HTTPException(status_code=400, detail="worksheet_name is required")
    
    tasks = [process_stud_worksheets(token_no, worksheet_name, file) for file in files]
    results = await asyncio.gather(*tasks)
    
    processed = [r for r in results if r.get("success", False)]
    errors = [r for r in results if not r.get("success", False)]
    
    for r in processed:
        r.pop("success", None)
    
    logger.info(f"Processed {len(processed)} files with {len(errors)} errors")
    return {
        "success": len(processed) > 0,
        "processed": processed,
        "errors": errors
    }