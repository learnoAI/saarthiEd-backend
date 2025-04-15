from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import tempfile
import shutil
from utils import use_groq, extract_entries_from_response, upload_to_s3
from datetime import datetime
from conns import collection
import uvicorn
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a thread pool executor
executor = ThreadPoolExecutor(max_workers=5)

@app.get("/")
async def root():
    return {"message": "SaarthiEd API is running"}

async def process_single_file(file):
    """Process a single file in a separate thread"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp:
            shutil.copyfileobj(file.file, temp)
            temp_path = temp.name
        
        worksheet_name = os.path.splitext(file.filename)[0]
        
        # Run S3 upload in thread pool
        s3_url = await asyncio.get_event_loop().run_in_executor(
            executor, upload_to_s3, temp_path
        )
        
        if not s3_url:
            raise Exception(f"Failed to upload image to S3: {file.filename}")
        
        # Read image bytes
        with open(temp_path, "rb") as image_file:
            image_bytes = image_file.read()
        
        # Run Groq processing in thread pool
        gr_response = await asyncio.get_event_loop().run_in_executor(
            executor, use_groq, image_bytes
        )
        
        if "error" in gr_response:
            os.unlink(temp_path)
            return {"filename": file.filename, "error": gr_response["error"], "success": False}
        
        # Extract entries
        entries = extract_entries_from_response(gr_response)
        
        worksheet_doc = {
            "name": worksheet_name,
            "entries": entries,
            "processor": "groq",
            "model": "llama-4-scout-17b-16e-instruct",
            "processed_at": datetime.now(),
            "source_image": s3_url
        }
        
        # Run MongoDB insertion in thread pool
        await asyncio.get_event_loop().run_in_executor(
            executor, collection.insert_one, worksheet_doc
        )
        
        # Clean up temp file
        os.unlink(temp_path)
        
        return {
            "filename": file.filename,
            "worksheet_name": worksheet_name,
            "s3_url": s3_url,
            "entries_count": len(entries),
            "success": True
        }
            
    except Exception as e:
        try:
            os.unlink(temp_path)
        except:
            pass
        return {"filename": file.filename, "error": str(e), "success": False}

@app.post("/process-worksheets")
async def process_worksheets(files: List[UploadFile] = File(...)):
    """
    Upload and process student worksheet images concurrently.
    Returns the processed data and saves it to the database.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded")
    
    # Process all files concurrently
    tasks = [process_single_file(file) for file in files]
    results = await asyncio.gather(*tasks)
    
    # Separate successful results and errors
    processed = [r for r in results if r.get("success", False)]
    errors = [r for r in results if not r.get("success", False)]
    
    # Remove success field from processed results
    for r in processed:
        r.pop("success", None)
    
    return {
        "success": len(processed) > 0,
        "processed": processed,
        "errors": errors
    }

if __name__ == "__main__":
    uvicorn.run("app:app", port=8000, reload=True)
