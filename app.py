from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import tempfile
import shutil
from utils import use_groq, extract_entries_from_response, upload_to_s3
from datetime import datetime
from conns import collection
# import uvicorn
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

executor = ThreadPoolExecutor(max_workers=5)

@app.get("/")
async def root():
    return {"message": "SaarthiEd API is running"}

@app.get("/healthcheck")
async def healthcheck():
    return {"message": "ok"}

async def process_stud_worksheets(token_no, worksheet_name, file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp:
            shutil.copyfileobj(file.file, temp)
            temp_path = temp.name
        
        
        s3_url = await asyncio.get_event_loop().run_in_executor(
            executor, upload_to_s3, temp_path
        )
        
        if not s3_url:
            raise Exception(f"Failed to upload image to S3: {file.filename}")
        
        with open(temp_path, "rb") as image_file:
            image_bytes = image_file.read()
        
        gr_response = await asyncio.get_event_loop().run_in_executor(
            executor, use_groq, image_bytes
        )
        
        if "error" in gr_response:
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
        
        await asyncio.get_event_loop().run_in_executor(
            executor, collection.insert_one, worksheet_doc
        )
        
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
async def process_worksheets(token_no: str, worksheet_name: str, files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded")
    
    if not token_no:
        raise HTTPException(status_code=400, detail="token_no is required")
    
    if not worksheet_name:
        raise HTTPException(status_code=400, detail="worksheet_name is required")
    
    tasks = [process_stud_worksheets(token_no, worksheet_name, file) for file in files]
    results = await asyncio.gather(*tasks)
    
    processed = [r for r in results if r.get("success", False)]
    errors = [r for r in results if not r.get("success", False)]
    
    for r in processed:
        r.pop("success", None)
    
    return {
        "success": len(processed) > 0,
        "processed": processed,
        "errors": errors
    }

# if __name__ == "__main__":
#     uvicorn.run("app:app", port=8080, reload=True)
