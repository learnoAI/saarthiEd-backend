from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import tempfile
import shutil
from utils import use_groq, extract_entries_from_response, upload_to_s3
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
    return {"message": "SaarthiEd API is running"}

@app.get("/healthcheck")
async def healthcheck():
    return {"message": "ok"}

def evaluate_answers(worksheet_name, entries):
    print(f"Evaluating worksheet: {worksheet_name} with {len(entries)} entries")
    marks = 0
    total_questions = 40
    evaluated_entries = []
    error_message = None
    
    worksheet_range = None
    for range_key in EXPECTED_ANSWERS.keys():
        if worksheet_name in EXPECTED_ANSWERS[range_key]:
            worksheet_range = range_key
            print(f"Found worksheet {worksheet_name} in range {range_key}")
            break
    
    if worksheet_range is None:
        error_message = f"No expected answers found for worksheet {worksheet_name}"
        print(f"ERROR: {error_message}")
        return {
            "marks": 0,
            "total_questions": total_questions,
            "evaluated_entries": entries,
            "error": error_message
        }
    
    expected_answers = EXPECTED_ANSWERS[worksheet_range].get(worksheet_name, [])
    
    if not expected_answers:
        error_message = f"No expected answers found for worksheet {worksheet_name} in range {worksheet_range}"
        print(f"ERROR: {error_message}")
        return {
            "marks": 0,
            "total_questions": total_questions,
            "evaluated_entries": entries,
            "error": error_message
        }
    
    print(f"Found {len(expected_answers)} expected answers for worksheet {worksheet_name}")
    
    entries_by_id = {}
    for entry in entries:
        question_id = entry['question_id'].lower().strip()
        numeric_id = ''.join(filter(str.isdigit, question_id))
        if numeric_id:
            try:
                idx = int(numeric_id) - 1
                entries_by_id[idx] = entry
            except ValueError as e:
                print(f"ERROR parsing numeric ID from {question_id}: {e}")
    
    actual_question_count = len(expected_answers)
    
    correct_answers = 0
    for i, expected in enumerate(expected_answers):
        if i in entries_by_id:
            entry = entries_by_id[i]
            expected_answer = expected.get('answer', '')
            student_answer = entry['answer'].strip()
            
            is_correct = student_answer == expected_answer
            
            print(f"Q{i+1}: Student: '{student_answer}' vs Expected: '{expected_answer}' => {is_correct}")
            
            if is_correct:
                correct_answers += 1
            
            evaluated_entry = entry.copy()
            evaluated_entry['expected_answer'] = expected_answer
            evaluated_entry['is_correct'] = is_correct
            evaluated_entries.append(evaluated_entry)
        else:
            print(f"Q{i+1}: Missing student answer, expected: '{expected.get('answer', '')}'")
            evaluated_entries.append({
                'question_id': f'q{i+1}',
                'question': f'Question {i+1}',
                'answer': '',
                'expected_answer': expected.get('answer', ''),
                'is_correct': False
            })
    
    if actual_question_count > 0:
        marks = round((correct_answers / actual_question_count) * total_questions)
    
    print(f"Raw score: {correct_answers}/{actual_question_count}")
    print(f"Final score (out of 40): {marks}/{total_questions}")
    
    return {
        "marks": marks,
        "total_questions": total_questions,
        "evaluated_entries": evaluated_entries,
        "error": error_message
    }

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
        
        evaluation_result = evaluate_answers(worksheet_name, entries)
        
        os.unlink(temp_path)
        
        result = {
            "filename": file.filename,
            "worksheet_name": worksheet_name,
            "s3_url": s3_url,
            "entries_count": len(entries),
            "marks": evaluation_result["marks"],
            "total_questions": 40,
            "success": True
        }
        
        if evaluation_result["marks"] == 0:
            result["zero_marks_reason"] = evaluation_result.get("error") or "None of the answers matched the expected answers"
            if "error" not in evaluation_result:
                incorrect_answers = [
                    f"Q{idx+1}: answered '{entry.get('answer', '')}' instead of '{entry.get('expected_answer', '')}'"
                    for idx, entry in enumerate(evaluation_result["evaluated_entries"])
                    if not entry.get("is_correct", False) and entry.get("answer", "").strip() != ""
                ]
                unanswered = sum(1 for entry in evaluation_result["evaluated_entries"] 
                                if entry.get("answer", "").strip() == "")
                
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

if __name__ == "__main__":
    import uvicorn 
    uvicorn.run("app:app", port=8080, reload=True)
