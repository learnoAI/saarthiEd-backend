# SaarthiEd Backend - Gemini 2.5 Flash Integration

This backend application processes student worksheets using Google Gemini 2.5 Flash for both OCR (text extraction) and intelligent scoring.

## Features

- **Image Processing**: Uses Gemini 2.5 Flash for OCR to extract questions and answers from worksheet images
- **Intelligent Scoring**: Uses Gemini 2.5 Flash as an LLM to score student answers against expected answers
- **Parallel Processing**: Processes multiple worksheets simultaneously for optimal performance
- **MongoDB Integration**: Saves scoring results and question-wise scores to MongoDB
- **S3 Storage**: Uploads original images to AWS S3 for archival

## Architecture

### OCR Pipeline
1. Upload worksheet image to S3
2. Use Gemini 2.5 Flash to extract text in structured JSON format
3. Parse questions and student answers

### Scoring Pipeline
1. Load expected answers from Results/dataset_questions_answers.json
2. Use Gemini 2.5 Flash to intelligently compare student answers with expected answers
3. Generate question-wise scores and overall score
4. Save results to MongoDB

## API Endpoints

### GET /
Health check endpoint

### GET /healthcheck
Returns service status and model information

### POST /process-worksheets
Process multiple student worksheets

**Parameters:**
- `token_no`: Student token number
- `worksheet_name`: Name of the worksheet
- `files`: List of image files (JPG, PNG, etc.)

**Response:**
```json
{
  "success": true,
  "processed_count": 5,
  "error_count": 0,
  "processed": [
    {
      "filename": "student1.jpg",
      "worksheet_name": "1001",
      "token_no": "12345",
      "s3_url": "https://...",
      "mongodb_id": "...",
      "overall_score": 35,
      "entries_count": 40,
      "question_scores": [...],
      "processed_with": "gemini-2.5-flash"
    }
  ],
  "errors": [],
  "model_used": "gemini-2.5-flash"
}
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Setup
1. Copy `.env.example` to `.env`
2. Fill in your actual credentials:
   - `MONGO_URI`: MongoDB connection string
   - `AWS_ACCESS_KEY_ID` & `AWS_SECRET_ACCESS_KEY`: AWS credentials
   - `GOOGLE_API_KEY`: Google Gemini API key

### 3. Get Google Gemini API Key
1. Visit [Google AI Studio](https://aistudio.google.com/)
2. Create a new API key
3. Add it to your `.env` file

### 4. Run the Application
```bash
python app.py
```

The server will start on `http://localhost:8080`

## Testing

Run the test script to verify Gemini integration:
```bash
python test_gemini_integration.py
```

## Key Changes from Previous Version

1. **Replaced Groq with Gemini 2.5 Flash**: Complete migration to Google's latest multimodal model
2. **Enhanced OCR**: Better text extraction with structured JSON output
3. **Intelligent Scoring**: LLM-based scoring instead of simple string matching
4. **MongoDB Integration**: Full database integration for result storage
5. **Improved Error Handling**: Better error messages and diagnostics
6. **Parallel Processing**: Maintained high-performance parallel processing

## File Structure

```
├── app.py                          # Main FastAPI application
├── utils.py                        # Gemini OCR and scoring functions
├── conns.py                        # Database and API connections
├── test_gemini_integration.py      # Test script
├── requirements.txt                # Dependencies
├── .env.example                    # Environment template
└── Results/
    └── dataset_questions_answers.json  # Expected answers database
```

## Performance Considerations

- **Parallel Processing**: Multiple worksheets processed simultaneously
- **Thread Pool**: Uses ThreadPoolExecutor for I/O bound operations
- **Image Optimization**: Automatic image format conversion and compression
- **Error Recovery**: Individual file failures don't stop batch processing

## MongoDB Schema

Documents are saved in the `worksheets` collection with the following structure:

```json
{
  "_id": "ObjectId",
  "token_no": "string",
  "worksheet_name": "string", 
  "filename": "string",
  "s3_url": "string",
  "overall_score": "number",
  "question_scores": [
    {
      "question_number": 1,
      "student_answer": "string",
      "expected_answer": "string", 
      "points_earned": 1,
      "points_possible": 1,
      "is_correct": true,
      "feedback": "string"
    }
  ],
  "timestamp": "ISODate",
  "processed_with": "gemini-2.5-flash"
}
```

## Error Handling

The system provides detailed error diagnostics:
- **Zero Score Analysis**: Explains why a student received zero marks
- **OCR Failures**: Reports text extraction issues
- **Scoring Errors**: Details scoring problems
- **File Upload Issues**: S3 upload error reporting

## Rate Limits

Google Gemini API has rate limits. The system handles these gracefully with:
- Automatic retry logic
- Error reporting
- Fallback mechanisms
