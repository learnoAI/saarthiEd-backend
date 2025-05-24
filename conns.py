from pymongo import MongoClient
from dotenv import load_dotenv
import os
import boto3
from google import genai

load_dotenv()

# MongoDB connection
mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client["saarthiEd"]
collection = db["worksheets"]

# AWS S3 connection
s3_client = boto3.client(
    service_name='s3',
    region_name=os.getenv("AWS_DEFAULT_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

# Gemini connection
gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))