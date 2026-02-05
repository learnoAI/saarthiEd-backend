from pymongo import MongoClient
from dotenv import load_dotenv
import os
import boto3
from google import genai
from groq import Groq
from openai import OpenAI

load_dotenv(override=True)

# MongoDB connection
mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client["saarthiEd"]
collection = db["worksheets"]
qacollection = db["QAworksheets"]
qacomments_collection = db["QAcomments"]
error_logs_collection = db["error_logs"]

# AWS S3 connection
# s3_client = boto3.client(
#     service_name='s3',
#     region_name=os.getenv("AWS_DEFAULT_REGION"),
#     aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
#     aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
# )

#r2 client
r2_client = boto3.client(
    service_name='s3',
    endpoint_url=os.getenv("R2_API_URL"),
    aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
    region_name="auto"
)

# Gemini connection
gemini_client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

# OpenAI Client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Groq Client
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))