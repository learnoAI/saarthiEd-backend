from pymongo import MongoClient
from dotenv import load_dotenv
import os
import boto3
from groq import Groq

load_dotenv()

mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client["saarthiEd"]
collection = db["worksheets"]

s3_client = boto3.client(
    service_name='s3',
    region_name=os.getenv("AWS_DEFAULT_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))