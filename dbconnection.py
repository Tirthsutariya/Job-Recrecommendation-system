from pymongo import MongoClient
import os
from dotenv import load_dotenv
load_dotenv() 

Mongo_URI = os.getenv('MONGO_URI')

def get_db():
    client = MongoClient(Mongo_URI)
    db = client["jobPortalDB"]
    return db
