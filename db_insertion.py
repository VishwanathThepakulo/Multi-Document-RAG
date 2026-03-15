from motor.motor_asyncio import AsyncIOMotorClient
import time
import os
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

class DB_Connection():
    def __init__(self):
        db_credential = os.getenv("DB_CREDENTIALS")
        if not db_credential:
            raise ValueError("DB credentials not found")
        self.client = AsyncIOMotorClient(db_credential)
        self.collection = self.client['Multi-Document-RAG']['searchable_docs']
    
    
    
db_connection = DB_Connection()














