from motor.motor_asyncio import AsyncIOMotorClient
import time
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

class DB_Connection():
    def __init__(self):
        db_credential = os.getenv("DB_CREDENTIALS")
        if not db_credential:
            raise ValueError("DB credentials not found")
        self.client = AsyncIOMotorClient(db_credential)
        self.collection = self.client['Multi-Document-RAG']['searchable_docs']
        
    def get_embeddings():
        docs_to_insert = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        pass
        
    def embedding_and_inserting(self,file_path):
        loader = PyPDFLoader(file_path.path)
        data = loader.load()
        pass
    
    
    
    
db_connection = DB_Connection()
















