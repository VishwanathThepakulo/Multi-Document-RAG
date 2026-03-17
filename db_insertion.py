from motor.motor_asyncio import AsyncIOMotorClient
import time
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
import asyncio

load_dotenv()

class DB_Connection():
    def __init__(self):
        db_credential = os.getenv("DB_CREDENTIALS")
        if not db_credential:
            raise ValueError("DB credentials not found")
        embedding_api_key = os.getenv('EMBEDDING_API_KEY')
        if not embedding_api_key:
            raise ValueError("Embedding key not found")
        self.client = AsyncIOMotorClient(db_credential)
        self.collection = self.client['multi_document_rag']['searchable_docs']
        self.embedding_model = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=embedding_api_key,
        )
        
    async def chunks_converter(self, data):
        docs_to_insert = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(data)
        texts = []
        for doc in documents:
            texts.append(doc.page_content)  
        embeddings = self.embedding_model.embed_documents(texts)    
        for text, emb in zip(texts, embeddings):
            item = {"text":text,'embeddings':emb}
            docs_to_insert.append(item)
        result = await self.uploading_into_db(docs_to_insert)
        return result
           
    async def pdf_loader(self,file_path):
        loader = PyPDFLoader(file_path)
        data = loader.load()
        print(type(data))
        result = await self.chunks_converter(data)
        return result
    
    async def uploading_into_db(self,docs_to_insert):
        result = await self.collection.insert_many(docs_to_insert)
        return {
            'status' : 200,
            "inserted_count":len(result.inserted_ids)
        }
        
    
    
    
    
db_connection = DB_Connection()

async def main():
    input_file_path = input("Enter File path :").strip('"')
    file_path = rf"{input_file_path}"
    result = await db_connection.pdf_loader(file_path)
    print(result)



if __name__=='__main__':
    asyncio.run(main())














