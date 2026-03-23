from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
import asyncio
from pymongo.operations import SearchIndexModel
import time
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.retrievers.hybrid_search import MongoDBAtlasHybridSearchRetriever
from pymongo import MongoClient
import logging

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

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
        self.sync_client = MongoClient(db_credential)
        self.sync_collection = self.sync_client['multi_document_rag']['searchable_docs']

        
        
    def reranking_vectors(self):
        vector_store = MongoDBAtlasVectorSearch(
            collection = self.sync_collection,
            embedding = self.embedding_model,
            index_name = "vector_index",
            embedding_key = "embeddings",
            text_key = 'text'
        )
        hybrid_retriever = MongoDBAtlasHybridSearchRetriever(
            vectorstore=vector_store,
            search_index_name="fulltext_index",      # ← your BM25/Atlas Search index name
            top_k=8,                                # candidates before fusion
            fulltext_penalty=40,                     # tune weights
            vector_penalty=60,
        )
        print(hybrid_retriever)
        return hybrid_retriever
    
    
    async def embeddings_to_insert_in_db(self, data):
        docs_to_insert = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(data)
        texts = []
        for doc in documents:
            texts.append(doc.page_content)  
            
        max_retries = 3
        embeddings = None
        for retry in range(max_retries):
            try:    
                embeddings = self.embedding_model.embed_documents(texts) 
                logger.info("Successfully generated embeddings")
                break
            except Exception as e:
                logger.warning(f"Embeddings attempt {retry+1} failed : {e}")
                if retry<max_retries-1:
                    await asyncio.sleep(2 ** retry)
                else:
                    logger.error("All embeddings retry failed")
                    return None
        if embeddings:
            for text, emb in zip(texts, embeddings):
                item = {"text":text,'embeddings':emb}
                docs_to_insert.append(item)
            result = await self.uploading_into_db(docs_to_insert)
            return result
        return None
    
    def query_embedding(self,query):
        embedding = self.embedding_model.embed_query(query)
        return embedding
           
    
    async def pdf_loader(self,file_path):
        try:
            loader = PyPDFLoader(file_path)
            data = loader.load()
            print(type(data))
            result = await self.embeddings_to_insert_in_db(data)
            return result
        except Exception as e:
            logger.info(f"Pdf upload failed {e}")
        
    async def uploading_into_db(self,docs_to_insert):
        result = await self.collection.insert_many(docs_to_insert)
        return {
            'status' : 200,
            "inserted_count":len(result.inserted_ids)
        }
        
    async def create_vector_index(self):
        existing = [idx async for idx in self.collection.list_search_indexes()]
        for idx in existing:
            if idx["name"] == "vector_index":
                if idx.get("queryable"):
                    print("✅ Index already READY")
                    return
                else:
                    print("⏳ Index exists but not ready, waiting...")
        search_index_model = SearchIndexModel(
            definition={
                "fields": [
                    {
                        "type": "vector",
                        "path": "embeddings",
                        "numDimensions": 384,
                        "similarity": "cosine"
                    }
                ]
            },
            name="vector_index",
            type="vectorSearch"
        )
        result = await self.collection.create_search_index(model=search_index_model)
        print(f"Creating index: {result}")
        max_attempts = 20
        attempt = 0
        while attempt < max_attempts:
            indexes = [idx async for idx in self.collection.list_search_indexes()]

            for idx in indexes:
                if idx["name"] == result and idx.get("queryable"):
                    print("✅ Index READY")
                    return

            print("⏳ Building index...")
            await asyncio.sleep(5)
            attempt += 1

        print("⚠️ Index creation timeout")

    async def vector_search(self, query):
        pipeline = [
            {
            '$vectorSearch':{
                'index':'vector_index',
                'path':'embeddings',
                'queryVector':query,
                'numCandidates':100,
                'limit': 10
            }
            },{
            '$project':{
                '_id':0,
                'text':1,
                "score": {"$meta": "vectorSearchScore"}
            }
            }
        ]
        result = self.collection.aggregate(pipeline)
        results = [doc async for doc in result]
        return results

    async def db_connection_close(self):
        await self.client.close()
        


db_connection = DB_Connection()

async def main():
    input_file_path = input("Enter File path :").strip('"')
    file_path = rf"{input_file_path}"
    result = await db_connection.pdf_loader(file_path)
    print(result)
    await db_connection.create_vector_index()



if __name__=='__main__':
    asyncio.run(main())














