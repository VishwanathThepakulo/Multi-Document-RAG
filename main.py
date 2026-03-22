# Multi-Document Enterprise RAG (MongoDB + Pinecone)
# Similar to RAG, but improve:
# Hybrid search (keyword + vector)
# Role-based access control
# Document versioning
# Metadata filtering

from pydantic import BaseModel
from fastapi import FastAPI
from db_insertion import DB_Connection
from retrieval_logic import ModelCaller

app = FastAPI()

class Validating_user_query(BaseModel):
    query:str
    
class Validating_db_insertion(BaseModel):
    path: str     # ← renamed for clarity

db_connection = DB_Connection()
model_caller = ModelCaller()

@app.post('/document/insertion/db')
async def emb_insertion_in_db(body:Validating_db_insertion):
    result = await db_connection.pdf_loader(body.path)
    print(result)
    await db_connection.create_vector_index()
    return {"status": 200, "inserted_count": result.get("inserted_count", 0)}


@app.post('/query')
async def user_query_function(query:Validating_user_query):
    result = await model_caller.user_question(query.query)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",              
        host='localhost',
        port=9000
        # reload=True             
    )










