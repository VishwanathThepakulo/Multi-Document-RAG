from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from db_insertion import DB_Connection
import asyncio

load_dotenv()
app = FastAPI()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("API key not found")
def model_calling(query,context):
    model = init_chat_model(
        model='openai/gpt-oss-120b',
        model_provider='groq',
        groq_api_key=api_key,
        temperature=0.7,
        timeout=30,
        max_tokens=1000,
        max_retries=6,
    )
    prompt = f"""Answer the question using the context below.
    Context:
    {context}
    Question:
    {query}
    Your a rag application only answer if you have complete context.
    Answer clearly. if you dont have context just say 'i dont have context to answer your question'"""
    ai_response = model.invoke(prompt)
    return ai_response


async def user_question(question):
    db_connection = DB_Connection()
    embedding_result = db_connection.query_embedding(question)
    vector_search_result = await db_connection.vector_search(embedding_result)
    texts = []
    for doc in vector_search_result:
        print("-----------------------------------------------------------------------------\n")
        print(doc["text"][:200])
        print("-----------------------------------------------------------------------------")
        texts.append(doc['text'])
    context = "\n\n".join(texts)
    
    ai_reply = model_calling(question, context)
    # await db_connection.db_connection_close()
    return {
        'status': 200,
        'answer': ai_reply.content
    }
async def main():
    user_query = input("Enter a query : ")
    response = await user_question(user_query)
    print(response)
    
    
if __name__=='__main__':
    asyncio.run(main())
    # main()