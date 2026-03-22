# retrieval_logic.py
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
from db_insertion import DB_Connection

load_dotenv()

class ModelCaller:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env")
            
        self.llm = init_chat_model(
            model="openai/gpt-oss-120b", 
            model_provider='groq',
            api_key=api_key,
            temperature=0.0,                  # usually better for RAG
            max_tokens=1500,
            max_retries=4,
            # reasoning_format="parsed"       # if you want structured output later
        )
        # self.db_connection = DB_Connection()

    async def generate(self, query: str, context: str) -> str:
        prompt = f"""You are a precise RAG assistant. Answer only using the provided context.
If the context does not contain enough information, reply exactly: "I don't have enough context to answer."

Context:
{context}

Question: {query}

Answer:"""

        response = await self.llm.ainvoke(prompt)
        return response.content.strip()

    async def user_question(self,question):
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
        
        ai_reply = await self.generate(question, context)
        # await db_connection.db_connection_close()
        return {
            'status': 200,
            'answer': ai_reply
        }

async def main():
    model_calling = ModelCaller()
    user_query = input("Enter a query : ")
    response = await model_calling.user_question(user_query)
    print(response)
        
        
if __name__=='__main__':
    asyncio.run(main())
    # main()