# retrieval_logic.py
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
from db_insertion import DB_Connection
import asyncio
import logging 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
load_dotenv()

logger = logging.getLogger(__name__)

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
        self.db_connection = DB_Connection()
        self.retriever = self.db_connection.reranking_vectors()

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
        # retriever = self.db_connection.reranking_vectors()
        
        # docs = db_connection.hybrid_retriever.invoke(question)
        # if not docs:
        #         return {"status": 200, "answer": "No relevant context found."}
        # texts = [doc.page_content for doc in docs]
        
        docs = await asyncio.get_event_loop().run_in_executor(
            None,
            self.retriever.invoke,
            question
        )
        
        # filtered_docs = [doc for doc in docs if docs.metadata.get("score",0)>0.5]
        # filtered_docs = [doc for doc in docs if doc.metadata.get("score", 0) > 0.5]
        # logger.info(f"================\n{filtered_docs}")
        
        if not docs:
                return {"status": 200, "answer": "No relevant context found."}
        texts = [doc.page_content for doc in docs if doc.metadata.get('score')>0.015]
        
        if not texts:
            logger.info("No relative chunk found")
            return None
        
        i = 0
        for doc in docs:
            i+=1
            # print(f'{i}\t{doc.page_content}')
            logger.debug(doc.metadata)
            logger.debug('============================')
            logger.debug(doc.metadata.get('score'))
            logger.debug("----------------------")
    
        # embedding_result = db_connection.query_embedding(question)
        # vector_search_result = await db_connection.vector_search(embedding_result)
        # texts = []
        # for doc in vector_search_result:
        #     print("-----------------------------------------------------------------------------\n")
        #     print(doc["text"][:200])
        #     print("-----------------------------------------------------------------------------")
        #     texts.append(doc['text'])
        context = "\n\n".join(texts)
        
        sources  = list(set([
            f"{doc.metadata.get('source')} - Page {doc.metadata.get('page')}" 
            for doc in docs if doc.metadata.get('score') > 0.015
        ]))
        
        
        ai_reply = await self.generate(question, context)
        # await db_connection.db_connection_close()
        logger.info(f"AI response ========\n {ai_reply} \n source is =============\n {sources}")
        return {
            'status': 200,
            'answer': ai_reply,
            'sources' : sources 
        }

async def main():
    model_calling = ModelCaller()
    user_query = input("Enter a query : ")
    response = await model_calling.user_question(user_query)
    print(response)
        
        
if __name__=='__main__':
    asyncio.run(main())
    # main()