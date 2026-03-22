from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

class RAGQASystem:
    def __init__(self, vector_store, llm_api_key, llm_base_url, 
                 llm_model, retrieval_k=4):
        self.vector_store = vector_store
        
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=0.1,
            openai_api_key=llm_api_key,
            base_url=llm_base_url
        )
        
        self.prompt = ChatPromptTemplate.from_template("""你是一个专业的企业知识库助手。请根据以下提供的文档内容来回答用户的问题。
            
            如果你在文档中找不到相关信息，请明确告诉用户"在文档中未找到相关信息"，不要编造答案。
            
            文档内容：
            {context}
            
            用户问题：{question}
            
            请用中文回答：""")
        
        self.retriever = self.vector_store.vector_store.as_retriever(search_kwargs={"k": retrieval_k})
        
        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def ask_question(self, question):
        result = self.rag_chain.invoke(question)
        
        source_docs = self.retriever.invoke(question)
        sources = []
        for doc in source_docs:
            sources.append({
                "source": doc.metadata.get("source", "未知"),
                "chunk": doc.metadata.get("chunk", 0),
                "content": doc.page_content[:200] + "..."
            })
        
        return result, sources
