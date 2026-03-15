import os
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document as LangchainDocument

class VectorStore:
    def __init__(self, pinecone_api_key, index_name, embedding_api_key, embedding_base_url, embedding_model, region="us-east-1", embedding_dim=1536):
        self.pinecone_api_key = pinecone_api_key
        self.index_name = index_name
        self.region = region
        self.embedding_dim = embedding_dim
        
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=embedding_api_key,
            base_url=embedding_base_url,
            model=embedding_model
        )
        self.vector_store = None
        self._initialize_index()
    
    def _initialize_index(self):
        # 检查索引是否存在，不存在则创建
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=self.region
                )
            )

        self.vector_store = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings,
            pinecone_api_key=self.pinecone_api_key
        )
    
    def add_documents(self, chunks, source_name):
        documents = [
            LangchainDocument(
                page_content=chunk,
                metadata={"source": source_name, "chunk": i}
            )
            for i, chunk in enumerate(chunks)
        ]
        self.vector_store.add_documents(documents)
    
    def similarity_search(self, query, k=4):
        return self.vector_store.similarity_search(query, k=k)
    
    def clear_index(self):
        try:
            self.pc.delete_index(self.index_name)
            # 等待删除完成（可选）
            import time
            time.sleep(2)
        except Exception as e:
            raise RuntimeError(f"删除索引失败: {e}")
        self._initialize_index()
