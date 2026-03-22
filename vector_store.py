import os
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document as LangchainDocument
from typing import List

class VolcanoEmbeddings:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        try:
            from volcenginesdkarkruntime import Ark
            self.client = Ark(api_key=api_key)
        except ImportError:
            raise ImportError("请安装火山方舟 SDK: pip install volcengine-python-sdk")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            try:
                resp = self.client.multimodal_embeddings.create(
                    model=self.model,
                    input=[{"type": "text", "text": text}]
                )
                if resp.data:
                    embeddings.append(resp.data.embedding)
                else:
                    raise ValueError("嵌入响应为空")
            except Exception as e:
                raise RuntimeError(f"火山方舟嵌入失败: {str(e)}")
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        try:
            resp = self.client.multimodal_embeddings.create(
                model=self.model,
                input=[{"type": "text", "text": text}]
            )
            if resp.data:
                return resp.data.embedding
            else:
                raise ValueError("嵌入响应为空")
        except Exception as e:
            raise RuntimeError(f"火山方舟嵌入失败: {str(e)}")

class VectorStore:
    def __init__(self, pinecone_api_key, index_name, embedding_api_key, embedding_base_url, embedding_model, region="us-east-1", embedding_dim=1536, provider="OpenAI"):
        self.pinecone_api_key = pinecone_api_key
        self.index_name = index_name
        self.region = region
        self.embedding_dim = embedding_dim
        self.provider = provider
        
        self.pc = Pinecone(api_key=pinecone_api_key)
        
        if provider == "火山方舟 (豆包)":
            self.embeddings = VolcanoEmbeddings(
                api_key=embedding_api_key,
                model=embedding_model
            )
        else:
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=embedding_api_key,
                base_url=embedding_base_url,
                model=embedding_model
            )
        
        self.vector_store = None
        self._initialize_index()
    
    def _initialize_index(self):
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
            import time
            time.sleep(2)
        except Exception as e:
            raise RuntimeError(f"删除索引失败: {e}")
        self._initialize_index()
