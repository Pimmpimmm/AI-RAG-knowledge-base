import os
import streamlit as st
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_qa import RAGQASystem

load_dotenv()

st.set_page_config(page_title="企业 AI 知识库", page_icon="📚", layout="wide")

st.title("📚 企业级 AI 知识库")

st.sidebar.header("配置设置")

llm_provider = st.sidebar.selectbox(
    "选择大模型提供商",
    ["OpenAI", "DeepSeek", "火山方舟 (豆包)"],
    index=2
)

llm_api_key = st.sidebar.text_input("API Key", type="password", value=os.getenv("API_KEY", ""))
pinecone_api_key = st.sidebar.text_input("Pinecone API Key", type="password", value=os.getenv("PINECONE_API_KEY", ""))
pinecone_env = st.sidebar.text_input("Pinecone 环境", value=os.getenv("PINECONE_ENV", "us-east-1"))
index_name = st.sidebar.text_input("Pinecone 索引名称", value="enterprise-knowledge-base")

LLM_CONFIGS = {
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "chat_model": "gpt-3.5-turbo",
        "embedding_model": "text-embedding-3-small",
        "embedding_dim": 1536
    },
    "DeepSeek": {
        "base_url": "https://api.deepseek.com/v1",
        "chat_model": "deepseek-chat",
        "embedding_model": "text-embedding-ada-002",
        "embedding_dim": 1536
    },
    "火山方舟 (豆包)": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "chat_model": "ep-20241203005209-8v284",
        "embedding_model": "text-embedding-3-small",
        "embedding_dim": 1536
    }
}

# 初始化时传入 embedding_dim
st.session_state.vector_store = VectorStore(
    pinecone_api_key=pinecone_api_key,
    pinecone_env=pinecone_env,
    index_name=index_name,
    embedding_api_key=llm_api_key,
    embedding_base_url=config["base_url"],
    embedding_model=config["embedding_model"],
    embedding_dim=config["embedding_dim"]  # 新增参数
)

config = LLM_CONFIGS[llm_provider]

if llm_provider == "火山方舟 (豆包)":
    custom_chat_model = st.sidebar.text_input(
        "豆包接入点 ID", 
        value=config["chat_model"],
        help="在火山方舟控制台创建的接入点 ID"
    )
    config["chat_model"] = custom_chat_model

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "qa_system" not in st.session_state:
    st.session_state.qa_system = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents_uploaded" not in st.session_state:
    st.session_state.documents_uploaded = False

st.sidebar.markdown("---")

if st.sidebar.button("初始化系统"):
    if not llm_api_key or not pinecone_api_key:
        st.sidebar.error("请先配置 API Key 和 Pinecone API Key")
    else:
        with st.spinner("正在初始化系统..."):
            try:
                st.session_state.vector_store = VectorStore(
                    pinecone_api_key=pinecone_api_key,
                    index_name=index_name,
                    embedding_api_key=llm_api_key,
                    embedding_base_url=config["base_url"],
                    embedding_model=config["embedding_model"],
                    region=pinecone_env
                )
                st.session_state.qa_system = RAGQASystem(
                    vector_store=st.session_state.vector_store,
                    llm_api_key=llm_api_key,
                    llm_base_url=config["base_url"],
                    llm_model=config["chat_model"]
                )
                st.sidebar.success("系统初始化成功！")
            except Exception as e:
                st.sidebar.error(f"初始化失败: {str(e)}")

st.sidebar.markdown("---")

st.header("📤 文档上传")

uploaded_files = st.file_uploader("上传 PDF 或 Word 文档", type=["pdf", "docx"], accept_multiple_files=True)

if uploaded_files and st.session_state.vector_store:
    if st.button("处理并上传文档"):
        doc_processor = DocumentProcessor()
        total_chunks = 0
        
        with st.spinner("正在处理文档..."):
            progress_bar = st.progress(0)
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    chunks = doc_processor.process_document(uploaded_file)
                    st.session_state.vector_store.add_documents(chunks, uploaded_file.name)
                    total_chunks += len(chunks)
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    st.success(f"✅ {uploaded_file.name} 处理完成，切分为 {len(chunks)} 个片段")
                except Exception as e:
                    st.error(f"❌ 处理 {uploaded_file.name} 失败: {str(e)}")
        
        st.session_state.documents_uploaded = True
        st.success(f"🎉 所有文档处理完成！共生成 {total_chunks} 个片段")

st.markdown("---")

st.header("💬 智能问答")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("请输入您的问题..."):
    if not st.session_state.qa_system:
        st.error("请先在侧边栏初始化系统！")
    elif not st.session_state.documents_uploaded:
        st.warning("请先上传文档！")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("正在思考..."):
                try:
                    answer, sources = st.session_state.qa_system.ask_question(prompt)
                    st.markdown(answer)
                    
                    if sources:
                        with st.expander("📖 查看参考来源"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**来源 {i}**: {source['source']} (片段 {source['chunk']})")
                                st.markdown(f"> {source['content']}")
                except Exception as e:
                    st.error(f"回答生成失败: {str(e)}")
                    answer = f"抱歉，出现了错误: {str(e)}"
        
        st.session_state.messages.append({"role": "assistant", "content": answer})

st.sidebar.markdown("---")
if st.sidebar.button("清空对话历史"):
    st.session_state.messages = []
    st.sidebar.success("对话历史已清空！")

if st.sidebar.button("清空知识库"):
    if st.session_state.vector_store:
        with st.spinner("正在清空知识库..."):
            st.session_state.vector_store.clear_index()
            st.session_state.documents_uploaded = False
            st.sidebar.success("知识库已清空！")
    else:
        st.sidebar.warning("请先初始化系统！")

st.sidebar.markdown("---")
st.sidebar.info("""
💡 **提示**:
- 火山方舟: 需要在 https://console.volcengine.com/ark 创建接入点
- DeepSeek: API Key 从 https://platform.deepseek.com 获取
- OpenAI: API Key 从 https://platform.openai.com 获取
""")
