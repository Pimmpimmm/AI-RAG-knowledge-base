import os
import streamlit as st
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_qa import RAGQASystem

load_dotenv()

st.set_page_config(
    page_title="AI 智能知识库",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        .css-1d391kg {
            background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
        }
        h1 {
            color: #2c3e50;
            font-weight: 700;
            text-align: center;
            padding: 20px 0;
        }
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        .stSuccess {
            background-color: #d4edda;
            color: #155724;
            border-radius: 8px;
            padding: 12px;
        }
        .stError {
            background-color: #f8d7da;
            color: #721c24;
            border-radius: 8px;
            padding: 12px;
        }
        .stWarning {
            background-color: #fff3cd;
            color: #856404;
            border-radius: 8px;
            padding: 12px;
        }
        .stInfo {
            background-color: #d1ecf1;
            color: #0c5460;
            border-radius: 8px;
            padding: 12px;
        }
        .css-12oz5g7 {
            padding: 1rem;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .user-message {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .assistant-message {
            background-color: white;
            color: #2c3e50;
        }
        .sidebar .stSelectbox {
            margin-bottom: 1rem;
        }
        .divider {
            margin: 1.5rem 0;
            border-top: 2px solid #e9ecef;
        }
        .card {
            background-color: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🤖 AI 智能知识库</h1>", unsafe_allow_html=True)

st.sidebar.markdown("<h2 style='color: #2c3e50; text-align: center;'>⚙️ 系统配置</h2>", unsafe_allow_html=True)

llm_provider = st.sidebar.selectbox(
    "选择大模型提供商",
    ["OpenAI", "DeepSeek", "火山方舟 (豆包)"],
    index=2
)

llm_api_key = st.sidebar.text_input(
    "API Key",
    type="password",
    value=os.getenv("API_KEY", ""),
    help="请输入您的 API Key"
)

pinecone_api_key = st.sidebar.text_input(
    "Pinecone API Key",
    type="password",
    value=os.getenv("PINECONE_API_KEY", ""),
    help="请输入您的 Pinecone API Key"
)

pinecone_env = st.sidebar.text_input(
    "Pinecone 环境",
    value=os.getenv("PINECONE_ENV", "us-east-1"),
    help="Pinecone 索引所在的环境"
)

index_name = st.sidebar.text_input(
    "Pinecone 索引名称",
    value="enterprise-knowledge-base",
    help="Pinecone 向量数据库索引名称"
)

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
        "embedding_model": "deepseek-embed",
        "embedding_dim": 1536
    },
    "火山方舟 (豆包)": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "chat_model": "doubao-seed-1-8-251228",
        "embedding_model": "ep-20260322114942-pc4k5",
        "embedding_dim": 2048
    }
}

config = LLM_CONFIGS[llm_provider]

if llm_provider == "火山方舟 (豆包)":
    custom_chat_model = st.sidebar.text_input(
        "豆包接入点 ID", 
        value=config["chat_model"],
        help="在火山方舟控制台创建的接入点 ID"
    )
    
    config["chat_model"] = custom_chat_model

    # 添加嵌入模型配置选项
    custom_embedding_model = st.sidebar.text_input(
        "嵌入模型", 
        value=config["embedding_model"],
        help="火山方舟支持的嵌入模型"
    )
    config["embedding_model"] = custom_embedding_model

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "qa_system" not in st.session_state:
    st.session_state.qa_system = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents_uploaded" not in st.session_state:
    st.session_state.documents_uploaded = False

st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True)

col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("🚀 初始化系统", use_container_width=True):
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
                        region=pinecone_env,
                        embedding_dim=config["embedding_dim"],
                        provider=llm_provider
                    )
                    st.session_state.qa_system = RAGQASystem(
                        vector_store=st.session_state.vector_store,
                        llm_api_key=llm_api_key,
                        llm_base_url=config["base_url"],
                        llm_model=config["chat_model"]
                    )
                    st.sidebar.success("✅ 系统初始化成功！")
                except Exception as e:
                    st.sidebar.error(f"初始化失败: {str(e)}")

with col2:
    if st.button("🔄 重置系统", use_container_width=True):
        st.session_state.vector_store = None
        st.session_state.qa_system = None
        st.session_state.messages = []
        st.session_state.documents_uploaded = False
        st.sidebar.info("系统已重置！")

st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True)

col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.sidebar.success("对话历史已清空！")

with col2:
    if st.button("💾 清空知识库", use_container_width=True):
        if st.session_state.vector_store:
            with st.spinner("正在清空知识库..."):
                st.session_state.vector_store.clear_index()
                st.session_state.documents_uploaded = False
                st.sidebar.success("知识库已清空！")
        else:
            st.sidebar.warning("请先初始化系统！")

st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True)

with st.sidebar.expander("📖 使用说明"):
    st.info("""
    **快速开始：**
    1. 选择大模型提供商
    2. 配置 API Key 和 Pinecone API Key
    3. 点击「初始化系统」
    4. 上传文档并处理
    5. 开始智能问答
    
    **支持的模型：**
    - 🔥 OpenAI (GPT-3.5/4)
    - 🚀 DeepSeek
    - 🤖 火山方舟 (豆包)
    """)

with st.sidebar.expander("🔗 相关链接"):
    st.markdown("""
    - [火山方舟控制台](https://console.volcengine.com/ark)
    - [DeepSeek 平台](https://platform.deepseek.com)
    - [OpenAI 平台](https://platform.openai.com)
    - [Pinecone 控制台](https://app.pinecone.io)
    """)

st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True)

st.sidebar.markdown("""
    <div style='text-align: center; padding: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 8px; color: white;'>
        <p style='margin: 0; font-size: 12px;'>Made with ❤️ by AI RAG</p>
    </div>
""", unsafe_allow_html=True)

main_col1, main_col2 = st.columns([1, 1])

with main_col1:
    st.markdown("""
        <div class="card">
            <h3 style='color: #2c3e50; margin-top: 0;'>📤 文档上传</h3>
            <p style='color: #6c757d; margin-bottom: 1rem;'>上传 PDF 或 Word 文档到知识库</p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "选择文档",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files and st.session_state.vector_store:
        if st.button("✨ 处理并上传文档", use_container_width=True):
            doc_processor = DocumentProcessor()
            total_chunks = 0
            
            with st.spinner("正在处理文档，请稍候..."):
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
            st.markdown(f"""
                <div class="stSuccess" style='margin-top: 1rem;'>
                    <strong>🎉 所有文档处理完成！</strong><br>
                    共生成 <strong>{total_chunks}</strong> 个片段
                </div>
            """, unsafe_allow_html=True)
    
    if not st.session_state.vector_store:
        st.warning("⚠️ 请先在左侧侧边栏初始化系统")

with main_col2:
    st.markdown("""
        <div class="card">
            <h3 style='color: #2c3e50; margin-top: 0;'>💬 智能问答</h3>
            <p style='color: #6c757d; margin-bottom: 1rem;'>基于文档内容回答您的问题</p>
        </div>
    """, unsafe_allow_html=True)
    
    for message in st.session_state.messages:
        message_class = "user-message" if message["role"] == "user" else "assistant-message"
        avatar = "👤" if message["role"] == "user" else "🤖"
        st.markdown(f"""
            <div class="chat-message {message_class}">
                <div style='font-weight: 600; margin-bottom: 0.5rem;'>{avatar} {'用户' if message["role"] == "user" else 'AI 助手'}</div>
                <div>{message['content']}</div>
            </div>
        """, unsafe_allow_html=True)
    
    if prompt := st.chat_input("请输入您的问题..."):
        if not st.session_state.qa_system:
            st.error("⚠️ 请先在左侧侧边栏初始化系统！")
        elif not st.session_state.documents_uploaded:
            st.warning("📝 请先上传文档！")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.markdown(f"""
                <div class="chat-message user-message">
                    <div style='font-weight: 600; margin-bottom: 0.5rem;'>👤 用户</div>
                    <div>{prompt}</div>
                </div>
            """, unsafe_allow_html=True)
            
            with st.spinner("🤖 正在思考..."):
                try:
                    answer, sources = st.session_state.qa_system.ask_question(prompt)
                    st.markdown(f"""
                        <div class="chat-message assistant-message">
                            <div style='font-weight: 600; margin-bottom: 0.5rem;'>🤖 AI 助手</div>
                            <div>{answer}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if sources:
                        with st.expander("📖 查看参考来源"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**来源 {i}**: {source['source']} (片段 {source['chunk']})")
                                st.markdown(f"> {source['content']}")
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"回答生成失败: {str(e)}")
                    answer = f"抱歉，出现了错误: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": answer})
