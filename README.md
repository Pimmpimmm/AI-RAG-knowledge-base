# 企业级 AI 知识库 (RAG)

一个基于 Python + Streamlit + Pinecone 的企业级 AI 知识库系统。支持 OpenAI、DeepSeek 和豆包（火山引擎）等多个大模型提供商。

## 功能特性

- 📄 **文档上传**: 支持 PDF 和 Word 文档上传
- 🤖 **智能问答**: AI 仅根据文档内容回答问题
- 🔍 **来源追溯**: 显示答案的参考来源
- 💬 **对话界面**: 友好的聊天式交互
- 🌐 **多模型支持**: OpenAI、DeepSeek、豆包

## 技术栈

- **Python**: 后端编程语言
- **Streamlit**: 前端框架
- **Pinecone**: 向量数据库
- **LangChain**: RAG 框架
- **OpenAI/DeepSeek/豆包**: 嵌入模型和 LLM

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并填入你的 API Key：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```
API_KEY=你的-api-key
PINECONE_API_KEY=你的-pinecone-api-key
PINECONE_ENV=us-east-1
```

### 3. 运行应用

```bash
python -m streamlit run app.py
```

### 4. 使用步骤

1. 在侧边栏选择大模型提供商（OpenAI/DeepSeek/豆包）
2. 配置 API Key 并点击"初始化系统"
3. 上传 PDF 或 Word 文档
4. 点击"处理并上传文档"
5. 在对话框中提问，AI 将基于文档内容回答

## API Key 获取方式

### DeepSeek (推荐)
- 访问 https://platform.deepseek.com
- 注册并获取 API Key
- 新用户有免费额度

### 豆包 (火山引擎)
- 访问 https://console.volcengine.com/ark
- 创建豆包模型接入点
- 获取 API Key 和接入点 ID

### OpenAI
- 访问 https://platform.openai.com/api-keys
- 注册并获取 API Key

## 项目结构

```
AI RAG/
├── app.py                  # Streamlit 主应用
├── document_processor.py   # 文档处理模块
├── vector_store.py         # 向量数据库模块
├── rag_qa.py              # RAG 问答系统
├── requirements.txt       # 依赖列表
├── .env.example           # 环境变量示例
└── README.md              # 项目说明
```
