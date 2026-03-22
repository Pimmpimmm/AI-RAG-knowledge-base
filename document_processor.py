import os
from PyPDF2 import PdfReader
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
    
    def extract_text_from_pdf(self, pdf_file):
        text = ""
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    
    def extract_text_from_docx(self, docx_file):
        text = ""
        doc = Document(docx_file)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def process_document(self, uploaded_file):
        try:
            filename = uploaded_file.name.lower()
            if filename.endswith('.pdf'):
                text = self.extract_text_from_pdf(uploaded_file)
            elif filename.endswith('.docx'):
                text = self.extract_text_from_docx(uploaded_file)
            else:
                raise ValueError(f"不支持的文件格式: {filename}")
        except Exception as e:
            raise RuntimeError(f"处理文件 {uploaded_file.name} 时出错: {e}")
        
        if not text.strip():
            raise ValueError("文档内容为空，无法处理")
        
        chunks = self.text_splitter.split_text(text)
        return chunks
