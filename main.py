#!/usr/bin/env python3
"""
Dynamic HackRx System - No Hardcoding, Works with Any Document and Questions
"""

import os
import time
import hashlib
import logging
import requests
import re
from datetime import datetime
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import google.generativeai as genai
import cohere
import pinecone
from io import BytesIO

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - Use environment variables for Azure deployment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_2xp1MM_TD5DpfJfkF1yhW2k3bjE6qppBMP7WakP4Cw8ppX8Hdh2PKw7uoJ3jb1sC4e65mK")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "hackrx-index")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyB41JhMyAVkx_m-Wu9StVHuHQUB1HJxlcQ")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "qZmghdKw7d7YxNryMj57OsMN0jLsQSCy0c7xulRA")

# Initialize services
genai.configure(api_key=GEMINI_API_KEY)
co = cohere.Client(COHERE_API_KEY)
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

app = FastAPI(title="HackRx Dynamic", version="1.0.0")

class DynamicDocumentProcessor:
    """Dynamic document processing for any file type"""
    
    @staticmethod
    def extract_text_from_url(url: str) -> str:
        """Extract text from any document type (PDF, DOCX, EML, TXT)"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            content = response.content
            content_type = response.headers.get('Content-Type', '').lower()
            ext = os.path.splitext(url.split('?')[0])[1].lower()
            
            # PDF Processing
            if ext == '.pdf' or 'pdf' in content_type:
                import fitz  # PyMuPDF
                doc = fitz.open(stream=content, filetype="pdf")
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                return text
            
            # DOCX Processing
            elif ext == '.docx' or 'word' in content_type or 'docx' in content_type:
                import docx
                doc = docx.Document(BytesIO(content))
                text = '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
                return text
            
            # EML Processing
            elif ext == '.eml' or 'message/rfc822' in content_type:
                from email import policy
                from email.parser import BytesParser
                msg = BytesParser(policy=policy.default).parsebytes(content)
                text = ''
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == 'text/plain':
                            text += part.get_content() + '\n'
                else:
                    text = msg.get_content()
                return text
            
            # Plain text fallback
            else:
                return content.decode(errors='ignore')
                
        except Exception as e:
            logger.error(f"Document extraction failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Document extraction failed: {str(e)}")

class DynamicChunker:
    """Dynamic text chunking for any document type"""
    
    @staticmethod
    def chunk_text(text: str) -> List[str]:
        """Intelligent chunking that adapts to document structure"""
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split by multiple strategies
        chunks = []
        
        # Strategy 1: Split by paragraphs
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            para = para.strip()
            if len(para) > 30:  # Minimum meaningful length
                chunks.append(para)
        
        # Strategy 2: If not enough chunks, split by sentences
        if len(chunks) < 10:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) > 30:
                    chunks.append(sent)
        
        # Strategy 3: If still not enough, split by fixed length
        if len(chunks) < 5:
            words = text.split()
            current_chunk = []
            for word in words:
                current_chunk.append(word)
                if len(' '.join(current_chunk)) > 200:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
            if current_chunk:
                chunks.append(' '.join(current_chunk))
        
        return chunks[:100]  # Limit to 100 chunks for performance

class DynamicAnswerGenerator:
    """Dynamic answer generation with intelligent fallback"""
    
    def __init__(self):
        self.gemini_model = genai.GenerativeModel('gemini-1.5-pro')
    
    def generate_answer(self, context: str, question: str) -> str:
        """Generate answer with intelligent context analysis"""
        
        # Try Gemini first
        try:
            prompt = f"""You are an expert document analyst. Answer the question based ONLY on the provided context.

Context: {context}

Question: {question}

Instructions:
- Answer accurately using only the provided context
- If information is not in context, say "Information not found in the document"
- Be comprehensive and detailed - provide full explanations
- Include specific details, numbers, percentages, time periods, and conditions when mentioned
- Format answers professionally with proper sentence structure
- Provide complete information from the context
- Use clear, detailed language similar to professional documentation
- Do not make assumptions beyond what is stated in the context
- Aim for comprehensive answers that fully address the question

Answer:"""
            
            response = self.gemini_model.generate_content(prompt)
            answer = response.text.strip()
            
            if answer and "error" not in answer.lower():
                return answer
            else:
                raise Exception("Gemini returned error")
                
        except Exception as e:
            logger.warning(f"Gemini failed: {str(e)}")
            
            # Fallback to Cohere
            try:
                response = co.generate(
                    model='command-r-plus',
                    prompt=f"""You are an expert document analyst. Answer the question based ONLY on the provided context.

Context: {context}

Question: {question}

Instructions:
- Answer accurately using only the provided context
- If information is not in context, say "Information not found in the document"
- Be comprehensive and detailed - provide full explanations
- Include specific details, numbers, percentages, time periods, and conditions when mentioned
- Format answers professionally with proper sentence structure
- Provide complete information from the context
- Use clear, detailed language similar to professional documentation
- Do not make assumptions beyond what is stated in the context
- Aim for comprehensive answers that fully address the question

Answer:""",
                    max_tokens=300,
                    temperature=0.1
                )
                return response.generations[0].text.strip()
                
            except Exception as e2:
                logger.error(f"Cohere also failed: {str(e2)}")
                return "Unable to process the question due to technical issues."

class DynamicHackRxSystem:
    """Dynamic HackRx system - works with any document and questions"""
    
    def __init__(self):
        self.doc_processor = DynamicDocumentProcessor()
        self.chunker = DynamicChunker()
        self.answer_generator = DynamicAnswerGenerator()
    
    def process_document_and_questions(self, doc_url: str, questions: List[str]) -> Dict[str, Any]:
        """Process any document and answer any questions dynamically"""
        start_time = time.time()
        
        try:
            # Extract text from any document type
            logger.info(f"Processing document: {doc_url}")
            text = self.doc_processor.extract_text_from_url(doc_url)
            
            # Generate unique document ID
            doc_id = hashlib.md5(doc_url.encode()).hexdigest()
            
            # Chunk text dynamically
            chunks = self.chunker.chunk_text(text)
            logger.info(f"Created {len(chunks)} chunks")
            
            # Store in Pinecone for semantic search
            try:
                for i, chunk in enumerate(chunks[:50]):  # Store first 50 chunks
                    embedding = genai.embed_content(
                        model='models/embedding-001',
                        content=chunk
                    )['embedding']
                    
                    index.upsert(
                        vectors=[(f"{doc_id}_chunk_{i}", embedding, {"doc_id": doc_id})]
                    )
            except Exception as e:
                logger.warning(f"Pinecone storage failed: {str(e)}")
            
            # Process questions dynamically
            answers = []
            
            for i, question in enumerate(questions):
                question_start = time.time()
                
                # Dynamic context selection based on question
                # Use semantic search if available, otherwise use intelligent chunk selection
                try:
                    # Try semantic search first
                    question_embedding = genai.embed_content(
                        model='models/embedding-001',
                        content=question
                    )['embedding']
                    
                    # Query Pinecone for relevant chunks
                    results = index.query(
                        vector=question_embedding,
                        top_k=8,
                        include_metadata=True
                    )
                    
                    # Extract relevant chunks
                    relevant_chunks = []
                    for match in results.matches:
                        chunk_id = match.id
                        chunk_index = int(chunk_id.split('_')[-1])
                        if chunk_index < len(chunks):
                            relevant_chunks.append(chunks[chunk_index])
                    
                    # If semantic search didn't find enough, add more chunks
                    if len(relevant_chunks) < 5:
                        relevant_chunks.extend(chunks[:8])
                    
                    context = "\n\n".join(relevant_chunks[:12])
                    
                except Exception as e:
                    logger.warning(f"Semantic search failed: {str(e)}")
                    # Fallback to intelligent chunk selection
                    context = "\n\n".join(chunks[:12])
                
                # Generate answer
                answer = self.answer_generator.generate_answer(context, question)
                answers.append(answer)
                
                question_time = time.time() - question_start
                logger.info(f"Question {i+1} processed in {question_time:.2f}s")
                
                # Small delay between questions to avoid rate limits
                if i < len(questions) - 1:
                    time.sleep(1)
            
            total_time = time.time() - start_time
            logger.info(f"Total processing time: {total_time:.2f}s")
            
            return {
                "answers": answers
            }
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# Initialize system
hackrx_system = DynamicHackRxSystem()

@app.post("/hackrx/run")
async def hackrx_run(request: Request):
    """Dynamic endpoint - works with any document and questions"""
    start_time = time.time()
    
    # Authentication
    auth = request.headers.get("authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized - Bearer token required")
    
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    
    # Validate input
    doc_url = data.get("documents")
    questions = data.get("questions")
    
    if not doc_url:
        raise HTTPException(status_code=400, detail="'documents' field is required")
    if not isinstance(questions, list) or not questions:
        raise HTTPException(status_code=400, detail="'questions' must be a non-empty list")
    
    # Process document and questions dynamically
    try:
        result = hackrx_system.process_document_and_questions(doc_url, questions)
        return JSONResponse(result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "HackRx API is running", "endpoint": "/hackrx/run"}

if __name__ == "__main__":
    import uvicorn
    # Use port 8000 for Azure Web App
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
