# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

import time
from typing import List, Optional, Union
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from service.local_embedding import LocalEmbeddings
from service.reranker import DefaultReranker, QwenReranker
from langchain_core.documents import Document
from utils import logger
from config import config

# Global service instances
embedding_service: Optional[LocalEmbeddings] = None
reranker_service: Optional[Union[DefaultReranker, QwenReranker]] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app initialization and cleanup"""
    global embedding_service, reranker_service
    
    # Initialize services
    try:
        # Initialize embedding service
        logger.info(f"Initializing embedding service with model: {config.embedding.model_path}")
        embedding_service = LocalEmbeddings(
            model_name_or_path=config.embedding.model_path,
            model_engine=config.embedding.model_engine,
            dim=config.embedding.dim,
            device=config.embedding.device
        )
        logger.info("Embedding service initialized successfully")
        
        # Initialize reranker service (optional)
        if config.reranker.model_path:
            logger.info(f"Initializing reranker service with model: {config.reranker.model_path}")
            if config.reranker.reranker_type == "qwen":
                reranker_service = QwenReranker(
                    model_name_or_path=config.reranker.model_path,
                    use_fp16=config.reranker.use_fp16,
                    max_length=config.reranker.max_length,
                    device=config.reranker.device
                )
            else:
                reranker_service = DefaultReranker(
                    model_name_or_path=config.reranker.model_path,
                    use_fp16=config.reranker.use_fp16,
                    max_length=config.reranker.max_length,
                    device=config.reranker.device
                )
            logger.info("Reranker service initialized successfully")
        else:
            logger.info("Reranker service not configured, skipping initialization")
            
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}", exc_info=True)
        raise
    
    yield
    
    # Cleanup (if needed)
    logger.info("Shutting down services")


# Create FastAPI app
app = FastAPI(
    title=config.app.title,
    description=config.app.description,
    version=config.app.version,
    lifespan=lifespan
)


# Request/Response Models

class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request"""
    input: Union[str, List[str]] = Field(..., description="Input text(s) to embed")
    model: Optional[str] = Field(None, description="Model name (for compatibility)")
    encoding_format: Optional[str] = Field("float", description="Encoding format")
    dimensions: Optional[int] = Field(None, description="Number of dimensions")
    user: Optional[str] = Field(None, description="User identifier")


class EmbeddingData(BaseModel):
    """Embedding data item"""
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response"""
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: dict


class RerankRequest(BaseModel):
    """Rerank request"""
    query: str = Field(..., description="Query text")
    documents: List[str] = Field(..., description="List of documents to rerank")
    top_n: Optional[int] = Field(None, description="Number of top results to return")
    return_documents: Optional[bool] = Field(True, description="Whether to return document content")
    threshold: Optional[float] = Field(0.0, description="Score threshold for filtering")


class RerankResult(BaseModel):
    """Rerank result item"""
    index: int
    relevance_score: float
    document: Optional[str] = None


class RerankResponse(BaseModel):
    """Rerank response"""
    results: List[RerankResult]
    model: str
    usage: dict


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": config.app.title,
        "version": config.app.version,
        "endpoints": {
            "embeddings": "/v1/embeddings",
            "rerank": "/v1/rerank",
            "health": "/health"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "embedding_service": embedding_service is not None,
        "reranker_service": reranker_service is not None
    }
    return status


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    request: EmbeddingRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Create embeddings for input text(s).
    OpenAI-compatible endpoint.
    """
    if embedding_service is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding service is not initialized"
        )
    
    try:
        # Normalize input to list
        if isinstance(request.input, str):
            inputs = [request.input]
        else:
            inputs = request.input
        
        if not inputs:
            raise HTTPException(
                status_code=400,
                detail="Input cannot be empty"
            )
        
        # Generate embeddings
        start_time = time.time()
        embeddings = embedding_service.embed_documents(inputs)
        elapsed_time = time.time() - start_time
        
        # Build response
        embedding_data = []
        for idx, embedding in enumerate(embeddings):
            embedding_data.append(
                EmbeddingData(
                    object="embedding",
                    embedding=embedding,
                    index=idx
                )
            )
        
        # Determine model name
        model_name = request.model or embedding_service.model_name_or_path
        
        response = EmbeddingResponse(
            object="list",
            data=embedding_data,
            model=model_name,
            usage={
                "prompt_tokens": sum(len(text.split()) for text in inputs),
                "total_tokens": sum(len(text.split()) for text in inputs),
                "processing_time": elapsed_time
            }
        )
        
        logger.info(f"Generated embeddings for {len(inputs)} input(s) in {elapsed_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error creating embeddings: {str(e)}"
        )


@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank_documents(
    request: RerankRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Rerank documents based on query relevance.
    """
    if reranker_service is None:
        raise HTTPException(
            status_code=503,
            detail="Reranker service is not initialized"
        )
    
    try:
        if not request.query:
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )
        
        if not request.documents:
            raise HTTPException(
                status_code=400,
                detail="Documents list cannot be empty"
            )
        
        # Convert documents to Document objects
        docs = [
            Document(page_content=doc, metadata={"index": i})
            for i, doc in enumerate(request.documents)
        ]
        
        # Perform reranking
        start_time = time.time()
        results = reranker_service.rank(
            query=request.query,
            docs=docs,
            top_k=request.top_n,
            return_documents=request.return_documents,
            threshold=request.threshold
        )
        elapsed_time = time.time() - start_time
        
        # Build response
        rerank_results = []
        for result in results:
            rerank_result = RerankResult(
                index=result["corpus_id"],
                relevance_score=result["score"]
            )
            if request.return_documents and "document" in result:
                rerank_result.document = result["document"].page_content
            rerank_results.append(rerank_result)
        
        # Determine model name
        model_name = getattr(reranker_service, "model_name_or_path", "reranker")
        
        response = RerankResponse(
            results=rerank_results,
            model=model_name,
            usage={
                "query_tokens": len(request.query.split()),
                "document_count": len(request.documents),
                "results_count": len(rerank_results),
                "processing_time": elapsed_time
            }
        )
        
        logger.info(
            f"Reranked {len(request.documents)} documents for query "
            f"'{request.query[:50]}...' in {elapsed_time:.2f}s, "
            f"returned {len(rerank_results)} results"
        )
        return response
        
    except Exception as e:
        logger.error(f"Error reranking documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error reranking documents: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    import sys
    import os
    from pathlib import Path
    
    # Add current directory to Python path to ensure imports work
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

    
    config.embedding.model_path = os.path.join(current_dir, "models", "Qwen/Qwen3-Embedding-0.6B")
    config.reranker.model_path = os.path.join(current_dir, "models", "Qwen/Qwen3-Reranker-0.6B")
    
    logger.info(f"Starting RAG Service API on {config.server.host}:{config.server.port}")
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload,
        log_level=config.server.log_level
    )

