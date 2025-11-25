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

import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    """Embedding service configuration"""
    model_path: str = Field(
        default="Qwen/Qwen3-Embedding-0.6B",
        description="Path to the embedding model"
    )
    model_engine: str = Field(
        default="qwen",
        description="Model engine type (qwen, huggingface, etc.)"
    )
    dim: int = Field(
        default=1024,
        description="Embedding dimension"
    )
    device: str = Field(
        default="auto",
        description="Device to use (auto, cpu, cuda, mps)"
    )

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        """Create configuration from environment variables"""
        return cls(
            model_path=os.getenv("EMBEDDING_MODEL_PATH", "Qwen/Qwen3-Embedding-0.6B"),
            model_engine=os.getenv("EMBEDDING_MODEL_ENGINE", "qwen"),
            dim=int(os.getenv("EMBEDDING_DIM", "1024")),
            device=os.getenv("EMBEDDING_DEVICE", "auto")
        )


class RerankerConfig(BaseModel):
    """Reranker service configuration"""
    model_path: Optional[str] = Field(
        default=None,
        description="Path to the reranker model (optional)"
    )
    reranker_type: str = Field(
        default="qwen",
        description="Reranker type (qwen or default)"
    )
    device: str = Field(
        default="auto",
        description="Device to use (auto, cpu, cuda, mps)"
    )
    use_fp16: bool = Field(
        default=False,
        description="Whether to use FP16 precision"
    )
    max_length: int = Field(
        default=4096,
        description="Maximum sequence length"
    )

    @classmethod
    def from_env(cls) -> "RerankerConfig":
        """Create configuration from environment variables"""
        model_path = os.getenv("RERANKER_MODEL_PATH", None)
        reranker_type = os.getenv("RERANKER_TYPE", "qwen").lower()
        
        # Set default max_length based on reranker type
        default_max_length = 4096 if reranker_type == "qwen" else 512
        
        return cls(
            model_path=model_path,
            reranker_type=reranker_type,
            device=os.getenv("RERANKER_DEVICE", "auto"),
            use_fp16=os.getenv("RERANKER_USE_FP16", "false").lower() == "true",
            max_length=int(os.getenv("RERANKER_MAX_LENGTH", str(default_max_length)))
        )


class ServerConfig(BaseModel):
    """Server configuration"""
    host: str = Field(
        default="0.0.0.0",
        description="Server host address"
    )
    port: int = Field(
        default=8090,
        description="Server port"
    )
    reload: bool = Field(
        default=False,
        description="Enable auto-reload for development"
    )
    log_level: str = Field(
        default="info",
        description="Logging level"
    )

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Create configuration from environment variables"""
        return cls(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8090")),
            reload=os.getenv("RELOAD", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "info")
        )


class AppConfig(BaseModel):
    """FastAPI application configuration"""
    title: str = Field(
        default="RAG Service API",
        description="API title"
    )
    description: str = Field(
        default="OpenAI-compatible embedding and rerank API",
        description="API description"
    )
    version: str = Field(
        default="1.0.0",
        description="API version"
    )

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create configuration from environment variables"""
        return cls(
            title=os.getenv("API_TITLE", "RAG Service API"),
            description=os.getenv("API_DESCRIPTION", "OpenAI-compatible embedding and rerank API"),
            version=os.getenv("API_VERSION", "1.0.0")
        )


class VectorStoreConfig(BaseModel):
    """Vector store configuration"""
    host: str = Field(
        default="127.0.0.1",
        description="Vector store host address (empty for Milvus Lite)"
    )
    port: str = Field(
        default="19530",
        description="Vector store port (empty for Milvus Lite)"
    )
    user: str = Field(
        default="",
        description="Vector store username"
    )
    password: str = Field(
        default="",
        description="Vector store password"
    )
    name: str = Field(
        default="default",
        description="Database/collection name"
    )

    collection_name: str = Field(
        default="collection",
        description="Collection name"
    )
    
    reranker_type: str = Field(
        default="weighted",
        description="Reranker type"
    )

    reranker_weight: float = Field(
        default=0.5,
        description="Reranker weight"
    )

    chunk_context_window_size: int = Field(
        default=0,
        description="Context window size for long document recall"
    )

    kb_root_path: str = Field(
        default="./knowledge_base",
        description="Knowledge base root path for persistent storage"
    )
    kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments (index_params, search_params, etc.)"
    )

    @classmethod
    def from_env(cls) -> "VectorStoreConfig":
        """Create configuration from environment variables"""
        # Parse kwargs from environment if provided
        kwargs = {}
        index_params_str = os.getenv("VECTOR_STORE_INDEX_PARAMS")
        search_params_str = os.getenv("VECTOR_STORE_SEARCH_PARAMS")
        
        if index_params_str:
            try:
                import json
                kwargs["index_params"] = json.loads(index_params_str)
            except Exception:
                pass
        
        if search_params_str:
            try:
                import json
                kwargs["search_params"] = json.loads(search_params_str)
            except Exception:
                pass
        
        return cls(
            host=os.getenv("VECTOR_STORE_HOST", "127.0.0.1"),
            port=os.getenv("VECTOR_STORE_PORT", "19530"),
            user=os.getenv("VECTOR_STORE_USER", ""),
            password=os.getenv("VECTOR_STORE_PASSWORD", ""),
            name=os.getenv("VECTOR_STORE_NAME", "default"),
            kwargs=kwargs
        )


class ServiceConfig(BaseModel):
    """Main service configuration"""
    embedding: EmbeddingConfig = Field(
        default_factory=EmbeddingConfig.from_env,
        description="Embedding service configuration"
    )
    reranker: RerankerConfig = Field(
        default_factory=RerankerConfig.from_env,
        description="Reranker service configuration"
    )
    vector_store: VectorStoreConfig = Field(
        default_factory=VectorStoreConfig.from_env,
        description="Vector store configuration"
    )
    server: ServerConfig = Field(
        default_factory=ServerConfig.from_env,
        description="Server configuration"
    )
    app: AppConfig = Field(
        default_factory=AppConfig.from_env,
        description="Application configuration"
    )

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        """Create configuration from environment variables"""
        return cls(
            embedding=EmbeddingConfig.from_env(),
            reranker=RerankerConfig.from_env(),
            vector_store=VectorStoreConfig.from_env(),
            server=ServerConfig.from_env(),
            app=AppConfig.from_env()
        )


# Global configuration instance
config = ServiceConfig.from_env()

