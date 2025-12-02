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

import json
import os
import uuid
from typing import List
from pathlib import Path
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from .milvus_hybrid import MilvusHybridVectorStore
    from .service.local_embedding import LocalEmbeddings
    from .config import config
    from .base import CorpusFile
    from .utils import logger
except ImportError:
    # For direct script execution
    from milvus_hybrid import MilvusHybridVectorStore
    from service.local_embedding import LocalEmbeddings
    from config import config
    from base import CorpusFile
    from utils import logger


class CorpusFileImpl(CorpusFile):
    """CorpusFile implementation"""
    
    def __init__(self, filename: str, metadata: dict):
        self.filename = filename
        self.metadata = metadata


def load_corpus_json(file_path: str) -> List[dict]:
    """
    Load corpus.json file
    
    Args:
        file_path: Path to corpus.json file
    
    Returns:
        List of document dictionaries
    """
    logger.info(f"Loading corpus from: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} documents from corpus.json")
    return data


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Split text into chunks using RecursiveCharacterTextSplitter
    
    Args:
        text: Text to split
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", ",", ";", "。", "?", "!", "；"]
    )
    
    chunks = splitter.split_text(text)
    return chunks


def create_documents_from_chunks(
    chunks: List[str],
    base_metadata: dict,
    chunk_index_offset: int = 0
) -> List[Document]:
    """
    Create Document objects from text chunks
    
    Args:
        chunks: List of text chunks
        base_metadata: Base metadata to attach to each chunk
        chunk_index_offset: Starting index for chunk numbering
    
    Returns:
        List of Document objects
    """
    documents = []
    
    for idx, chunk in enumerate(chunks):
        # Create metadata for this chunk
        chunk_metadata = base_metadata.copy()
        chunk_metadata['chunk_index'] = chunk_index_offset + idx
        chunk_metadata['id'] = str(uuid.uuid4())
        
        # Create Document object
        doc = Document(
            page_content=chunk,
            metadata=chunk_metadata
        )
        documents.append(doc)
    
    return documents


def index_corpus_to_milvus(
    corpus_file_path: str,
    collection_name: str = "corpus_collection",
    chunk_size: int = 500,
    smaller_chunk_size: int = 0,
    chunk_overlap: int = 50,
    batch_size: int = 100
):
    """
    Index corpus.json to Milvus vector database
    
    Args:
        corpus_file_path: Path to corpus.json file
        collection_name: Name of the Milvus collection
        chunk_size: Size of each text chunk
        chunk_overlap: Overlap between chunks
        batch_size: Number of documents to process in each batch
    """
    # Initialize embedding model
    logger.info("Initializing embedding model...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    embedding_model_path = os.path.join(current_dir, "models", config.embedding.model_path)
    embedding_model = LocalEmbeddings(
        model_name_or_path=embedding_model_path,
        model_engine=config.embedding.model_engine,
        dim=config.embedding.dim,
        device=config.embedding.device
    )
    logger.info("Embedding model initialized")
    
    # Create vector store config
    vector_store_config = config.vector_store.model_copy()
    vector_store_config.collection_name = collection_name
    logger.info(f"Vector store config: {vector_store_config}")
    # Initialize Milvus vector store
    logger.info(f"Initializing Milvus vector store with collection: {collection_name}")
    vector_store = MilvusHybridVectorStore(
        embedding_model=embedding_model,
        config=vector_store_config
    )
    logger.info("Milvus vector store initialized")
    
    # Load corpus data
    corpus_data = load_corpus_json(corpus_file_path)
    
    # Process documents in batches
    total_chunks = 0
    total_documents = len(corpus_data)
    
    logger.info(f"Starting to index {total_documents} documents...")
    logger.info(f"Chunk size: {chunk_size}, Chunk overlap: {chunk_overlap}")
    
    for batch_start in tqdm(range(0, total_documents, batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, total_documents)
        batch_data = corpus_data[batch_start:batch_end]
        
        for doc_data in batch_data:
            try:
                # Extract document fields
                title = doc_data.get('title', '')
                body = doc_data.get('body', '')
                url = doc_data.get('url', '')
                source = doc_data.get('source', '')
                author = doc_data.get('author')
                published_at = doc_data.get('published_at', '')
                category = doc_data.get('category', '')
                
                # Skip if body is empty
                if not body or not body.strip():
                    logger.warning(f"Skipping document with empty body: {title}")
                    continue
                
                # Split body into chunks
                chunks = split_text(body, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                
                if not chunks:
                    logger.warning(f"No chunks generated for document: {title}")
                    continue
                
                # Create base metadata
                base_metadata = {
                    'title': title,
                    'source': source,
                    'url': url,
                    'author': author if author else '',
                    'published_at': published_at,
                    'category': category,
                    "chunk_type": "original"
                }
                # Create Document objects from chunks
                documents = create_documents_from_chunks(chunks, base_metadata)
                # Create CorpusFile object
                corpus_file = CorpusFileImpl(
                    filename=url or title,
                    metadata=base_metadata
                )
                
                # Add documents to Milvus
                doc_infos = vector_store.add_doc(corpus_file, documents)
                total_chunks += len(documents)

                if smaller_chunk_size > 0:
                    # split the documents into smaller chunks
                    smaller_documents = []
                    for doc in documents:
                        smaller_chunks = split_text(doc.page_content, chunk_size=smaller_chunk_size, chunk_overlap=0)
                        if smaller_chunks:
                            for smaller_chunk in smaller_chunks:
                                smaller_base_metadata = base_metadata.copy()
                                smaller_base_metadata["chunk_type"] = "smaller"
                                smaller_base_metadata['parent_id'] = doc.metadata['id']
                                smaller_documents = create_documents_from_chunks(smaller_chunks, smaller_base_metadata)
                                smaller_corpus_file = CorpusFileImpl(
                                    filename=url or title,
                                    metadata=smaller_base_metadata
                                )
                                smaller_doc_infos = vector_store.add_doc(smaller_corpus_file, smaller_documents)
                                total_chunks += len(smaller_documents)
                
                logger.debug(
                    f"Indexed document '{title}': "
                    f"{len(chunks)} chunks, {len(documents)} documents"
                )
                
            except Exception as e:
                logger.error(f"Error processing document '{doc_data.get('title', 'unknown')}': {e}", exc_info=True)
                continue
        
        logger.info(f"Processed batch {batch_start // batch_size + 1}: "
                   f"{batch_start + 1}-{batch_end} / {total_documents}")
    
    logger.info(f"Indexing completed!")
    logger.info(f"Total documents processed: {total_documents}")
    logger.info(f"Total chunks created: {total_chunks}")
    logger.info(f"Collection name: {collection_name}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Index corpus.json to Milvus vector database")
    parser.add_argument(
        '--corpus_file',
        type=str,
        default='dataset/corpus.json',
        help='Path to corpus.json file'
    )
    parser.add_argument(
        '--collection_name',
        type=str,
        default='corpus_collection_1000',
        help='Name of the Milvus collection'
    )
    parser.add_argument(
        '--chunk_size',
        type=int,
        default=500,
        help='Size of each text chunk'
    )

    parser.add_argument(
        '--smaller_chunk_size',
        type=int,
        default=0,
        help='Size of each smaller text chunk'
    )

    parser.add_argument(
        '--chunk_overlap',
        type=int,
        default=100,
        help='Overlap between chunks'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Number of documents to process in each batch'
    )
    
    args = parser.parse_args()
    
    # Get absolute path
    script_dir = Path(__file__).parent
    corpus_path = script_dir / args.corpus_file
    
    if not corpus_path.exists():
        logger.error(f"Corpus file not found: {corpus_path}")
        exit(1)
    
    # Index corpus to Milvus
    index_corpus_to_milvus(
        corpus_file_path=str(corpus_path),
        collection_name=args.collection_name,
        chunk_size=args.chunk_size,
        smaller_chunk_size=args.smaller_chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size
    )

