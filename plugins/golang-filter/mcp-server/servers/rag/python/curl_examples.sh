#!/bin/bash

# RAG Service API - cURL Test Examples
# Base URL (adjust according to your deployment)
BASE_URL="http://localhost:8000"

echo "=========================================="
echo "RAG Service API - cURL Test Examples"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================
# 1. Health Check
# ============================================
echo -e "${BLUE}1. Health Check${NC}"
echo "GET $BASE_URL/health"
echo ""
curl -X GET "$BASE_URL/health" \
  -H "Content-Type: application/json" \
  | jq '.'
echo ""
echo ""

# ============================================
# 2. Root Endpoint
# ============================================
echo -e "${BLUE}2. Root Endpoint${NC}"
echo "GET $BASE_URL/"
echo ""
curl -X GET "$BASE_URL/" \
  -H "Content-Type: application/json" \
  | jq '.'
echo ""
echo ""

# ============================================
# 3. Embedding - Single Text
# ============================================
echo -e "${GREEN}3. Embedding - Single Text${NC}"
echo "POST $BASE_URL/v1/embeddings"
echo ""
curl -X POST "$BASE_URL/v1/embeddings" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key-here" \
  -d '{
    "input": "What is artificial intelligence?",
    "model": "Qwen3-Embedding-0.6B"
  }' \
  | jq '.'
echo ""
echo ""

# ============================================
# 4. Embedding - Multiple Texts
# ============================================
echo -e "${GREEN}4. Embedding - Multiple Texts${NC}"
echo "POST $BASE_URL/v1/embeddings"
echo ""
curl -X POST "$BASE_URL/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [
      "What is machine learning?",
      "How does deep learning work?",
      "Explain neural networks"
    ],
    "model": "Qwen3-Embedding-0.6B"
  }' \
  | jq '.'
echo ""
echo ""

# ============================================
# 5. Embedding - With All Parameters
# ============================================
echo -e "${GREEN}5. Embedding - With All Parameters${NC}"
echo "POST $BASE_URL/v1/embeddings"
echo ""
curl -X POST "$BASE_URL/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Natural language processing is a branch of artificial intelligence.",
    "model": "Qwen3-Embedding-0.6B",
    "encoding_format": "float",
    "dimensions": 1024,
    "user": "test-user-123"
  }' \
  | jq '.'
echo ""
echo ""

# ============================================
# 6. Embedding - Long Text
# ============================================
echo -e "${GREEN}6. Embedding - Long Text${NC}"
echo "POST $BASE_URL/v1/embeddings"
echo ""
curl -X POST "$BASE_URL/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Retrieval-Augmented Generation (RAG) is a technique that enhances the accuracy and reliability of generative AI models with facts fetched from external knowledge bases. By grounding LLMs on the most accurate, up-to-date information and giving users insight into LLMs generative process, RAG helps models produce more reliable outputs."
  }' \
  | jq '.'
echo ""
echo ""

# ============================================
# 7. Rerank - Basic Example
# ============================================
echo -e "${YELLOW}7. Rerank - Basic Example${NC}"
echo "POST $BASE_URL/v1/rerank"
echo ""
curl -X POST "$BASE_URL/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is artificial intelligence?",
    "documents": [
      "Artificial intelligence is the simulation of human intelligence by machines.",
      "Machine learning is a subset of AI that enables systems to learn from data.",
      "The weather today is sunny and warm.",
      "Deep learning uses neural networks to process complex patterns."
    ]
  }' \
  | jq '.'
echo ""
echo ""

# ============================================
# 8. Rerank - With Top N
# ============================================
echo -e "${YELLOW}8. Rerank - With Top N${NC}"
echo "POST $BASE_URL/v1/rerank"
echo ""
curl -X POST "$BASE_URL/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does neural network training work?",
    "documents": [
      "Neural networks are trained using backpropagation algorithm.",
      "Training involves forward pass and backward pass.",
      "The training data is split into batches.",
      "Gradient descent optimizes the network weights.",
      "Overfitting can occur if training goes on too long.",
      "Regularization techniques help prevent overfitting."
    ],
    "top_n": 3
  }' \
  | jq '.'
echo ""
echo ""

# ============================================
# 9. Rerank - With Threshold
# ============================================
echo -e "${YELLOW}9. Rerank - With Threshold${NC}"
echo "POST $BASE_URL/v1/rerank"
echo ""
curl -X POST "$BASE_URL/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is transformer architecture?",
    "documents": [
      "Transformers use self-attention mechanism.",
      "BERT and GPT are transformer-based models.",
      "The transformer architecture revolutionized NLP.",
      "Cooking recipes often use fresh ingredients.",
      "Attention mechanism allows models to focus on relevant parts."
    ],
    "top_n": 5,
    "threshold": 0.5
  }' \
  | jq '.'
echo ""
echo ""

# ============================================
# 10. Rerank - Without Returning Documents
# ============================================
echo -e "${YELLOW}10. Rerank - Without Returning Documents${NC}"
echo "POST $BASE_URL/v1/rerank"
echo ""
curl -X POST "$BASE_URL/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain gradient descent",
    "documents": [
      "Gradient descent is an optimization algorithm.",
      "It minimizes the loss function iteratively.",
      "Learning rate controls the step size.",
      "Stochastic gradient descent uses random samples."
    ],
    "return_documents": false
  }' \
  | jq '.'
echo ""
echo ""

# ============================================
# 11. Rerank - Real-world Example
# ============================================
echo -e "${YELLOW}11. Rerank - Real-world Example${NC}"
echo "POST $BASE_URL/v1/rerank"
echo ""
curl -X POST "$BASE_URL/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How to implement RAG system?",
    "documents": [
      "RAG combines retrieval and generation for better AI responses.",
      "First, retrieve relevant documents from a vector database.",
      "Then, use the retrieved context to generate answers.",
      "Vector embeddings enable semantic search.",
      "Chunking documents improves retrieval accuracy.",
      "The system needs an embedding model and a language model."
    ],
    "top_n": 4,
    "threshold": 0.3,
    "return_documents": true
  }' \
  | jq '.'
echo ""
echo ""

# ============================================
# Error Cases
# ============================================
echo -e "${BLUE}=========================================="
echo "Error Cases"
echo "==========================================${NC}"
echo ""

# ============================================
# 12. Embedding - Empty Input
# ============================================
echo -e "${YELLOW}12. Embedding - Empty Input (Error)${NC}"
echo "POST $BASE_URL/v1/embeddings"
echo ""
curl -X POST "$BASE_URL/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": ""
  }' \
  | jq '.'
echo ""
echo ""

# ============================================
# 13. Rerank - Empty Query
# ============================================
echo -e "${YELLOW}13. Rerank - Empty Query (Error)${NC}"
echo "POST $BASE_URL/v1/rerank"
echo ""
curl -X POST "$BASE_URL/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "",
    "documents": ["Document 1", "Document 2"]
  }' \
  | jq '.'
echo ""
echo ""

# ============================================
# 14. Rerank - Empty Documents
# ============================================
echo -e "${YELLOW}14. Rerank - Empty Documents (Error)${NC}"
echo "POST $BASE_URL/v1/rerank"
echo ""
curl -X POST "$BASE_URL/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Test query",
    "documents": []
  }' \
  | jq '.'
echo ""
echo ""

echo "=========================================="
echo "All tests completed!"
echo "=========================================="

