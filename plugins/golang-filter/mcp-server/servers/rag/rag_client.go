package rag

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/config"
	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/embedding"
	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/llm"
	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/reranker"
	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/schema"
	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/textsplitter"
	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/vectordb"
	"github.com/distribution/distribution/v3/uuid"
)

const (
	MAX_LIST_KNOWLEDGE_ROW_COUNT = 1000
	MAX_LIST_DOCUMENT_ROW_COUNT  = 1000
)

// RAGClient represents the RAG (Retrieval-Augmented Generation) client
type RAGClient struct {
	config            *config.Config
	vectordbProvider  vectordb.VectorStoreProvider
	embeddingProvider embedding.Provider
	textSplitter      textsplitter.TextSplitter
	llmProvider       llm.Provider
	rerankerClient    *reranker.RerankerClient
}

// NewRAGClient creates a new RAG client instance
func NewRAGClient(config *config.Config) (*RAGClient, error) {
	// api.LogDebugf("RAG NewRAGClient: %+v", config)
	ragclient := &RAGClient{
		config: config,
	}
	textSplitter, err := textsplitter.NewTextSplitter(&config.RAG.Splitter)
	if err != nil {
		return nil, fmt.Errorf("create text splitter failed, err: %w", err)
	}
	ragclient.textSplitter = textSplitter

	// api.LogDebugf("RAG New Embedding Provider: %+v", ragclient.config.Embedding)
	embeddingProvider, err := embedding.NewEmbeddingProvider(ragclient.config.Embedding)
	if err != nil {
		return nil, fmt.Errorf("create embedding provider failed, err: %w", err)
	}
	ragclient.embeddingProvider = embeddingProvider

	// api.LogDebugf("RAG New LLM Provider: %+v", ragclient.config.LLM)
	if ragclient.config.LLM.Provider == "" {
		ragclient.llmProvider = nil
	} else {
		llmProvider, err := llm.NewLLMProvider(ragclient.config.LLM)
		if err != nil {
			return nil, fmt.Errorf("create llm provider failed, err: %w", err)
		}
		ragclient.llmProvider = llmProvider
	}

	// api.LogDebugf("RAG New VectorDB Provider: %+v", ragclient.config.VectorDB)
	dim := ragclient.config.Embedding.Dimensions
	provider, err := vectordb.NewVectorDBProvider(&ragclient.config.VectorDB, dim)
	if err != nil {
		return nil, fmt.Errorf("create vector store provider failed, err: %w", err)
	}
	ragclient.vectordbProvider = provider

	// Initialize reranker client if reranking is enabled in RAG config
	if ragclient.config.RAG.Rerank {
		// api.LogDebugf("RAG New Reranker Client: %+v", ragclient.config.Reranker)
		rerankerClient, err := reranker.NewRerankerClient(&ragclient.config.Reranker)
		if err != nil {
			return nil, fmt.Errorf("create reranker client failed, err: %w", err)
		}
		ragclient.rerankerClient = rerankerClient
	}

	return ragclient, nil
}

// ListChunks lists document chunks by knowledge ID, returns in ascending order of DocumentIndex
func (r *RAGClient) ListChunks() ([]schema.Document, error) {
	docs, err := r.vectordbProvider.ListDocs(context.Background(), MAX_LIST_DOCUMENT_ROW_COUNT)
	if err != nil {
		return nil, fmt.Errorf("list chunks failed, err: %w", err)
	}
	return docs, nil
}

// DeleteChunk deletes a specific document chunk
func (r *RAGClient) DeleteChunk(id string) error {
	if err := r.vectordbProvider.DeleteDocs(context.Background(), []string{id}); err != nil {
		return fmt.Errorf("delete chunk failed, err: %w", err)
	}
	return nil
}

func (r *RAGClient) CreateChunkFromText(text string, title string) ([]schema.Document, error) {

	docs, err := textsplitter.CreateDocuments(r.textSplitter, []string{text}, make([]map[string]any, 0))
	if err != nil {
		return nil, fmt.Errorf("create documents failed, err: %w", err)
	}

	results := make([]schema.Document, 0, len(docs))

	for chunkIndex, doc := range docs {
		doc.ID = uuid.Generate().String()
		doc.Metadata["chunk_index"] = chunkIndex
		doc.Metadata["chunk_title"] = title
		doc.Metadata["chunk_size"] = len(doc.Content)
		// Generate embedding for the document
		embedding, err := r.embeddingProvider.GetEmbedding(context.Background(), doc.Content)
		if err != nil {
			return nil, fmt.Errorf("create embedding failed, err: %w", err)
		}
		doc.Vector = embedding
		doc.CreatedAt = time.Now()
		results = append(results, doc)
	}

	if err := r.vectordbProvider.AddDoc(context.Background(), results); err != nil {
		return nil, fmt.Errorf("add documents failed, err: %w", err)
	}

	return results, nil
}

// SearchChunks searches for document chunks with optional reranking
// If reranking is enabled, it first retrieves more candidates (RerankTopK, default 20),
// then reranks them to return topK results, similar to Python's simple_retrieval.py
func (r *RAGClient) SearchChunks(query string, topK int, threshold float64) ([]schema.SearchResult, error) {
	vector, err := r.embeddingProvider.GetEmbedding(context.Background(), query)
	// fmt.Printf("vector: %+v\n", vector)

	if err != nil {
		return nil, fmt.Errorf("create embedding failed, err: %w", err)
	}

	// Check if reranking should be used
	useRerank := r.config.RAG.Rerank && r.rerankerClient != nil

	var searchTopK int
	if useRerank {
		// If reranking is enabled, retrieve more candidates first (similar to Python's similarity_top_k=20)
		searchTopK = r.config.RAG.RerankTopK
		if searchTopK == 0 {
			searchTopK = 20 // Default to 20 candidates for reranking, same as Python implementation
		}
		// Ensure we have enough candidates for reranking
		if searchTopK < topK {
			searchTopK = topK * 2
		}
	} else {
		// Direct retrieval without reranking
		searchTopK = topK
	}

	options := &schema.SearchOptions{
		TopK:      searchTopK,
		Threshold: threshold,
	}

	var docs []schema.SearchResult
	if r.config.VectorDB.HybridSearch.Enabled {
		docs, err = r.vectordbProvider.SearchHybridDocs(context.Background(), query, vector, options)
		if err != nil {
			return nil, fmt.Errorf("search hybrid chunks failed, err: %w", err)
		}
	} else {
		// fmt.Printf("search query: %s, options: %+v\n", query, options)
		docs, err = r.vectordbProvider.SearchDocs(context.Background(), query, vector, options)
		if err != nil {
			return nil, fmt.Errorf("search chunks failed, err: %w", err)
		}
	}

	// Apply reranking if enabled
	if useRerank {
		// Use reranker threshold if configured, otherwise use the provided threshold
		rerankThreshold := threshold
		if r.config.Reranker.Threshold >= 0 {
			rerankThreshold = r.config.Reranker.Threshold
		}

		docs, err = r.rerankerClient.RerankSearchResults(context.Background(), query, docs, topK, rerankThreshold)
		if err != nil {
			return nil, fmt.Errorf("rerank failed, err: %w", err)
		}
	}

	return docs, nil
}

// Chat generates a response using LLM
func (r *RAGClient) Chat(query string) (string, error) {
	if r.llmProvider == nil {
		return "", fmt.Errorf("llm provider not initialized")
	}

	docs, err := r.SearchChunks(query, r.config.RAG.TopK, r.config.RAG.Threshold)
	if err != nil {
		return "", fmt.Errorf("search chunks failed, err: %w", err)
	}

	contexts := make([]string, 0, len(docs))
	for _, doc := range docs {
		contexts = append(contexts, strings.ReplaceAll(doc.Document.Content, "\n", " "))
	}

	prompt := llm.BuildPrompt(query, contexts, "\n\n")
	resp, err := r.llmProvider.GenerateCompletion(context.Background(), prompt)
	if err != nil {
		return "", fmt.Errorf("generate completion failed, err: %w", err)
	}
	return resp, nil
}
