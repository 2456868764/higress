package rag

import (
	"context"
	"fmt"
	"time"

	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/agent"
	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/config"
	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/embedding"
	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/llm"
	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/reranker"
	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/schema"
	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/textsplitter"
	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/vectordb"
	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/websearch"
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
	websearchProvider websearch.InternetSearchProvider // Web search provider (optional)
	ragAgent          agent.RAGAgent                   // The RAG agent instance (DefaultRAG, ChainOfRAG, or RAGRouter)
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
	fmt.Printf("[RAGClient] RAG.Rerank config value: %v\n", ragclient.config.RAG.Rerank)
	if ragclient.config.RAG.Rerank {
		// api.LogDebugf("RAG New Reranker Client: %+v", ragclient.config.Reranker)
		rerankerClient, err := reranker.NewRerankerClient(&ragclient.config.Reranker)
		if err != nil {
			return nil, fmt.Errorf("create reranker client failed, err: %w", err)
		}
		ragclient.rerankerClient = rerankerClient
		fmt.Printf("[RAGClient] Reranker client initialized\n")
	} else {
		fmt.Printf("[RAGClient] Reranker client NOT initialized (RAG.Rerank is false)\n")
		ragclient.rerankerClient = nil // Explicitly set to nil to ensure it's not initialized
	}

	// Initialize websearch provider if websearch is enabled in config
	if ragclient.config.WebSearch.Enabled && ragclient.config.WebSearch.Provider != "" {
		// api.LogDebugf("RAG New WebSearch Provider: %+v", ragclient.config.WebSearch)
		websearchProvider, err := websearch.NewWebSearchProvider(ragclient.config.WebSearch)
		if err != nil {
			return nil, fmt.Errorf("create websearch provider failed, err: %w", err)
		}
		ragclient.websearchProvider = websearchProvider
	}

	// Initialize RAG agent based on configuration
	ragAgent, err := ragclient.createRAGAgent()
	if err != nil {
		return nil, fmt.Errorf("create RAG agent failed, err: %w", err)
	}
	ragclient.ragAgent = ragAgent

	return ragclient, nil
}

// createRAGAgent creates the appropriate RAG agent based on configuration
func (r *RAGClient) createRAGAgent() (agent.RAGAgent, error) {
	agentType := r.config.RAG.Agent
	if agentType == "" {
		agentType = "default" // Default to "default" if not specified
	}

	// Create DefaultRAGConfig from RAGConfig
	agentConfig := &agent.DefaultRAGConfig{
		TopK:         r.config.RAG.TopK,
		Threshold:    r.config.RAG.Threshold,
		Rerank:       r.config.RAG.Rerank,
		RerankTopK:   r.config.RAG.RerankTopK,
		HybridSearch: r.config.VectorDB.HybridSearch.Enabled,
	}
	fmt.Printf("[RAGClient] Creating agent with config - Rerank: %v, RerankTopK: %d\n", agentConfig.Rerank, agentConfig.RerankTopK)

	switch agentType {
	case "default":
		return r.createDefaultRAG(agentConfig)
	case "chain_of_rag":
		return r.createChainOfRAG(agentConfig)
	case "router":
		return r.createRouterRAG(agentConfig)
	case "deep_search":
		return r.createDeepSearch(agentConfig)
	default:
		return nil, fmt.Errorf("unknown agent type: %s (supported types: default, chain_of_rag, router, deep_search)", agentType)
	}
}

// createDefaultRAG creates a DefaultRAG agent
func (r *RAGClient) createDefaultRAG(config *agent.DefaultRAGConfig) (agent.RAGAgent, error) {
	fmt.Printf("createDefaultRAG: %+v\n", config)
	if r.llmProvider == nil {
		return nil, fmt.Errorf("llm provider is required for DefaultRAG agent")
	}

	return agent.NewDefaultRAG(
		r.llmProvider,
		r.embeddingProvider,
		r.vectordbProvider,
		r.rerankerClient,
		config,
	)
}

// createChainOfRAG creates a ChainOfRAG agent
func (r *RAGClient) createChainOfRAG(config *agent.DefaultRAGConfig) (agent.RAGAgent, error) {
	fmt.Printf("createChainOfRAG: %+v\n", config)
	if r.llmProvider == nil {
		return nil, fmt.Errorf("llm provider is required for ChainOfRAG agent")
	}

	// Default ChainOfRAG parameters
	maxIter := 3
	earlyStopping := true
	textWindowSplit := false

	return agent.NewChainOfRAG(
		r.llmProvider,
		r.embeddingProvider,
		r.vectordbProvider,
		r.rerankerClient,
		config,
		maxIter,
		earlyStopping,
		textWindowSplit,
	)
}

// createRouterRAG creates a RAGRouter agent that routes queries to appropriate agents
func (r *RAGClient) createRouterRAG(config *agent.DefaultRAGConfig) (agent.RAGAgent, error) {
	fmt.Printf("createRouterRAG: %+v\n", config)
	if r.llmProvider == nil {
		return nil, fmt.Errorf("llm provider is required for RAGRouter agent")
	}

	// Create multiple agents for routing
	agents := make([]agent.RAGAgent, 0)

	// Create DefaultRAG agent
	defaultRAG, err := r.createDefaultRAG(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create DefaultRAG for router: %w", err)
	}
	agents = append(agents, defaultRAG)

	// Create ChainOfRAG agent
	chainOfRAG, err := r.createChainOfRAG(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create ChainOfRAG for router: %w", err)
	}
	agents = append(agents, chainOfRAG)

	// Create router
	return agent.NewRAGRouter(r.llmProvider, agents)
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
		doc.Metadata["chunk_type"] = "original"
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
// If a RAG agent is configured, it uses the agent's Retrieve method.
// Otherwise, it falls back to direct vector database search.
func (r *RAGClient) SearchChunks(query string, topK int, threshold float64) ([]schema.SearchResult, error) {
	// Use RAG agent if available
	if r.ragAgent != nil {
		kwargs := map[string]interface{}{
			"top_k":     topK,
			"threshold": threshold,
		}
		retrievalResults, _, _, err := r.ragAgent.Retrieve(context.Background(), query, kwargs)
		if err != nil {
			return nil, fmt.Errorf("agent retrieve failed, err: %w", err)
		}

		// Convert RetrievalResult to schema.SearchResult
		searchResults := make([]schema.SearchResult, 0, len(retrievalResults))
		for _, result := range retrievalResults {
			searchResults = append(searchResults, schema.SearchResult{
				Document: result.Document,
				Score:    result.Score,
			})
		}

		return searchResults, nil
	}
	return nil, fmt.Errorf("rag agent not initialized")
}

// Chat generates a response using LLM
// If a RAG agent is configured, it uses the agent's Query method.
// Otherwise, it falls back to the default implementation.
func (r *RAGClient) Chat(query string) (string, error) {
	// Use RAG agent if available
	if r.ragAgent != nil {
		answer, _, _, err := r.ragAgent.Query(context.Background(), query, nil)
		if err != nil {
			return "", fmt.Errorf("agent query failed, err: %w", err)
		}
		return answer, nil
	}
	return "", fmt.Errorf("rag agent not initialized")
}

// createDeepSearch creates a DeepSearch agent
func (r *RAGClient) createDeepSearch(config *agent.DefaultRAGConfig) (agent.RAGAgent, error) {
	fmt.Printf("createDeepSearch: %+v\n", config)
	if r.llmProvider == nil {
		return nil, fmt.Errorf("llm provider is required for DeepSearch agent")
	}

	// Default DeepSearch parameters
	maxIter := 3
	textWindowSplit := false

	// Configure internet search from config
	internetSearchConfig := websearch.DefaultInternetSearchConfig()
	if r.config.WebSearch.Enabled && r.websearchProvider != nil {
		internetSearchConfig.Enabled = true
		internetSearchConfig.Provider = r.websearchProvider
		if r.config.WebSearch.MaxResults > 0 {
			internetSearchConfig.MaxResults = r.config.WebSearch.MaxResults
		}
	}

	return agent.NewDeepSearch(
		r.llmProvider,
		r.embeddingProvider,
		r.vectordbProvider,
		config,
		maxIter,
		textWindowSplit,
		internetSearchConfig,
	)
}
