package agent

import (
	"context"
	"fmt"
	"strings"

	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/embedding"
	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/llm"
	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/reranker"
	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/schema"
	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/vectordb"
)

// DefaultRAG implements a simple RAG agent that performs basic retrieval and Q&A.
// This agent is suitable for straightforward factual queries and single-hop questions.
// It performs simple retrieval and generates answers based on retrieved context.
type DefaultRAG struct {
	llm            llm.Provider
	embeddingModel embedding.Provider
	vectorDB       vectordb.VectorStoreProvider
	rerankerClient *reranker.RerankerClient
	config         *DefaultRAGConfig
}

// DefaultRAGDescription is the description for DefaultRAG agent.
// This agent performs simple retrieval and Q&A, suitable for straightforward factual queries.
var DefaultRAGDescription = DescribeAgent(
	"This agent performs simple retrieval and Q&A. " +
		"It is suitable for straightforward factual queries and single-hop questions. " +
		"It retrieves relevant documents and generates answers based on the retrieved context.",
)

// Ensure DefaultRAG implements RAGAgent interface at compile time
var _ RAGAgent = (*DefaultRAG)(nil)

// NewDefaultRAG creates a new DefaultRAG agent
func NewDefaultRAG(
	llm llm.Provider,
	embeddingModel embedding.Provider,
	vectorDB vectordb.VectorStoreProvider,
	rerankerClient *reranker.RerankerClient,
	config *DefaultRAGConfig,
) (*DefaultRAG, error) {
	if config == nil {
		config = DefaultRAGConfigWithDefaults()
	}
	if llm == nil {
		return nil, fmt.Errorf("llm provider cannot be nil")
	}
	if embeddingModel == nil {
		return nil, fmt.Errorf("embedding provider cannot be nil")
	}
	if vectorDB == nil {
		return nil, fmt.Errorf("vector DB provider cannot be nil")
	}

	return &DefaultRAG{
		llm:            llm,
		embeddingModel: embeddingModel,
		vectorDB:       vectorDB,
		rerankerClient: rerankerClient,
		config:         config,
	}, nil
}

// Description returns the description of DefaultRAG agent
// This implements the DescribableAgent interface for use with RAGRouter
func (d *DefaultRAG) Description() string {
	return DefaultRAGDescription.Description
}

// Invoke executes the agent with the given query and returns the result.
// This implements the BaseAgent interface.
// For DefaultRAG, Invoke calls Query and returns the answer string.
func (d *DefaultRAG) Invoke(ctx context.Context, query string, kwargs map[string]interface{}) (interface{}, error) {
	answer, _, _, err := d.Query(ctx, query, kwargs)
	if err != nil {
		return nil, err
	}
	return answer, nil
}

// Retrieve retrieves relevant documents from the knowledge base based on the query.
// This implements the RAGAgent interface.
// It performs vector search (with optional hybrid search and reranking) and returns retrieval results.
func (d *DefaultRAG) Retrieve(ctx context.Context, query string, kwargs map[string]interface{}) ([]RetrievalResult, int, map[string]interface{}, error) {
	// Get topK and threshold from kwargs if provided, otherwise use config defaults
	topK := d.config.TopK
	threshold := d.config.Threshold

	if kwargs != nil {
		if topKVal, ok := kwargs["top_k"].(int); ok {
			topK = topKVal
		}
		if thresholdVal, ok := kwargs["threshold"].(float64); ok {
			threshold = thresholdVal
		}
	}

	// Get embedding for the query
	queryVector, err := d.embeddingModel.GetEmbedding(ctx, query)
	if err != nil {
		return nil, 0, nil, fmt.Errorf("failed to get embedding: %w", err)
	}

	// Determine search topK based on reranking configuration
	useRerank := d.config.Rerank && d.rerankerClient != nil
	var searchTopK int
	if useRerank {
		searchTopK = d.config.RerankTopK
		if searchTopK == 0 {
			searchTopK = 20 // Default to 20 candidates for reranking
		}
		// Ensure we have enough candidates for reranking
		if searchTopK < topK {
			searchTopK = topK * 2
		}
	} else {
		searchTopK = topK
	}

	// Build search options
	options := &schema.SearchOptions{
		TopK:      searchTopK,
		Threshold: threshold,
	}

	// Perform search (hybrid or standard)
	var searchResults []schema.SearchResult
	if d.config.HybridSearch {
		searchResults, err = d.vectorDB.SearchHybridDocs(ctx, query, queryVector, options)
		if err != nil {
			return nil, 0, nil, fmt.Errorf("hybrid search failed: %w", err)
		}
	} else {
		searchResults, err = d.vectorDB.SearchDocs(ctx, query, queryVector, options)
		if err != nil {
			return nil, 0, nil, fmt.Errorf("search failed: %w", err)
		}
	}

	// Apply reranking if enabled
	if useRerank {
		rerankThreshold := threshold
		// Note: We don't have access to reranker config threshold here,
		// so we use the provided threshold
		searchResults, err = d.rerankerClient.RerankSearchResults(ctx, query, searchResults, topK, rerankThreshold)
		if err != nil {
			return nil, 0, nil, fmt.Errorf("rerank failed: %w", err)
		}
	}

	// Convert search results to RetrievalResult
	retrievalResults := make([]RetrievalResult, 0, len(searchResults))
	for _, result := range searchResults {
		retrievalResults = append(retrievalResults, RetrievalResult{
			Text:     result.Document.Content,
			Score:    result.Score,
			Document: result.Document,
			Metadata: result.Document.Metadata,
		})
	}

	// Token usage: only embedding tokens (LLM not used in retrieval)
	// Note: We don't have direct access to embedding token count,
	// so we return 0 for now. In practice, embedding models typically
	// don't expose token counts separately.
	tokenUsage := 0

	metadata := map[string]interface{}{
		"top_k":     topK,
		"threshold": threshold,
		"rerank":    useRerank,
	}

	return retrievalResults, tokenUsage, metadata, nil
}

// Query executes a query and returns the final answer along with retrieved results.
// This implements the RAGAgent interface.
// It performs retrieval and then generates an answer using the LLM.
func (d *DefaultRAG) Query(ctx context.Context, query string, kwargs map[string]interface{}) (string, []RetrievalResult, int, error) {
	// Retrieve relevant documents
	retrievalResults, nTokenRetrieval, _, err := d.Retrieve(ctx, query, kwargs)
	if err != nil {
		return "", nil, 0, err
	}

	// Build context from retrieved documents
	contexts := make([]string, 0, len(retrievalResults))
	for _, result := range retrievalResults {
		// Clean up the text (replace newlines with spaces for better formatting)
		cleanedText := strings.ReplaceAll(result.Text, "\n", " ")
		contexts = append(contexts, cleanedText)
	}

	// Build prompt using LLM's BuildPrompt function
	prompt := llm.BuildPrompt(query, contexts, "\n\n")

	// Generate answer using LLM
	answer, err := d.llm.GenerateCompletion(ctx, prompt)
	if err != nil {
		return "", nil, 0, fmt.Errorf("failed to generate completion: %w", err)
	}

	// Estimate token usage for LLM generation
	// Rough approximation: 1 token ≈ 4 characters
	// This is a simple estimation; actual token counts depend on the tokenizer
	promptTokens := len(prompt) / 4
	answerTokens := len(answer) / 4
	totalTokens := nTokenRetrieval + promptTokens + answerTokens

	return strings.TrimSpace(answer), retrievalResults, totalTokens, nil
}
