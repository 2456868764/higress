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
	fmt.Printf("[DefaultRAG] ===== Starting Retrieve Phase =====\n")
	fmt.Printf("[DefaultRAG] Query: %s\n", query)

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

	fmt.Printf("[DefaultRAG] Step 1: Getting query embedding...\n")
	fmt.Printf("[DefaultRAG]   TopK: %d, Threshold: %.2f\n", topK, threshold)

	// Get embedding for the query
	queryVector, err := d.embeddingModel.GetEmbedding(ctx, query)
	if err != nil {
		fmt.Printf("[DefaultRAG]   ERROR: Failed to get embedding: %v\n", err)
		return nil, 0, nil, fmt.Errorf("failed to get embedding: %w", err)
	}
	fmt.Printf("[DefaultRAG]   Query embedding obtained (dimension: %d)\n", len(queryVector))

	// Determine search topK based on reranking configuration
	fmt.Printf("[DefaultRAG] Config check - Rerank: %v, rerankerClient != nil: %v\n", d.config.Rerank, d.rerankerClient != nil)
	useRerank := d.config.Rerank && d.rerankerClient != nil
	fmt.Printf("[DefaultRAG] useRerank decision: %v\n", useRerank)
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
		fmt.Printf("[DefaultRAG]   Reranking enabled: SearchTopK=%d (will rerank to TopK=%d)\n", searchTopK, topK)
	} else {
		searchTopK = topK
		fmt.Printf("[DefaultRAG]   Reranking disabled: SearchTopK=%d\n", searchTopK)
	}

	// Build search options
	options := &schema.SearchOptions{
		TopK:      searchTopK,
		Threshold: threshold,
	}

	// Perform search (hybrid or standard)
	fmt.Printf("[DefaultRAG] Step 2: Performing vector search...\n")
	fmt.Printf("[DefaultRAG]   Search mode: %s\n", map[bool]string{true: "Hybrid Search", false: "Standard Search"}[d.config.HybridSearch])

	var searchResults []schema.SearchResult
	if d.config.HybridSearch {
		searchResults, err = d.vectorDB.SearchHybridDocs(ctx, query, queryVector, options)
		if err != nil {
			fmt.Printf("[DefaultRAG]   ERROR: Hybrid search failed: %v\n", err)
			return nil, 0, nil, fmt.Errorf("hybrid search failed: %w", err)
		}
	} else {
		searchResults, err = d.vectorDB.SearchDocs(ctx, query, queryVector, options)
		if err != nil {
			fmt.Printf("[DefaultRAG]   ERROR: Search failed: %v\n", err)
			return nil, 0, nil, fmt.Errorf("search failed: %w", err)
		}
	}
	fmt.Printf("[DefaultRAG]   Retrieved %d documents from vector DB\n", len(searchResults))

	// Apply reranking if enabled
	if useRerank {
		fmt.Printf("[DefaultRAG] Step 3: Applying reranking...\n")
		rerankThreshold := threshold
		// Note: We don't have access to reranker config threshold here,
		// so we use the provided threshold
		searchResults, err = d.rerankerClient.RerankSearchResults(ctx, query, searchResults, topK, rerankThreshold)
		if err != nil {
			fmt.Printf("[DefaultRAG]   ERROR: Rerank failed: %v\n", err)
			return nil, 0, nil, fmt.Errorf("rerank failed: %w", err)
		}
		fmt.Printf("[DefaultRAG]   Reranked to %d documents\n", len(searchResults))
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

	fmt.Printf("[DefaultRAG] Step 4: Finalizing results...\n")
	fmt.Printf("[DefaultRAG]   Final retrieval results: %d documents\n", len(retrievalResults))
	if len(retrievalResults) > 0 {
		fmt.Printf("[DefaultRAG]   Top score: %.4f\n", retrievalResults[0].Score)
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

	fmt.Printf("[DefaultRAG] ===== Retrieve Phase Completed =====\n\n")
	return retrievalResults, tokenUsage, metadata, nil
}

// Query executes a query and returns the final answer along with retrieved results.
// This implements the RAGAgent interface.
// It performs retrieval and then generates an answer using the LLM.
func (d *DefaultRAG) Query(ctx context.Context, query string, kwargs map[string]interface{}) (string, []RetrievalResult, int, error) {
	fmt.Printf("[DefaultRAG] ===== Starting Query Phase =====\n")
	fmt.Printf("[DefaultRAG] Query: %s\n", query)

	// Retrieve relevant documents
	retrievalResults, nTokenRetrieval, _, err := d.Retrieve(ctx, query, kwargs)
	if err != nil {
		fmt.Printf("[DefaultRAG] ERROR: Retrieve failed: %v\n", err)
		return "", nil, 0, err
	}

	fmt.Printf("[DefaultRAG] Step 5: Building context from retrieved documents...\n")
	fmt.Printf("[DefaultRAG]   Retrieved documents: %d\n", len(retrievalResults))
	fmt.Printf("[DefaultRAG]   Retrieval tokens used: %d\n", nTokenRetrieval)

	// Build context from retrieved documents
	contexts := make([]string, 0, len(retrievalResults))
	totalContextLength := 0
	for _, result := range retrievalResults {
		// Clean up the text (replace newlines with spaces for better formatting)
		cleanedText := strings.ReplaceAll(result.Text, "\n", " ")
		contexts = append(contexts, cleanedText)
		totalContextLength += len(cleanedText)
	}
	fmt.Printf("[DefaultRAG]   Total context length: %d characters\n", totalContextLength)

	// Build prompt using LLM's BuildPrompt function
	fmt.Printf("[DefaultRAG] Step 6: Building prompt and generating answer...\n")
	prompt := llm.BuildPrompt(query, contexts, "\n\n")
	promptLength := len(prompt)
	fmt.Printf("[DefaultRAG]   Prompt length: %d characters\n", promptLength)

	// Generate answer using LLM
	answer, err := d.llm.GenerateCompletion(ctx, prompt)
	if err != nil {
		fmt.Printf("[DefaultRAG]   ERROR: Failed to generate completion: %v\n", err)
		return "", nil, 0, fmt.Errorf("failed to generate completion: %w", err)
	}

	answerLength := len(answer)
	fmt.Printf("[DefaultRAG]   Answer generated (length: %d characters)\n", answerLength)

	// Estimate token usage for LLM generation
	// Rough approximation: 1 token ≈ 4 characters
	// This is a simple estimation; actual token counts depend on the tokenizer
	promptTokens := promptLength / 4
	answerTokens := answerLength / 4
	totalTokens := nTokenRetrieval + promptTokens + answerTokens

	fmt.Printf("[DefaultRAG]   Estimated tokens - Prompt: %d, Answer: %d, Total: %d\n", promptTokens, answerTokens, totalTokens)
	fmt.Printf("[DefaultRAG] ===== Query Phase Completed =====\n\n")

	return strings.TrimSpace(answer), retrievalResults, totalTokens, nil
}
