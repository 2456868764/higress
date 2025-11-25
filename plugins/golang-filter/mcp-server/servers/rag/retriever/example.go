package retriever

import (
	"context"
	"fmt"

	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag"
	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/config"
)

// ExampleUsage demonstrates how to use the Retriever
func ExampleUsage() {
	// 1. Create configuration
	cfg := &config.Config{
		RAG: config.RAGConfig{
			TopK:      10,
			Threshold: 0.5,
			Rerank:    true,        // Enable reranking
			RerankTopK: 20,        // Retrieve 20 candidates before reranking
		},
		Reranker: config.RerankerConfig{
			Enabled:   true,
			BaseURL:   "http://localhost:8000", // Python reranker service URL
			Threshold: 0.0,                     // Reranker threshold
		},
		// ... other config fields (Embedding, VectorDB, etc.)
	}

	// 2. Create RAG client
	ragClient, err := rag.NewRAGClient(cfg)
	if err != nil {
		fmt.Printf("Failed to create RAG client: %v\n", err)
		return
	}

	// 3. Create Retriever
	retriever, err := NewRetriever(ragClient, cfg)
	if err != nil {
		fmt.Printf("Failed to create retriever: %v\n", err)
		return
	}

	// 4. Use retriever with default options
	query := "What is the capital of France?"
	results, err := retriever.RetrieveWithDefaults(context.Background(), query)
	if err != nil {
		fmt.Printf("Retrieval failed: %v\n", err)
		return
	}

	fmt.Printf("Retrieved %d results:\n", len(results))
	for i, result := range results {
		fmt.Printf("Result %d: Score=%.4f, Content=%s\n", i+1, result.Score, result.Document.Content[:50])
	}

	// 5. Use retriever with custom options
	customOptions := &RetrieverOptions{
		TopK:      5,
		Threshold: 0.6,
		Rerank:    false, // Disable reranking for this query
	}

	results2, err := retriever.Retrieve(context.Background(), query, customOptions)
	if err != nil {
		fmt.Printf("Retrieval failed: %v\n", err)
		return
	}

	fmt.Printf("Retrieved %d results without reranking:\n", len(results2))
	for i, result := range results2 {
		fmt.Printf("Result %d: Score=%.4f, Content=%s\n", i+1, result.Score, result.Document.Content[:50])
	}
}

