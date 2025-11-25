package retriever

import (
	"context"
	"fmt"

	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/config"
	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/reranker"
	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/schema"
)

// RetrieverOptions contains options for retrieval
type RetrieverOptions struct {
	TopK      int     // Number of results to return
	Threshold float64 // Score threshold for filtering
	Rerank    bool    // Whether to enable reranking
}

// Retriever handles document retrieval with optional reranking
type Retriever struct {
	ragClient   RAGClientInterface
	reranker    *reranker.RerankerClient
	config      *config.Config
	defaultTopK int
	rerankTopK  int // Number of candidates to retrieve before reranking
}

// RAGClientInterface defines the interface for RAG client operations
type RAGClientInterface interface {
	SearchChunks(query string, topK int, threshold float64) ([]schema.SearchResult, error)
}

// NewRetriever creates a new retriever instance
func NewRetriever(ragClient RAGClientInterface, cfg *config.Config) (*Retriever, error) {
	r := &Retriever{
		ragClient:   ragClient,
		config:      cfg,
		defaultTopK: cfg.RAG.TopK,
		rerankTopK:  cfg.RAG.RerankTopK,
	}

	// Initialize reranker if enabled
	if cfg.Reranker.Enabled {
		rerankerClient, err := reranker.NewRerankerClient(&cfg.Reranker)
		if err != nil {
			return nil, fmt.Errorf("failed to create reranker client: %w", err)
		}
		r.reranker = rerankerClient
	}

	// Set default rerankTopK if not configured
	if r.rerankTopK == 0 {
		r.rerankTopK = 20 // Default to 20 candidates for reranking, same as Python implementation
	}

	return r, nil
}

// Retrieve retrieves documents based on query, with optional reranking
// This implements the same logic as Python's simple_retrieval.py retriever
func (r *Retriever) Retrieve(ctx context.Context, query string, options *RetrieverOptions) ([]schema.SearchResult, error) {
	if options == nil {
		options = &RetrieverOptions{
			TopK:      r.defaultTopK,
			Threshold: r.config.RAG.Threshold,
			Rerank:    r.config.RAG.Rerank,
		}
	}

	// Determine if reranking should be used
	useRerank := options.Rerank && r.reranker != nil

	var results []schema.SearchResult
	var err error

	if useRerank {
		// If reranking is enabled, retrieve more candidates first (similar to Python's similarity_top_k=20)
		// Then rerank to top_k
		searchTopK := r.rerankTopK
		if searchTopK < options.TopK {
			searchTopK = options.TopK * 2 // Ensure we have enough candidates
		}

		results, err = r.ragClient.SearchChunks(query, searchTopK, options.Threshold)
		if err != nil {
			return nil, fmt.Errorf("search chunks failed: %w", err)
		}

		// Rerank the results
		threshold := options.Threshold
		if r.config.Reranker.Threshold > 0 {
			threshold = r.config.Reranker.Threshold
		}

		results, err = r.reranker.RerankSearchResults(ctx, query, results, options.TopK, threshold)
		if err != nil {
			return nil, fmt.Errorf("rerank failed: %w", err)
		}
	} else {
		// Direct retrieval without reranking
		results, err = r.ragClient.SearchChunks(query, options.TopK, options.Threshold)
		if err != nil {
			return nil, fmt.Errorf("search chunks failed: %w", err)
		}
	}

	return results, nil
}

// RetrieveWithDefaults retrieves documents using default configuration
func (r *Retriever) RetrieveWithDefaults(ctx context.Context, query string) ([]schema.SearchResult, error) {
	return r.Retrieve(ctx, query, nil)
}
