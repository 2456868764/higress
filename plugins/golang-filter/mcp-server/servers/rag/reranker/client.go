package reranker

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/common"
	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/config"
	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/schema"
)

// RerankerClient handles reranking operations
type RerankerClient struct {
	httpClient *common.HTTPClient
	config     *config.RerankerConfig
}

// RerankRequest represents the request to rerank API
type RerankRequest struct {
	Query           string   `json:"query"`
	Documents       []string `json:"documents"`
	TopN            *int     `json:"top_n,omitempty"`
	ReturnDocuments *bool    `json:"return_documents,omitempty"`
	Threshold       *float64 `json:"threshold,omitempty"`
}

// RerankResult represents a single rerank result
type RerankResult struct {
	Index          int     `json:"index"`
	RelevanceScore float64 `json:"relevance_score"`
	Document       string  `json:"document,omitempty"`
}

// RerankResponse represents the response from rerank API
type RerankResponse struct {
	Results []RerankResult         `json:"results"`
	Model   string                 `json:"model"`
	Usage   map[string]interface{} `json:"usage"`
}

// NewRerankerClient creates a new reranker client
// Note: The caller should check RAGConfig.Rerank before calling this function
func NewRerankerClient(cfg *config.RerankerConfig) (*RerankerClient, error) {
	if cfg == nil || cfg.BaseURL == "" {
		return nil, fmt.Errorf("reranker base URL is not configured")
	}

	headers := make(map[string]string)
	if cfg.APIKey != "" {
		headers["Authorization"] = fmt.Sprintf("Bearer %s", cfg.APIKey)
	}

	httpClient := common.NewHTTPClient(cfg.BaseURL, headers)

	return &RerankerClient{
		httpClient: httpClient,
		config:     cfg,
	}, nil
}

// Rerank reranks documents based on query relevance
func (c *RerankerClient) Rerank(ctx context.Context, query string, documents []string, topN int, threshold float64) ([]RerankResult, error) {
	if len(documents) == 0 {
		return []RerankResult{}, nil
	}

	returnDocuments := true
	req := RerankRequest{
		Query:           query,
		Documents:       documents,
		TopN:            &topN,
		ReturnDocuments: &returnDocuments,
		Threshold:       &threshold,
	}

	respBody, err := c.httpClient.Post("/rerank", req)
	if err != nil {
		return nil, fmt.Errorf("rerank API request failed: %w", err)
	}

	var response RerankResponse
	if err := json.Unmarshal(respBody, &response); err != nil {
		return nil, fmt.Errorf("failed to unmarshal rerank response: %w", err)
	}

	return response.Results, nil
}

// RerankSearchResults reranks SearchResult objects and returns reranked SearchResults
func (c *RerankerClient) RerankSearchResults(ctx context.Context, query string, results []schema.SearchResult, topN int, threshold float64) ([]schema.SearchResult, error) {
	if len(results) == 0 {
		return []schema.SearchResult{}, nil
	}

	// Extract document contents
	documents := make([]string, 0, len(results))
	for _, result := range results {
		documents = append(documents, result.Document.Content)
	}

	// Call rerank API
	rerankResults, err := c.Rerank(ctx, query, documents, topN, threshold)
	if err != nil {
		return nil, err
	}

	// Map rerank results back to SearchResult
	reranked := make([]schema.SearchResult, 0, len(rerankResults))
	for _, rr := range rerankResults {
		if rr.Index < len(results) {
			// Create a new SearchResult with reranked score
			rerankedResult := schema.SearchResult{
				Document: results[rr.Index].Document,
				Score:    rr.RelevanceScore,
			}
			reranked = append(reranked, rerankedResult)
		}
	}

	return reranked, nil
}
