package websearch

import (
	"context"
)

// InternetSearchResult represents a result from internet search
type InternetSearchResult struct {
	Title   string
	Link    string
	Content string
	Score   float64 // Relevance score (if available)
}

// InternetSearchProvider defines the interface for internet/web search providers
type InternetSearchProvider interface {
	// Search performs an internet search for the given query
	// Returns search results and any error encountered
	Search(ctx context.Context, query string, maxResults int) ([]InternetSearchResult, error)
}

// InternetSearchConfig holds configuration for internet search
type InternetSearchConfig struct {
	Enabled    bool // Whether internet search is enabled
	MaxResults int  // Maximum number of results to return per query
	Provider   InternetSearchProvider
}

// DefaultInternetSearchConfig returns default internet search configuration
func DefaultInternetSearchConfig() *InternetSearchConfig {
	return &InternetSearchConfig{
		Enabled:    false,
		MaxResults: 5,
		Provider:   nil,
	}
}
