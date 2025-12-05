package websearch

import (
	"fmt"

	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/config"
)

// Provider type constants for different web search services
const (
	// Google Custom Search API
	PROVIDER_TYPE_GOOGLE = "google"
	// DuckDuckGo search service
	PROVIDER_TYPE_DUCKDUCKGO = "duckduckgo"
)

// Factory interface for creating Provider instances
type providerInitializer interface {
	// Creates a new Provider with the given configuration
	CreateProvider(config.WebSearchConfig) (InternetSearchProvider, error)
}

// Maps provider types to their initializers
var (
	providerInitializers = map[string]providerInitializer{
		PROVIDER_TYPE_GOOGLE:     &googleProviderInitializer{},
		PROVIDER_TYPE_DUCKDUCKGO: &duckDuckGoProviderInitializer{},
	}
)

// Creates a new web search Provider based on the configuration
// Returns error if provider type is not supported
func NewWebSearchProvider(cfg config.WebSearchConfig) (InternetSearchProvider, error) {
	initializer, ok := providerInitializers[cfg.Provider]
	if !ok {
		return nil, fmt.Errorf("no initializer found for provider type: %s", cfg.Provider)
	}
	return initializer.CreateProvider(cfg)
}
