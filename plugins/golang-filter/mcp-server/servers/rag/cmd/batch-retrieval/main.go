package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag"
	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/config"
	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/schema"
)

// QueryData represents a single query entry from MultiHopRAG.json
type QueryData struct {
	Query        string        `json:"query"`
	Answer       string        `json:"answer"`
	QuestionType string        `json:"question_type"`
	EvidenceList []interface{} `json:"evidence_list"`
}

// RetrievalItem represents a single retrieval result item
type RetrievalItem struct {
	Text  string  `json:"text"`
	Score float64 `json:"score"`
}

// RetrievalResult represents the saved retrieval result for a query
type RetrievalResult struct {
	Query         string          `json:"query"`
	Answer        string          `json:"answer"`
	QuestionType  string          `json:"question_type"`
	RetrievalList []RetrievalItem `json:"retrieval_list"`
	GoldList      []interface{}   `json:"gold_list"`
}

// RetrieverOptions contains options for retrieval
// Note: Rerank option is now handled by RAGClient.SearchChunks based on config.RAG.Rerank
// This option is kept for future extensibility but currently not used
type RetrieverOptions struct {
	TopK      int     // Number of results to return
	Threshold float64 // Score threshold for filtering
	Rerank    bool    // Whether to enable reranking (deprecated: use config.RAG.Rerank instead)
}

// Retriever handles document retrieval
// Reranking is now handled internally by RAGClient.SearchChunks based on configuration
type Retriever struct {
	ragClient   *rag.RAGClient
	config      *config.Config
	defaultTopK int
}

// NewRetriever creates a new retriever instance
// The reranking logic is now handled by RAGClient.SearchChunks based on config.RAG.Rerank
func NewRetriever(ragClient *rag.RAGClient, cfg *config.Config) (*Retriever, error) {
	r := &Retriever{
		ragClient:   ragClient,
		config:      cfg,
		defaultTopK: cfg.RAG.TopK,
	}

	return r, nil
}

// Retrieve retrieves documents based on query
// Reranking is automatically handled by RAGClient.SearchChunks if config.RAG.Rerank is enabled
// This implements the same logic as Python's simple_retrieval.py retriever
func (r *Retriever) Retrieve(ctx context.Context, query string, options *RetrieverOptions) ([]schema.SearchResult, error) {
	if options == nil {
		options = &RetrieverOptions{
			TopK:      r.defaultTopK,
			Threshold: r.config.RAG.Threshold,
			Rerank:    r.config.RAG.Rerank,
		}
	}

	// RAGClient.SearchChunks now handles reranking internally based on config.RAG.Rerank
	// We just need to call it with the desired topK and threshold
	results, err := r.ragClient.SearchChunks(query, options.TopK, options.Threshold)
	if err != nil {
		return nil, fmt.Errorf("search chunks failed: %w", err)
	}

	return results, nil
}

// RetrieveWithDefaults retrieves documents using default configuration
func (r *Retriever) RetrieveWithDefaults(ctx context.Context, query string) ([]schema.SearchResult, error) {
	return r.Retrieve(ctx, query, nil)
}

// BatchRetrieval performs batch retrieval on query data, similar to Python's simple_retrieval.py
// It reads queries from inputFile, performs retrieval with optional reranking,
// and saves results to outputFile
func BatchRetrieval(
	ctx context.Context,
	ragClient *rag.RAGClient,
	cfg *config.Config,
	inputFile string,
	outputFile string,
) error {
	// 1. Load query data from JSON file
	queryData, err := loadQueryData(inputFile)
	if err != nil {
		return fmt.Errorf("failed to load query data: %w", err)
	}

	fmt.Printf("Loaded %d queries from %s\n", len(queryData), inputFile)
	fmt.Println("Starting retrieval...")
	topK := cfg.RAG.TopK
	threshold := cfg.RAG.Threshold
	// 3. Process each query
	retrievalSaveList := make([]RetrievalResult, 0, len(queryData))
	for i, data := range queryData {
		if (i+1)%100 == 0 {
			fmt.Printf("Processed %d/%d queries...\n", i+1, len(queryData))
		}

		query := data.Query
		// Perform retrieval (reranking is handled by SearchChunks based on config)
		results, err := ragClient.SearchChunks(query, topK, threshold)
		// fmt.Printf("search results count: %d\n", len(results))
		if err != nil {
			return fmt.Errorf("retrieval failed for query '%s': %w", query, err)
		}

		// Build retrieval list
		retrievalList := make([]RetrievalItem, 0, len(results))
		for _, result := range results {
			retrievalList = append(retrievalList, RetrievalItem{
				Text:  result.Document.Content,
				Score: result.Score,
			})
		}

		// Build save result
		save := RetrievalResult{
			Query:         data.Query,
			Answer:        data.Answer,
			QuestionType:  data.QuestionType,
			RetrievalList: retrievalList,
			GoldList:      data.EvidenceList,
		}

		retrievalSaveList = append(retrievalSaveList, save)
	}

	// 5. Save results to JSON file
	if err := saveRetrievalResults(outputFile, retrievalSaveList); err != nil {
		return fmt.Errorf("failed to save results: %w", err)
	}

	fmt.Printf("Retrieval completed. Saved %d results to %s\n", len(retrievalSaveList), outputFile)
	return nil
}

// loadQueryData loads query data from a JSON file
func loadQueryData(filePath string) ([]QueryData, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	var queryData []QueryData
	if err := json.Unmarshal(data, &queryData); err != nil {
		return nil, fmt.Errorf("failed to unmarshal JSON: %w", err)
	}

	return queryData, nil
}

// saveRetrievalResults saves retrieval results to a JSON file
func saveRetrievalResults(filePath string, results []RetrievalResult) error {
	// Create output directory if it doesn't exist
	if err := os.MkdirAll(getDir(filePath), 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	data, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal JSON: %w", err)
	}

	if err := os.WriteFile(filePath, data, 0644); err != nil {
		return fmt.Errorf("failed to write file: %w", err)
	}

	return nil
}

func getDir(path string) string {
	for i := len(path) - 1; i >= 0; i-- {
		if path[i] == '/' || path[i] == '\\' {
			return path[:i]
		}
	}
	return "."
}

func main() {
	// Define command line flags
	var (
		inputFile    = flag.String("input", "/Users/jun/GolandProjects/higress/higress/plugins/golang-filter/mcp-server/servers/rag/python/dataset/MultiHopRAG.json", "Input JSON file containing queries")
		outputFile   = flag.String("output", "output/retrieval_hybrid_re_500.json", "Output JSON file for retrieval results")
		agent        = flag.String("agent", "default", "rag agent type: default, chain_of_rag, router")
		topK         = flag.Int("topk", 10, "Number of top results to return")
		threshold    = flag.Float64("threshold", 0.0, "Score threshold for filtering")
		rerank       = flag.Bool("rerank", false, "Enable reranking")
		collection   = flag.String("collection", "corpus_collection_500", "collection name")
		rerankTopK   = flag.Int("rerank_topk", 20, "Number of candidates to retrieve before reranking")
		hybridSearch = flag.Bool("hybrid_search", true, "Enable hybrid search")
	)

	flag.Parse()

	// Print usage if help is requested
	if len(os.Args) > 1 && (os.Args[1] == "-h" || os.Args[1] == "--help") {
		fmt.Println("RAG Batch Retrieval Tool")
		fmt.Println("\nUsage:")
		flag.PrintDefaults()
		fmt.Println("\nExample:")
		fmt.Println("  go run cmd/batch-retrieval/main.go -input dataset/MultiHopRAG.json -output output/results.json -rerank -topk 10")
		fmt.Println("\nEnvironment Variables:")
		fmt.Println("  OPENAI_API_KEY       - OpenAI API key (required)")
		fmt.Println("  OPENAI_BASE_URL      - OpenAI API base URL (optional)")
		fmt.Println("  MILVUS_HOST          - Milvus host (default: localhost)")
		fmt.Println("  MILVUS_PORT          - Milvus port (default: 19530)")
		fmt.Println("  MILVUS_DATABASE      - Milvus database name (default: default)")
		fmt.Println("  MILVUS_COLLECTION    - Milvus collection name (default: corpus_collection_500)")
		fmt.Println("  MILVUS_USERNAME      - Milvus username (optional)")
		fmt.Println("  MILVUS_PASSWORD      - Milvus password (optional)")
		return
	}

	// Create configuration from command line flags
	cfg := &config.Config{
		RAG: config.RAGConfig{
			Splitter: config.SplitterConfig{
				Provider:       "recursive",
				ChunkSize:      500,
				ChunkOverlap:   50,
				SmallChunkSize: 0,
			},
			TopK:       *topK,
			Threshold:  *threshold,
			Rerank:     *rerank,
			RerankTopK: *rerankTopK,
			Agent:      *agent,
		},

		LLM: config.LLMConfig{
			Provider: "openai",
			APIKey:   getEnvOrDefault("OPENAI_API_KEY", "sk-xxx"),
			BaseURL:  getEnvOrDefault("OPENAI_BASE_URL", "http://localhost:8090/v1"),
			Model:    "gpt-4o",
		},
		Reranker: config.RerankerConfig{
			BaseURL:   getEnvOrDefault("RERANKER_BASE_URL", "http://localhost:8090/v1"),
			APIKey:    getEnvOrDefault("RERANKER_API_KEY", "sk-xxx"),
			Threshold: 0.0,
		},
		// Set default values for required fields
		Embedding: config.EmbeddingConfig{
			Provider:   "openai",
			APIKey:     getEnvOrDefault("OPENAI_API_KEY", "sk-xxx"),
			BaseURL:    getEnvOrDefault("OPENAI_BASE_URL", "http://localhost:8090/v1"),
			Model:      "Qwen3-Embedding-0.6B",
			Dimensions: 1024,
		},
		VectorDB: config.VectorDBConfig{
			Provider:   "milvus",
			Host:       "localhost",
			Port:       19530,
			Database:   "default",
			Collection: *collection,
			HybridSearch: config.HybridSearchConfig{
				Enabled:      *hybridSearch,
				Ranker:       config.RFRanker,
				VectorWeight: 0.5,
			},
			Mapping: config.MappingConfig{
				Fields: []config.FieldMapping{
					{
						StandardName: "id",
						RawName:      "id",
						Properties: map[string]interface{}{
							"max_length": 256,
							"auto_id":    false,
						},
					},
					{
						StandardName: "content",
						RawName:      "content",
						Properties: map[string]interface{}{
							"max_length": 2048,
						},
					},
					{
						StandardName: "vector",
						RawName:      "vector",
						Properties:   make(map[string]interface{}),
					},

					{
						StandardName: "sparse_vector",
						RawName:      "sparse_vector",
						Properties:   make(map[string]interface{}),
					},
					{
						StandardName: "metadata",
						RawName:      "metadata",
						Properties:   make(map[string]interface{}),
					},
				},
				Index: config.IndexConfig{
					IndexType: "HNSW",
					Params:    map[string]interface{}{"M": 8, "efConstruction": 64},
				},
				Search: config.SearchConfig{
					MetricType: "IP",
					Params:     map[string]interface{}{"nprobe": 32},
				},
			},
		},
	}
	// Validate required configuration
	if cfg.Embedding.APIKey == "" {
		fmt.Fprintf(os.Stderr, "Error: OPENAI_API_KEY environment variable is required\n")
		fmt.Fprintf(os.Stderr, "Set it with: export OPENAI_API_KEY=your-api-key\n")
		os.Exit(1)
	}

	if *rerank && cfg.Reranker.BaseURL == "" {
		fmt.Fprintf(os.Stderr, "Error: reranker URL is required when rerank is enabled\n")
		os.Exit(1)
	}

	// Create RAG client
	fmt.Println("Initializing RAG client...")
	ragClient, err := rag.NewRAGClient(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create RAG client: %v\n", err)
		os.Exit(1)
	}

	// Perform batch retrieval
	ctx := context.Background()
	fmt.Printf("Starting batch retrieval:\n")
	fmt.Printf("  Agent: %s\n", *agent)
	fmt.Printf("  Input file: %s\n", *inputFile)
	fmt.Printf("  Output file: %s\n", *outputFile)
	fmt.Printf("  TopK: %d\n", *topK)
	fmt.Printf("  Threshold: %.2f\n", *threshold)
	fmt.Printf("  Rerank: %v\n", *rerank)
	fmt.Printf("  Hybrid Search: %v\n", *hybridSearch)
	fmt.Printf("  RerankTopK: %d\n", *rerankTopK)
	fmt.Printf("  Collection: %s\n", *collection)
	fmt.Println()

	if err := BatchRetrieval(ctx, ragClient, cfg, *inputFile, *outputFile); err != nil {
		fmt.Fprintf(os.Stderr, "Batch retrieval failed: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("Batch retrieval completed successfully!")
}

// Helper functions for environment variables
func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvIntOrDefault(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		var result int
		if _, err := fmt.Sscanf(value, "%d", &result); err == nil {
			return result
		}
	}
	return defaultValue
}
