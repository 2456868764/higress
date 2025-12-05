package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sync"
	"sync/atomic"

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

// Task represents a single retrieval task
type Task struct {
	Index int
	Data  QueryData
}

// TaskResult represents the result of a retrieval task
type TaskResult struct {
	Index  int
	Result RetrievalResult
	Error  error
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
// workers: number of concurrent workers (0 or 1 means sequential processing)
// skipQuery: number of queries to skip from the beginning (default: 0)
func BatchRetrieval(
	ctx context.Context,
	ragClient *rag.RAGClient,
	cfg *config.Config,
	inputFile string,
	outputFile string,
	maxQuery int,
	skipQuery int,
	workers int,
) error {
	// 1. Load query data from JSON file
	queryData, err := loadQueryData(inputFile)
	if err != nil {
		return fmt.Errorf("failed to load query data: %w", err)
	}

	fmt.Printf("Loaded %d queries from %s\n", len(queryData), inputFile)

	// Skip queries based on skipQuery
	if skipQuery > 0 {
		if skipQuery >= len(queryData) {
			return fmt.Errorf("skip_query (%d) is greater than or equal to total queries (%d)", skipQuery, len(queryData))
		}
		queryData = queryData[skipQuery:]
		fmt.Printf("Skipped %d queries, remaining %d queries\n", skipQuery, len(queryData))
	}

	// Filter queries based on maxQuery
	if maxQuery > 0 && maxQuery < len(queryData) {
		queryData = queryData[:maxQuery]
		fmt.Printf("Limited to %d queries (max_query=%d)\n", len(queryData), maxQuery)
	}

	// Use sequential processing if workers <= 1
	if workers <= 1 {
		return batchRetrievalSequential(ctx, ragClient, cfg, queryData, outputFile)
	}

	// Use concurrent processing
	return batchRetrievalConcurrent(ctx, ragClient, cfg, queryData, outputFile, workers)
}

// batchRetrievalSequential performs sequential batch retrieval
func batchRetrievalSequential(
	ctx context.Context,
	ragClient *rag.RAGClient,
	cfg *config.Config,
	queryData []QueryData,
	outputFile string,
) error {
	fmt.Println("Starting sequential retrieval...")
	topK := cfg.RAG.TopK
	threshold := cfg.RAG.Threshold
	retrievalSaveList := make([]RetrievalResult, 0, len(queryData))
	const saveInterval = 10 // Save every 10 queries

	for i, data := range queryData {
		fmt.Printf("[BatchRetrieval] Processing query %d: %s\n", i+1, data.Query)
		// Progress logging every 100 queries
		if (i+1)%100 == 0 {
			fmt.Printf("Processed %d/%d queries...\n", i+1, len(queryData))
		}

		query := data.Query
		// Perform retrieval (reranking is handled by SearchChunks based on config)
		results, err := ragClient.SearchChunks(query, topK, threshold)
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

		// Save every 10 queries to prevent data loss
		if (i+1)%saveInterval == 0 {
			if err := saveRetrievalResults(outputFile, retrievalSaveList); err != nil {
				return fmt.Errorf("failed to save intermediate results at query %d: %w", i+1, err)
			}
			fmt.Printf("✓ Saved intermediate results: %d/%d queries processed\n", i+1, len(queryData))
		}
	}

	// Final save: Save all remaining results to JSON file
	if len(retrievalSaveList) > 0 {
		if err := saveRetrievalResults(outputFile, retrievalSaveList); err != nil {
			return fmt.Errorf("failed to save final results: %w", err)
		}
		fmt.Printf("✓ Final save completed. Total %d results saved to %s\n", len(retrievalSaveList), outputFile)
	}
	return nil
}

// batchRetrievalConcurrent performs concurrent batch retrieval using worker pool
func batchRetrievalConcurrent(
	ctx context.Context,
	ragClient *rag.RAGClient,
	cfg *config.Config,
	queryData []QueryData,
	outputFile string,
	workers int,
) error {
	fmt.Printf("Starting concurrent retrieval with %d workers...\n", workers)
	topK := cfg.RAG.TopK
	threshold := cfg.RAG.Threshold
	const saveInterval = 10 // Save every 10 queries

	// Create channels
	taskChan := make(chan Task, workers*2)
	resultChan := make(chan TaskResult, workers*2)

	// Track processed count atomically
	var processedCount int64
	var errorCount int64
	totalQueries := len(queryData)

	// Mutex for protecting resultMap
	var mu sync.Mutex
	resultMap := make(map[int]RetrievalResult) // Map to store results by index for ordering

	// Start worker goroutines
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for task := range taskChan {
				// Process retrieval
				results, err := ragClient.SearchChunks(task.Data.Query, topK, threshold)
				if err != nil {
					atomic.AddInt64(&errorCount, 1)
					resultChan <- TaskResult{
						Index: task.Index,
						Error: fmt.Errorf("retrieval failed for query '%s': %w", task.Data.Query, err),
					}
					atomic.AddInt64(&processedCount, 1)
					continue
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
					Query:         task.Data.Query,
					Answer:        task.Data.Answer,
					QuestionType:  task.Data.QuestionType,
					RetrievalList: retrievalList,
					GoldList:      task.Data.EvidenceList,
				}

				resultChan <- TaskResult{
					Index:  task.Index,
					Result: save,
				}
				atomic.AddInt64(&processedCount, 1)

				// Progress logging
				count := atomic.LoadInt64(&processedCount)
				if count%100 == 0 {
					fmt.Printf("[Worker %d] Processed %d/%d queries...\n", workerID, count, totalQueries)
				}
			}
		}(i)
	}

	// Start result collector goroutine
	var collectorWg sync.WaitGroup
	collectorWg.Add(1)
	go func() {
		defer collectorWg.Done()
		for result := range resultChan {
			if result.Error != nil {
				fmt.Printf("[BatchRetrieval] Error processing query %d: %v\n", result.Index+1, result.Error)
				continue
			}

			mu.Lock()
			resultMap[result.Index] = result.Result
			currentCount := len(resultMap)
			mu.Unlock()

			// Save every saveInterval results
			if currentCount%saveInterval == 0 {
				mu.Lock()
				// Build ordered list from map
				orderedList := make([]RetrievalResult, 0, currentCount)
				for i := 0; i < totalQueries; i++ {
					if result, ok := resultMap[i]; ok {
						orderedList = append(orderedList, result)
					}
				}
				mu.Unlock()

				if len(orderedList) > 0 {
					if err := saveRetrievalResults(outputFile, orderedList); err != nil {
						fmt.Printf("[BatchRetrieval] Failed to save intermediate results: %v\n", err)
					} else {
						count := atomic.LoadInt64(&processedCount)
						fmt.Printf("✓ Saved intermediate results: %d results (processed %d/%d queries)\n", len(orderedList), count, totalQueries)
					}
				}
			}
		}
	}()

	// Send tasks to workers
	for i, data := range queryData {
		taskChan <- Task{
			Index: i,
			Data:  data,
		}
		fmt.Printf("[BatchRetrieval] Queued query %d: %s\n", i+1, data.Query)
	}
	close(taskChan)

	// Wait for all workers to complete
	wg.Wait()
	close(resultChan)

	// Wait for collector to finish
	collectorWg.Wait()

	// Final save: Save all remaining results
	mu.Lock()
	orderedList := make([]RetrievalResult, 0, len(resultMap))
	for i := 0; i < totalQueries; i++ {
		if result, ok := resultMap[i]; ok {
			orderedList = append(orderedList, result)
		}
	}
	mu.Unlock()

	if len(orderedList) > 0 {
		if err := saveRetrievalResults(outputFile, orderedList); err != nil {
			return fmt.Errorf("failed to save final results: %w", err)
		}
		fmt.Printf("✓ Final save completed. Total %d results saved to %s\n", len(orderedList), outputFile)
	}

	errors := atomic.LoadInt64(&errorCount)
	if errors > 0 {
		fmt.Printf("Warning: %d queries failed during processing\n", errors)
	}

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
		outputFile   = flag.String("output", "output/retrieval_MultiHopRAG_01_500.json", "Output JSON file for retrieval results")
		agent        = flag.String("agent", "default", "rag agent type: default, chain_of_rag, deep_search, router")
		topK         = flag.Int("topk", 10, "Number of top results to return")
		threshold    = flag.Float64("threshold", 0.0, "Score threshold for filtering")
		rerank       = flag.Bool("rerank", false, "Enable reranking")
		maxQuery     = flag.Int("max_query", 0, "max query to excute, 0 means no limit")
		skipQuery    = flag.Int("skip_query", 0, "number of queries to skip from the beginning, default is 0")
		workers      = flag.Int("workers", 1, "Number of concurrent workers (0 or 1 means sequential processing)")
		collection   = flag.String("collection", "corpus_collection_500", "collection name")
		rerankTopK   = flag.Int("rerank_topk", 20, "Number of candidates to retrieve before reranking")
		hybridSearch = flag.Bool("hybrid_search", false, "Enable hybrid search")
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

		WebSearch: config.WebSearchConfig{
			Enabled:    false,
			Provider:   "google",
			APIKey:     "",
			CX:         "",
			MaxResults: 5,
		},

		// LLM: config.LLMConfig{
		// 	Provider: "openai",
		// 	APIKey:   getEnvOrDefault("OPENAI_API_KEY", "sk-xxx"),
		// 	BaseURL:  getEnvOrDefault("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
		// 	Model:    "qwen-plus",
		// },

		LLM: config.LLMConfig{
			Provider: "openai",
			APIKey:   getEnvOrDefault("OPENAI_API_KEY", "sk-44f9a216d01345b09e63f2bcc370b7af"),
			BaseURL:  getEnvOrDefault("OPENAI_BASE_URL", "https://api.deepseek.com"),
			Model:    "deepseek-reasoner",
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
	fmt.Printf("  Max query: %d\n", *maxQuery)
	fmt.Printf("  Skip query: %d\n", *skipQuery)
	fmt.Printf("  Workers: %d\n", *workers)
	fmt.Printf("  TopK: %d\n", *topK)
	fmt.Printf("  Threshold: %.2f\n", *threshold)
	fmt.Printf("  Rerank: %v\n", *rerank)
	fmt.Printf("  Hybrid Search: %v\n", *hybridSearch)
	fmt.Printf("  RerankTopK: %d\n", *rerankTopK)
	fmt.Printf("  Collection: %s\n", *collection)
	fmt.Println()

	if err := BatchRetrieval(ctx, ragClient, cfg, *inputFile, *outputFile, *maxQuery, *skipQuery, *workers); err != nil {
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
