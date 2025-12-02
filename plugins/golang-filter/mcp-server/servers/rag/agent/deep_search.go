package agent

import (
	"context"
	"fmt"
	"regexp"
	"strings"

	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/embedding"
	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/llm"
	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/schema"
	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/vectordb"
	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/websearch"
)

const (
	// Prompt templates for DeepSearch
	subQueryPrompt = `To answer this question more comprehensively, please break down the original question into up to four sub-questions. Return as list of str.
If this is a very simple question and no decomposition is necessary, then keep the only one original question in the python code list.

Original Question: %s

<EXAMPLE>
Example input:
"Explain deep learning"

Example output:
[
    "What is deep learning?",
    "What is the difference between deep learning and machine learning?",
    "What is the history of deep learning?"
]
</EXAMPLE>

Provide your response in a python code list of str format:`

	rerankPrompt = `Based on the query questions and the retrieved chunk, to determine whether the chunk is helpful in answering any of the query question, you can only return "YES" or "NO", without any other information.

Query Questions: %s
Retrieved Chunk: %s

Is the chunk helpful in answering the any of the questions?`

	reflectPrompt = `Determine whether additional search queries are needed based on the original query, previous sub queries, and all retrieved document chunks. If further research is required, provide a Python list of up to 3 search queries. If no further research is required, return an empty list.

If the original query is to write a report, then you prefer to generate some further queries, instead return an empty list.

Original Query: %s

Previous Sub Queries: %s

Related Chunks: 
%s

Respond exclusively in valid List of str format without any other text.`

	summaryPrompt = `You are a AI content analysis expert, good at summarizing content. Please summarize a specific and detailed answer or report based on the previous queries and the retrieved document chunks.

Original Query: %s

Previous Sub Queries: %s

Related Chunks: 
%s
`
)

// DeepSearchDescription is the description for DeepSearch agent.
// This agent is suitable for handling general and simple queries, such as given a topic and then writing a report, survey, or article.
var DeepSearchDescription = DescribeAgent(
	"This agent is suitable for handling general and simple queries, such as given a topic and then writing a report, survey, or article.",
)

// DeepSearch implements Deep Search agent for comprehensive information retrieval.
// This agent performs a thorough search through the knowledge base, analyzing
// multiple aspects of the query to provide comprehensive and detailed answers.
type DeepSearch struct {
	llm             llm.Provider
	embeddingModel  embedding.Provider
	vectorDB        vectordb.VectorStoreProvider
	config          *DefaultRAGConfig
	maxIter         int
	textWindowSplit bool
	internetSearch  *websearch.InternetSearchConfig // Internet search configuration
}

// Ensure DeepSearch implements RAGAgent interface at compile time
var _ RAGAgent = (*DeepSearch)(nil)

// NewDeepSearch creates a new DeepSearch agent
func NewDeepSearch(
	llm llm.Provider,
	embeddingModel embedding.Provider,
	vectorDB vectordb.VectorStoreProvider,
	config *DefaultRAGConfig,
	maxIter int,
	textWindowSplit bool,
	internetSearch *websearch.InternetSearchConfig,
) (*DeepSearch, error) {
	if config == nil {
		config = DefaultRAGConfigWithDefaults()
	}
	if internetSearch == nil {
		internetSearch = websearch.DefaultInternetSearchConfig()
	}
	return &DeepSearch{
		llm:             llm,
		embeddingModel:  embeddingModel,
		vectorDB:        vectorDB,
		config:          config,
		maxIter:         maxIter,
		textWindowSplit: textWindowSplit,
		internetSearch:  internetSearch,
	}, nil
}

// Description returns the description of DeepSearch agent
// This implements the DescribableAgent interface for use with RAGRouter
func (d *DeepSearch) Description() string {
	return DeepSearchDescription.Description
}

// Invoke executes the agent with the given query and returns the result.
// This implements the BaseAgent interface.
func (d *DeepSearch) Invoke(ctx context.Context, query string, kwargs map[string]interface{}) (interface{}, error) {
	answer, _, _, err := d.Query(ctx, query, kwargs)
	if err != nil {
		return nil, err
	}
	return answer, nil
}

// _generateSubQueries generates sub-queries from the original query
func (d *DeepSearch) _generateSubQueries(ctx context.Context, originalQuery string) ([]string, int, error) {
	prompt := fmt.Sprintf(subQueryPrompt, originalQuery)
	response, err := d.llm.Chat(ctx, []llm.ChatMessage{
		{Role: "user", Content: prompt},
	})
	if err != nil {
		return nil, 0, err
	}

	subQueries, err := LiteralEvalString(response.Content)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to parse sub queries: %w", err)
	}

	return subQueries, response.TotalTokens, nil
}

// _searchChunksFromVectorDB searches chunks from vector database and reranks them using LLM
func (d *DeepSearch) _searchChunksFromVectorDB(ctx context.Context, query string, subQueries []string) ([]RetrievalResult, int, error) {
	// Get embedding for the query
	queryVector, err := d.embeddingModel.GetEmbedding(ctx, query)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to get embedding: %w", err)
	}

	// Search in vector database
	searchOptions := &schema.SearchOptions{
		TopK:      d.config.TopK,
		Threshold: d.config.Threshold,
	}

	searchResults, err := d.vectorDB.SearchDocs(ctx, query, queryVector, searchOptions)
	if err != nil {
		return nil, 0, fmt.Errorf("search failed: %w", err)
	}

	// Rerank chunks using LLM
	allRetrievedResults := make([]RetrievalResult, 0)
	totalTokens := 0

	// Combine query and sub queries for reranking
	allQueries := make([]string, 0, len(subQueries)+1)
	allQueries = append(allQueries, query)
	allQueries = append(allQueries, subQueries...)
	queriesStr := fmt.Sprintf("%v", allQueries)

	for _, result := range searchResults {
		chunkText := fmt.Sprintf("<chunk>%s</chunk>", result.Document.Content)
		rerankPromptText := fmt.Sprintf(rerankPrompt, queriesStr, chunkText)

		response, err := d.llm.Chat(ctx, []llm.ChatMessage{
			{Role: "user", Content: rerankPromptText},
		})
		if err != nil {
			continue // Skip this chunk if reranking fails
		}
		totalTokens += response.TotalTokens

		responseContent := strings.ToUpper(strings.TrimSpace(response.Content))
		if strings.Contains(responseContent, "YES") && !strings.Contains(responseContent, "NO") {
			allRetrievedResults = append(allRetrievedResults, RetrievalResult{
				Text:     result.Document.Content,
				Score:    result.Score,
				Document: result.Document,
				Metadata: result.Document.Metadata,
			})
		}
	}

	return allRetrievedResults, totalTokens, nil
}

// _generateGapQueries generates gap queries based on reflection
func (d *DeepSearch) _generateGapQueries(ctx context.Context, originalQuery string, allSubQueries []string, allChunks []RetrievalResult) ([]string, int, error) {
	chunkTexts := make([]string, 0, len(allChunks))
	for _, chunk := range allChunks {
		chunkTexts = append(chunkTexts, chunk.Text)
	}
	chunkStr := d._formatChunkTexts(chunkTexts)
	if len(chunkStr) == 0 {
		chunkStr = "NO RELATED CHUNKS FOUND."
	}

	miniQuestionsStr := fmt.Sprintf("%v", allSubQueries)
	prompt := fmt.Sprintf(reflectPrompt, originalQuery, miniQuestionsStr, chunkStr)

	response, err := d.llm.Chat(ctx, []llm.ChatMessage{
		{Role: "user", Content: prompt},
	})
	if err != nil {
		return nil, 0, err
	}

	gapQueries, err := LiteralEvalString(response.Content)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to parse gap queries: %w", err)
	}

	return gapQueries, response.TotalTokens, nil
}

// _formatChunkTexts formats chunk texts for prompt
func (d *DeepSearch) _formatChunkTexts(chunkTexts []string) string {
	chunkStr := ""
	for i, chunk := range chunkTexts {
		chunkStr += fmt.Sprintf("<chunk_%d>\n%s\n</chunk_%d>\n", i, chunk, i)
	}
	return chunkStr
}

// _searchChunksFromInternet searches chunks from internet/web search
func (d *DeepSearch) _searchChunksFromInternet(ctx context.Context, query string, subQueries []string) ([]RetrievalResult, int, error) {
	// If internet search is not enabled or no provider is configured, return empty results
	if d.internetSearch == nil || !d.internetSearch.Enabled || d.internetSearch.Provider == nil {
		return []RetrievalResult{}, 0, nil
	}

	allRetrievedResults := make([]RetrievalResult, 0)
	totalTokens := 0

	// Search for the main query
	internetResults, err := d.internetSearch.Provider.Search(ctx, query, d.internetSearch.MaxResults)
	if err != nil {
		// Log error but don't fail the entire search
		// In production, you might want to use a logger here
		_ = err
		return []RetrievalResult{}, 0, nil
	}

	// Convert internet search results to RetrievalResult
	for _, internetResult := range internetResults {
		retrievalResult := ConvertInternetSearchResultToRetrievalResult(internetResult)
		allRetrievedResults = append(allRetrievedResults, retrievalResult)
	}

	// Optionally rerank internet results using LLM (similar to vector DB results)
	// This is optional but can improve quality
	if len(allRetrievedResults) > 0 {
		rerankedResults := make([]RetrievalResult, 0)
		allQueries := make([]string, 0, len(subQueries)+1)
		allQueries = append(allQueries, query)
		allQueries = append(allQueries, subQueries...)
		queriesStr := fmt.Sprintf("%v", allQueries)

		for _, result := range allRetrievedResults {
			chunkText := fmt.Sprintf("<chunk>%s</chunk>", result.Text)
			rerankPromptText := fmt.Sprintf(rerankPrompt, queriesStr, chunkText)

			response, err := d.llm.Chat(ctx, []llm.ChatMessage{
				{Role: "user", Content: rerankPromptText},
			})
			if err != nil {
				continue // Skip this chunk if reranking fails
			}
			totalTokens += response.TotalTokens

			responseContent := strings.ToUpper(strings.TrimSpace(response.Content))
			if strings.Contains(responseContent, "YES") && !strings.Contains(responseContent, "NO") {
				rerankedResults = append(rerankedResults, result)
			}
		}
		allRetrievedResults = rerankedResults
	}

	return allRetrievedResults, totalTokens, nil
}

// Retrieve retrieves relevant documents from the knowledge base for the given query.
// This implements the RAGAgent interface.
func (d *DeepSearch) Retrieve(ctx context.Context, query string, kwargs map[string]interface{}) ([]RetrievalResult, int, map[string]interface{}, error) {
	maxIter := d.maxIter
	if kwargs != nil {
		if maxIterVal, ok := kwargs["max_iter"].(int); ok {
			maxIter = maxIterVal
		}
	}

	allSearchRes := make([]RetrievalResult, 0)
	allSubQueries := make([]string, 0)
	totalTokens := 0

	// Generate sub queries
	subQueries, usedToken, err := d._generateSubQueries(ctx, query)
	if err != nil {
		return nil, 0, nil, fmt.Errorf("failed to generate sub queries: %w", err)
	}
	totalTokens += usedToken

	if len(subQueries) == 0 {
		return []RetrievalResult{}, totalTokens, map[string]interface{}{"all_sub_queries": []string{}}, nil
	}

	allSubQueries = append(allSubQueries, subQueries...)
	subGapQueries := subQueries

	// Iterate search
	for iter := 0; iter < maxIter; iter++ {
		searchResFromVectorDB := make([]RetrievalResult, 0)
		searchResFromInternet := make([]RetrievalResult, 0)

		// Search for each gap query
		for _, gapQuery := range subGapQueries {
			// Search in vector database
			searchRes, consumedToken, err := d._searchChunksFromVectorDB(ctx, gapQuery, subGapQueries)
			if err != nil {
				continue // Continue with next query if search fails
			}
			totalTokens += consumedToken
			searchResFromVectorDB = append(searchResFromVectorDB, searchRes...)

			// Search on internet (if enabled)
			internetRes, internetTokens, err := d._searchChunksFromInternet(ctx, gapQuery, subGapQueries)
			if err == nil {
				totalTokens += internetTokens
				searchResFromInternet = append(searchResFromInternet, internetRes...)
			}
		}

		// Deduplicate results
		searchResFromVectorDB = deduplicateResults(searchResFromVectorDB)
		searchResFromInternet = deduplicateResults(searchResFromInternet)
		allSearchRes = append(allSearchRes, searchResFromVectorDB...)
		allSearchRes = append(allSearchRes, searchResFromInternet...)

		if iter == maxIter-1 {
			break // Exceeded maximum iterations
		}

		// Generate gap queries for next iteration
		subGapQueries, consumedToken, err := d._generateGapQueries(ctx, query, allSubQueries, allSearchRes)
		if err != nil {
			break // Stop if gap query generation fails
		}
		totalTokens += consumedToken

		if len(subGapQueries) == 0 {
			break // No new search queries generated
		}

		allSubQueries = append(allSubQueries, subGapQueries...)
	}

	// Final deduplication
	allSearchRes = deduplicateResults(allSearchRes)
	additionalInfo := map[string]interface{}{
		"all_sub_queries": allSubQueries,
	}

	return allSearchRes, totalTokens, additionalInfo, nil
}

// Query executes a query and returns the final answer along with retrieved results.
// This implements the RAGAgent interface.
func (d *DeepSearch) Query(ctx context.Context, query string, kwargs map[string]interface{}) (string, []RetrievalResult, int, error) {
	allRetrievedResults, nTokenRetrieval, additionalInfo, err := d.Retrieve(ctx, query, kwargs)
	if err != nil {
		return "", nil, 0, err
	}

	if len(allRetrievedResults) == 0 {
		return fmt.Sprintf("No relevant information found for query '%s'.", query), []RetrievalResult{}, nTokenRetrieval, nil
	}

	allSubQueries, ok := additionalInfo["all_sub_queries"].([]string)
	if !ok {
		allSubQueries = []string{}
	}

	// Format chunk texts
	chunkTexts := make([]string, 0, len(allRetrievedResults))
	for _, chunk := range allRetrievedResults {
		chunkTexts = append(chunkTexts, chunk.Text)
	}

	miniQuestionsStr := fmt.Sprintf("%v", allSubQueries)
	chunkStr := d._formatChunkTexts(chunkTexts)
	summaryPromptText := fmt.Sprintf(summaryPrompt, query, miniQuestionsStr, chunkStr)

	response, err := d.llm.Chat(ctx, []llm.ChatMessage{
		{Role: "user", Content: summaryPromptText},
	})
	if err != nil {
		return "", nil, 0, fmt.Errorf("failed to generate summary: %w", err)
	}

	finalAnswer := response.Content
	return finalAnswer, allRetrievedResults, nTokenRetrieval + response.TotalTokens, nil
}

// LiteralEvalString parses a string response into a list of strings
func LiteralEvalString(content string) ([]string, error) {
	content = strings.TrimSpace(content)
	content = llm.RemoveThink(content)

	// Remove code blocks if present
	if strings.HasPrefix(content, "```") {
		lines := strings.Split(content, "\n")
		if len(lines) > 1 {
			content = strings.Join(lines[1:len(lines)-1], "\n")
		}
	}

	// Try to find list pattern
	re := regexp.MustCompile(`\[.*?\]`)
	matches := re.FindString(content)
	if matches == "" {
		return nil, fmt.Errorf("no list found in content: %s", content)
	}

	// Parse the list - remove brackets and split by comma
	matches = strings.Trim(matches, "[]")
	if matches == "" {
		return []string{}, nil // Empty list
	}

	// Split by comma and clean up each string
	parts := strings.Split(matches, ",")
	result := make([]string, 0, len(parts))

	for _, part := range parts {
		part = strings.TrimSpace(part)
		// Remove quotes if present
		part = strings.Trim(part, `"'`)
		if part == "" {
			continue
		}
		result = append(result, part)
	}

	return result, nil
}

// ConvertInternetSearchResultToRetrievalResult converts InternetSearchResult to RetrievalResult
func ConvertInternetSearchResultToRetrievalResult(internetResult websearch.InternetSearchResult) RetrievalResult {
	// Create metadata with URL and title
	metadata := make(map[string]interface{})
	metadata["url"] = internetResult.Link
	metadata["title"] = internetResult.Title
	metadata["source"] = "internet"

	// Format content with title and URL reference
	formattedContent := fmt.Sprintf("Title: %s\nURL: %s\n\n%s",
		internetResult.Title,
		internetResult.Link,
		internetResult.Content,
	)

	return RetrievalResult{
		Text:  formattedContent,
		Score: internetResult.Score,
		Document: schema.Document{
			ID:       internetResult.Link, // Use URL as ID
			Content:  formattedContent,
			Metadata: metadata,
		},
		Metadata: metadata,
	}
}
