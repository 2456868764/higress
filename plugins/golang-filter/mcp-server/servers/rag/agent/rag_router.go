package agent

import (
	"context"
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/llm"
)

const (
	// RAGRouterPrompt is the prompt template for routing queries to agents
	ragRouterPrompt = `Given a list of agent indexes and corresponding descriptions, each agent has a specific function. 
Given a query, select only one agent that best matches the agent handling the query, and return the index without any other information.

## Question
%s

## Agent Indexes and Descriptions
%s

Only return one agent index number that best matches the agent handling the query:`
)

// RAGRouter routes queries to the most appropriate RAG agent implementation.
// This class analyzes the content and requirements of a query and determines
// which RAG agent implementation is best suited to handle it.
type RAGRouter struct {
	llm               llm.Provider
	ragAgents         []RAGAgent
	agentDescriptions []string
}

// NewRAGRouter creates a new RAGRouter instance.
//
// Args:
//   - llm: The language model to use for analyzing queries
//   - ragAgents: A list of RAGAgent instances
//   - agentDescriptions: Optional list of descriptions for each agent.
//     If not provided, the router will try to get descriptions from agent metadata
func NewRAGRouter(llm llm.Provider, ragAgents []RAGAgent) (*RAGRouter, error) {
	if len(ragAgents) == 0 {
		return nil, fmt.Errorf("at least one RAG agent is required")
	}

	// Try to get descriptions from agent metadata
	// So we'll use a different approach - check if agents implement a Description() method
	descriptions := make([]string, len(ragAgents))
	for i, agent := range ragAgents {
		// Try to get description from agent if it implements DescribableAgent interface
		if describable, ok := agent.(DescribableAgent); ok {
			descriptions[i] = describable.Description()
		} else {
			// Fallback: use agent type name
			descriptions[i] = fmt.Sprintf("Agent %d: %T", i+1, agent)
		}
	}

	if len(descriptions) != len(ragAgents) {
		return nil, fmt.Errorf("number of descriptions (%d) must match number of agents (%d)", len(descriptions), len(ragAgents))
	}

	return &RAGRouter{
		llm:               llm,
		ragAgents:         ragAgents,
		agentDescriptions: descriptions,
	}, nil
}

// DescribableAgent is an optional interface that agents can implement
// to provide their own description
type DescribableAgent interface {
	Description() string
}

// route selects the most appropriate agent for the given query
// Returns the selected agent and the number of tokens used for routing
func (r *RAGRouter) route(ctx context.Context, query string) (RAGAgent, int, error) {
	// Build description string
	descriptionLines := make([]string, len(r.agentDescriptions))
	for i, desc := range r.agentDescriptions {
		descriptionLines[i] = fmt.Sprintf("[%d]: %s", i+1, desc)
	}
	descriptionStr := strings.Join(descriptionLines, "\n")

	// Build prompt
	prompt := fmt.Sprintf(ragRouterPrompt, query, descriptionStr)

	// Call LLM to select agent
	response, err := r.llm.Chat(ctx, []llm.ChatMessage{
		{Role: "user", Content: prompt},
	})
	if err != nil {
		return nil, 0, fmt.Errorf("failed to route query: %w", err)
	}

	// Parse response to get agent index
	// Note: RemoveThink is already called in llm.Provider.Chat
	content := response.Content
	selectedAgentIndex, err := r.parseAgentIndex(content)
	if err != nil {
		// Fallback: try to find the last digit
		lastDigit, findErr := r.findLastDigit(content)
		if findErr != nil {
			return nil, 0, fmt.Errorf("failed to parse agent index from response '%s': %w", content, err)
		}
		selectedAgentIndex = lastDigit - 1
	}

	// Validate index
	if selectedAgentIndex < 0 || selectedAgentIndex >= len(r.ragAgents) {
		return nil, 0, fmt.Errorf("invalid agent index %d (must be between 1 and %d)", selectedAgentIndex+1, len(r.ragAgents))
	}

	selectedAgent := r.ragAgents[selectedAgentIndex]
	return selectedAgent, response.TotalTokens, nil
}

// parseAgentIndex parses the agent index from LLM response
func (r *RAGRouter) parseAgentIndex(content string) (int, error) {
	content = strings.TrimSpace(content)

	// Try to parse as integer directly
	if idx, err := strconv.Atoi(content); err == nil {
		return idx - 1, nil // Convert from 1-based to 0-based
	}

	// Try to find a number in the content
	re := regexp.MustCompile(`\d+`)
	matches := re.FindString(content)
	if matches != "" {
		if idx, err := strconv.Atoi(matches); err == nil {
			return idx - 1, nil
		}
	}

	return 0, fmt.Errorf("no valid agent index found in response: %s", content)
}

// findLastDigit finds the last digit in a string and returns it as an integer
func (r *RAGRouter) findLastDigit(s string) (int, error) {
	for i := len(s) - 1; i >= 0; i-- {
		if s[i] >= '0' && s[i] <= '9' {
			return int(s[i] - '0'), nil
		}
	}
	return 0, fmt.Errorf("no digit found in string: %s", s)
}

// Invoke executes the agent with the given query and returns the result.
// This implements the BaseAgent interface.
func (r *RAGRouter) Invoke(ctx context.Context, query string, kwargs map[string]interface{}) (interface{}, error) {
	answer, _, _, err := r.Query(ctx, query, kwargs)
	if err != nil {
		return nil, err
	}
	return answer, nil
}

// Retrieve retrieves relevant documents from the knowledge base based on the query.
// This implements the RAGAgent interface.
func (r *RAGRouter) Retrieve(ctx context.Context, query string, kwargs map[string]interface{}) ([]RetrievalResult, int, map[string]interface{}, error) {
	agent, nTokenRouter, err := r.route(ctx, query)
	if err != nil {
		return nil, 0, nil, err
	}

	retrievedResults, nTokenRetrieval, metadata, err := agent.Retrieve(ctx, query, kwargs)
	if err != nil {
		return nil, 0, nil, err
	}

	// Add routing metadata
	if metadata == nil {
		metadata = make(map[string]interface{})
	}
	metadata["routing_tokens"] = nTokenRouter
	metadata["selected_agent"] = fmt.Sprintf("%T", agent)

	return retrievedResults, nTokenRouter + nTokenRetrieval, metadata, nil
}

// Query executes a query and returns the final answer along with retrieved results.
// This implements the RAGAgent interface.
func (r *RAGRouter) Query(ctx context.Context, query string, kwargs map[string]interface{}) (string, []RetrievalResult, int, error) {
	agent, nTokenRouter, err := r.route(ctx, query)
	if err != nil {
		return "", nil, 0, err
	}

	answer, retrievedResults, nTokenRetrieval, err := agent.Query(ctx, query, kwargs)
	if err != nil {
		return "", nil, 0, err
	}

	return answer, retrievedResults, nTokenRouter + nTokenRetrieval, nil
}

// Ensure RAGRouter implements RAGAgent interface at compile time
var _ RAGAgent = (*RAGRouter)(nil)
