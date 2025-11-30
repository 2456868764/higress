package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/llm"
	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/packages/param"
)

// ChatLLMAdapter wraps an existing LLM provider to add chat capabilities
type ChatLLMAdapter struct {
	llm.Provider
	client      *openai.Client
	model       string
	temperature float64
	maxTokens   int
}

// NewChatLLMAdapter creates a new ChatLLM adapter from an OpenAI provider
func NewChatLLMAdapter(baseProvider llm.Provider, client *openai.Client, model string, temperature float64, maxTokens int) ChatLLM {
	return &ChatLLMAdapter{
		Provider:    baseProvider,
		client:      client,
		model:       model,
		temperature: temperature,
		maxTokens:   maxTokens,
	}
}

// Chat sends a chat message to the language model and gets a response
func (c *ChatLLMAdapter) Chat(ctx context.Context, messages []ChatMessage) (*ChatResponse, error) {
	openaiMessages := make([]openai.ChatCompletionMessageParamUnion, 0, len(messages))
	for _, msg := range messages {
		switch msg.Role {
		case "user":
			openaiMessages = append(openaiMessages, openai.UserMessage(msg.Content))
		case "system":
			openaiMessages = append(openaiMessages, openai.SystemMessage(msg.Content))
		case "assistant":
			openaiMessages = append(openaiMessages, openai.AssistantMessage(msg.Content))
		default:
			openaiMessages = append(openaiMessages, openai.UserMessage(msg.Content))
		}
	}

	params := openai.ChatCompletionNewParams{
		Model:    c.model,
		Messages: openaiMessages,
	}

	if c.temperature > 0 {
		params.Temperature = param.Opt[float64]{Value: c.temperature}
	}

	if c.maxTokens > 0 {
		params.MaxTokens = param.Opt[int64]{Value: int64(c.maxTokens)}
	}

	response, err := c.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("chat completion error: %w", err)
	}

	if len(response.Choices) == 0 {
		return nil, fmt.Errorf("empty choices in response")
	}

	totalTokens := 0
	if response.Usage.TotalTokens > 0 {
		totalTokens = int(response.Usage.TotalTokens)
	}

	return &ChatResponse{
		Content:     response.Choices[0].Message.Content,
		TotalTokens: totalTokens,
	}, nil
}

// RemoveThink removes content between <think> tags
func (c *ChatLLMAdapter) RemoveThink(content string) string {
	return RemoveThink(content)
}

// LiteralEval parses a string response into a list of integers
func (c *ChatLLMAdapter) LiteralEval(content string) ([]int, error) {
	return LiteralEval(content)
}

// OpenAIProviderToChatLLM converts an OpenAI provider to ChatLLM
// This is a helper function that extracts the client from the provider
func OpenAIProviderToChatLLM(ctx context.Context, provider llm.Provider, model string, temperature float64, maxTokens int) (ChatLLM, error) {
	// Try to get the client from the provider
	// Since we can't directly access the client, we'll need to create a new one
	// For now, we'll use a workaround by calling GenerateCompletion to test
	_, err := provider.GenerateCompletion(ctx, "test")
	if err != nil {
		return nil, fmt.Errorf("provider test failed: %w", err)
	}

	// We need to create a new OpenAI client
	// This is a limitation - ideally the provider should expose its client
	// For now, we'll create a wrapper that uses the provider's GenerateCompletion
	return &ChatLLMWrapper{
		provider:    provider,
		model:       model,
		temperature: temperature,
		maxTokens:   maxTokens,
	}, nil
}

// ChatLLMWrapper wraps a standard LLM provider to provide chat capabilities
// by converting chat messages to a single prompt
type ChatLLMWrapper struct {
	provider    llm.Provider
	model       string
	temperature float64
	maxTokens   int
}

// Chat converts chat messages to a prompt and calls GenerateCompletion
func (c *ChatLLMWrapper) Chat(ctx context.Context, messages []ChatMessage) (*ChatResponse, error) {
	// Convert messages to a single prompt
	prompt := ""
	for _, msg := range messages {
		switch msg.Role {
		case "system":
			prompt += fmt.Sprintf("System: %s\n\n", msg.Content)
		case "user":
			prompt += fmt.Sprintf("User: %s\n\n", msg.Content)
		case "assistant":
			prompt += fmt.Sprintf("Assistant: %s\n\n", msg.Content)
		}
	}
	prompt += "Assistant:"

	content, err := c.provider.GenerateCompletion(ctx, prompt)
	if err != nil {
		return nil, err
	}

	// Estimate token usage (rough approximation: 1 token ≈ 4 characters)
	totalTokens := len(prompt+content) / 4

	return &ChatResponse{
		Content:     content,
		TotalTokens: totalTokens,
	}, nil
}

// GetProviderType returns the provider type
func (c *ChatLLMWrapper) GetProviderType() string {
	return c.provider.GetProviderType()
}

// GenerateCompletion delegates to the underlying provider
func (c *ChatLLMWrapper) GenerateCompletion(ctx context.Context, prompt string) (string, error) {
	return c.provider.GenerateCompletion(ctx, prompt)
}

// RemoveThink removes content between <think> tags
func (c *ChatLLMWrapper) RemoveThink(content string) string {
	return RemoveThink(content)
}

// LiteralEval parses a string response into a list of integers
func (c *ChatLLMWrapper) LiteralEval(content string) ([]int, error) {
	return LiteralEval(content)
}

// Improved LiteralEval that handles JSON arrays
func literalEvalJSON(content string) ([]int, error) {
	content = strings.TrimSpace(content)
	content = RemoveThink(content)

	// Try to parse as JSON first
	var result []int
	if err := json.Unmarshal([]byte(content), &result); err == nil {
		return result, nil
	}

	// Try to find JSON array in the content
	re := regexp.MustCompile(`\[[\d\s,]+\]`)
	matches := re.FindString(content)
	if matches != "" {
		if err := json.Unmarshal([]byte(matches), &result); err == nil {
			return result, nil
		}
	}

	// Fallback to string parsing
	return LiteralEval(content)
}
