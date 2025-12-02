package llm

import (
	"context"
	"fmt"
	"regexp"
	"strings"

	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/config"
)

const (
	// OpenAI LLM provider
	PROVIDER_TYPE_OPENAI = "openai"
	// More providers can be added (e.g., Qwen)
)

// ChatMessage represents a message in a chat conversation
type ChatMessage struct {
	Role    string
	Content string
}

// ChatResponse represents a response from a chat model
type ChatResponse struct {
	Content     string
	TotalTokens int
}

// Provider defines interface for LLM providers with prompt-response pattern.
// Extensible for future chat-style and streaming features.
type Provider interface {

	// Returns provider type for registration and lookup
	GetProviderType() string

	// Chat sends a chat message to the language model and gets a response
	Chat(ctx context.Context, messages []ChatMessage) (*ChatResponse, error)
	// Generates text response for given prompt
	//
	// ctx: For cancellation and timeout
	// prompt: Input text
	// Returns: Generated response and error if any
	GenerateCompletion(ctx context.Context, prompt string) (string, error)
}

// Factory interface for creating Provider instances
type providerInitializer interface {
	// Creates Provider with given config
	CreateProvider(config.LLMConfig) (Provider, error)
}

// Maps provider types to initializers
var (
	providerInitializers = map[string]providerInitializer{
		PROVIDER_TYPE_OPENAI: &openAIProviderInitializer{},
	}
)

// Creates Provider instance based on config
//
// cfg: Provider config
// Returns: Provider instance and error if any
func NewLLMProvider(cfg config.LLMConfig) (Provider, error) {
	initializer, ok := providerInitializers[cfg.Provider]
	if !ok {
		return nil, fmt.Errorf("no initializer found for llm provider type: %s", cfg.Provider)
	}
	return initializer.CreateProvider(cfg)
}

// RemoveThink removes content between various think tags
// This function removes:
// - <think>...</think>
// - <thinking>...</thinking>
// - [think]...[/think]
// - [thinking]...[/thinking]
func RemoveThink(content string) string {
	if content == "" {
		return content
	}

	// Remove <think>...</think> tags (case-insensitive, multiline)
	re1 := regexp.MustCompile(`(?i)(?s)<think>.*?</think>`)
	content = re1.ReplaceAllString(content, "")

	// Remove <thinking>...</thinking> tags (case-insensitive, multiline)
	re2 := regexp.MustCompile(`(?i)(?s)<thinking>.*?</thinking>`)
	content = re2.ReplaceAllString(content, "")

	// Remove [think]...[/think] tags (case-insensitive, multiline)
	re3 := regexp.MustCompile(`(?i)(?s)\[think\].*?\[/think\]`)
	content = re3.ReplaceAllString(content, "")

	// Remove [thinking]...[/thinking] tags (case-insensitive, multiline)
	re4 := regexp.MustCompile(`(?i)(?s)\[thinking\].*?\[/thinking\]`)
	content = re4.ReplaceAllString(content, "")

	// Clean up excessive newlines
	re5 := regexp.MustCompile(`\n{3,}`)
	content = re5.ReplaceAllString(content, "\n\n")

	return strings.TrimSpace(content)
}
