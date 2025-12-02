package websearch

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"time"
)

type GoogleProvider struct {
	apiKey string
	cx     string
	Client *http.Client
}

func NewGoogleProvider(apiKey, cx string) *GoogleProvider {
	return &GoogleProvider{
		apiKey: apiKey,
		cx:     cx,
		Client: &http.Client{
			Timeout: 10 * time.Second,
		},
	}
}

type googleSearchResponse struct {
	Items []googleSearchItem `json:"items"`
}

type googleSearchItem struct {
	Title   string `json:"title"`
	Link    string `json:"link"`
	Snippet string `json:"snippet"`
}

// Search implements InternetSearchProvider using Google Custom Search JSON API
func (g *GoogleProvider) Search(ctx context.Context, query string, maxResults int) ([]InternetSearchResult, error) {
	baseURL := "https://www.googleapis.com/customsearch/v1"

	u, err := url.Parse(baseURL)
	if err != nil {
		return nil, fmt.Errorf("failed to parse base URL: %w", err)
	}

	q := u.Query()
	q.Set("key", g.apiKey)
	q.Set("cx", g.cx)
	q.Set("q", query)
	if maxResults > 10 {
		maxResults = 10 // API limit per request
	}
	q.Set("num", fmt.Sprintf("%d", maxResults))

	u.RawQuery = q.Encode()

	req, err := http.NewRequestWithContext(ctx, "GET", u.String(), nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	client := g.Client
	if client == nil {
		client = http.DefaultClient
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to perform search request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	var searchResp googleSearchResponse
	if err := json.NewDecoder(resp.Body).Decode(&searchResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	var results []InternetSearchResult
	for i, item := range searchResp.Items {
		results = append(results, InternetSearchResult{
			Title:   item.Title,
			Link:    item.Link,
			Content: item.Snippet,
			Score:   1.0 - (float64(i) * 0.1), // Simple decay score
		})
	}

	return results, nil
}
