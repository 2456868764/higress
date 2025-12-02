package websearch

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"
)

type DuckDuckGoProvider struct {
	Client *http.Client
}

func NewDuckDuckGoProvider() *DuckDuckGoProvider {
	return &DuckDuckGoProvider{
		Client: &http.Client{
			Timeout: 10 * time.Second,
		},
	}
}

// Search implements InternetSearchProvider using DuckDuckGo HTML version
func (d *DuckDuckGoProvider) Search(ctx context.Context, query string, maxResults int) ([]InternetSearchResult, error) {
	searchURL := "https://html.duckduckgo.com/html/"

	// Build form data
	data := url.Values{}
	data.Set("q", query)

	req, err := http.NewRequestWithContext(ctx, "POST", searchURL, strings.NewReader(data.Encode()))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36")

	client := d.Client
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

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}
	htmlContent := string(bodyBytes)

	return parseDuckDuckGoHTML(htmlContent, maxResults), nil
}

func parseDuckDuckGoHTML(html string, limit int) []InternetSearchResult {
	// Regex to find result blocks
	// This is a fragile way to parse HTML but avoids external dependencies like goquery
	// Structure is roughly: <div class="result ..."> ... <a class="result__a" href="(url)">(title)</a> ... <a class="result__snippet" ...>(snippet)</a>

	var results []InternetSearchResult

	// Find result links (titles and URLs)
	// <a class="result__a" href="...">...</a>
	linkRegex := regexp.MustCompile(`<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>`)

	// Find snippets
	// <a class="result__snippet" href="...">...</a>
	snippetRegex := regexp.MustCompile(`<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>`)

	// Split by "result" div to process each result individually might be better,
	// but for simplicity let's try to match patterns sequentially or split by result wrapper

	// Splitting by <div class="result results_links ..."> is safer
	resultDivRegex := regexp.MustCompile(`<div[^>]*class="[^"]*result results_links[^"]*"[^>]*>`)
	fragments := resultDivRegex.Split(html, -1)

	for i, fragment := range fragments {
		if i == 0 {
			continue
		} // Skip content before first result
		if len(results) >= limit {
			break
		}

		// Extract Link and Title
		linkMatch := linkRegex.FindStringSubmatch(fragment)
		if len(linkMatch) < 3 {
			continue
		}
		urlStr := linkMatch[1]
		title := cleanHTML(linkMatch[2])

		// Extract Snippet
		snippetMatch := snippetRegex.FindStringSubmatch(fragment)
		content := ""
		if len(snippetMatch) >= 2 {
			content = cleanHTML(snippetMatch[1])
		}

		results = append(results, InternetSearchResult{
			Title:   title,
			Link:    urlStr,
			Content: content,
			Score:   1.0 - (float64(len(results)) * 0.1), // Simple decay score
		})
	}

	return results
}

func cleanHTML(input string) string {
	// Remove HTML tags
	tagRegex := regexp.MustCompile(`<[^>]*>`)
	output := tagRegex.ReplaceAllString(input, "")

	// Unescape common HTML entities
	output = strings.ReplaceAll(output, "&amp;", "&")
	output = strings.ReplaceAll(output, "&lt;", "<")
	output = strings.ReplaceAll(output, "&gt;", ">")
	output = strings.ReplaceAll(output, "&quot;", "\"")
	output = strings.ReplaceAll(output, "&#39;", "'")
	output = strings.ReplaceAll(output, "&nbsp;", " ")

	return strings.TrimSpace(output)
}
