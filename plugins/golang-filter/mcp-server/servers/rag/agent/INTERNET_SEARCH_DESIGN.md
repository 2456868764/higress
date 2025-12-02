# Internet Search 功能设计文档

## 概述

本文档描述了在 `DeepSearch` agent 中集成 Internet Search（互联网搜索）功能的设计方案。该功能允许 DeepSearch 在向量数据库搜索的基础上，同时从互联网搜索相关信息，从而提供更全面和最新的答案。

## 设计目标

1. **可扩展性**: 通过接口设计，支持多种互联网搜索提供商（Google、Bing、Quark等）
2. **可选性**: Internet search 是可选的，默认关闭，不影响现有功能
3. **一致性**: Internet search 结果与 vector DB 结果使用相同的格式和处理流程
4. **质量保证**: Internet search 结果也经过 LLM rerank，确保相关性

## 架构设计

### 1. 接口定义

```go
// InternetSearchProvider 定义了互联网搜索提供商的接口
type InternetSearchProvider interface {
    Search(ctx context.Context, query string, maxResults int) ([]InternetSearchResult, error)
}

// InternetSearchResult 表示互联网搜索结果
type InternetSearchResult struct {
    Title   string  // 页面标题
    Link    string  // URL链接
    Content string  // 页面内容摘要或全文
    Score   float64 // 相关性分数（如果可用）
}
```

### 2. 配置结构

```go
// InternetSearchConfig 保存互联网搜索配置
type InternetSearchConfig struct {
    Enabled    bool                  // 是否启用互联网搜索
    MaxResults int                   // 每次查询返回的最大结果数
    Provider   InternetSearchProvider // 搜索提供商实现
}
```

### 3. 集成到 DeepSearch

```go
type DeepSearch struct {
    // ... 现有字段
    internetSearch *InternetSearchConfig // 新增：互联网搜索配置
}
```

## 实现流程

### 1. 搜索流程

```
DeepSearch.Retrieve()
  ↓
迭代搜索（每次迭代）
  ↓
对每个 gap query:
  ├─→ _searchChunksFromVectorDB()  // 向量数据库搜索
  └─→ _searchChunksFromInternet()  // 互联网搜索（如果启用）
  ↓
合并和去重结果
```

### 2. Internet Search 处理流程

```
_searchChunksFromInternet(query, subQueries)
  ↓
检查是否启用
  ↓
调用 InternetSearchProvider.Search()
  ↓
转换结果格式（InternetSearchResult → RetrievalResult）
  ↓
可选：LLM Rerank（与 vector DB 结果一致）
  ↓
返回结果
```

### 3. 结果转换

Internet search 结果会被转换为 `RetrievalResult` 格式，包含：
- **Text**: 格式化的内容（包含标题、URL和内容）
- **Score**: 相关性分数
- **Metadata**: 
  - `url`: 页面URL
  - `title`: 页面标题
  - `source`: "internet"（标识来源）

## 使用示例

### 1. 创建带 Internet Search 的 DeepSearch

```go
// 创建 Internet Search Provider（需要实现 InternetSearchProvider 接口）
quarkProvider := &QuarkSearchProvider{
    APIKey: "your-api-key",
}

// 配置 Internet Search
internetConfig := &InternetSearchConfig{
    Enabled:    true,
    MaxResults: 5,
    Provider:   quarkProvider,
}

// 创建 DeepSearch
deepSearch, err := NewDeepSearch(
    llm,
    embeddingModel,
    vectorDB,
    config,
    3,              // maxIter
    false,          // textWindowSplit
    internetConfig, // internetSearch
)
```

### 2. 不使用 Internet Search（默认）

```go
// 传入 nil 或 DefaultInternetSearchConfig()（默认禁用）
deepSearch, err := NewDeepSearch(
    llm,
    embeddingModel,
    vectorDB,
    config,
    3,
    false,
    nil, // 或 DefaultInternetSearchConfig()
)
```

## Provider 实现示例

### QuarkSearchProvider 示例

```go
type QuarkSearchProvider struct {
    APIKey string
    Client *http.Client
}

func (q *QuarkSearchProvider) Search(ctx context.Context, query string, maxResults int) ([]InternetSearchResult, error) {
    // 调用 Quark Search API
    // 解析响应
    // 返回 InternetSearchResult 列表
    // ...
}
```

### GoogleProvider 示例

```go
type GoogleProvider struct {
	apiKey string
	cx     string
}

func (g *GoogleProvider) Search(ctx context.Context, query string, maxResults int) ([]InternetSearchResult, error) {
    // 调用 Google Custom Search API
    // ...
}
```

## 与 Python 版本的对应关系

| Python 版本 | Go 版本 |
|-------------|---------|
| `search_res_from_internet = []  # TODO` | `_searchChunksFromInternet()` |
| 未实现 | `InternetSearchProvider` 接口 |
| 未实现 | `InternetSearchConfig` 配置 |
| 合并到 `all_search_res` | 合并到 `allSearchRes` |

## 优势

1. **模块化设计**: 通过接口实现，易于扩展和替换不同的搜索提供商
2. **向后兼容**: 默认禁用，不影响现有代码
3. **统一处理**: Internet 结果和 Vector DB 结果使用相同的 rerank 和去重逻辑
4. **可配置**: 可以灵活配置是否启用、最大结果数等参数

## 注意事项

1. **Token 消耗: Internet search 会增加 LLM rerank 的 token 消耗
- 2网络延迟: Internet search 可能增加整体响应时间
- 3API 限制: 需要考虑搜索提供商的 API 调用限制和费用
- 4结果质量: Internet search 结果的质量取决于提供商和查询优化

## 未来扩展

1. **缓存机制**: 缓存常用的互联网搜索结果
2. **并发搜索**: 并行执行多个互联网搜索查询
3. **结果过滤**: 基于域名、日期等条件过滤结果
4. **多提供商**: 同时使用多个搜索提供商并合并结果
5. **智能选择**: 根据查询类型自动决定是否使用互联网搜索

