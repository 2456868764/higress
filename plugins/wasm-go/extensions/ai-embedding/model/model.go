package model

// TextEmbeddingRequest 表示文本嵌入请求的结构
type TextEmbeddingRequest struct {
	Model      string     `json:"model"`      // 使用的模型名称
	Input      Input      `json:"input"`      // 输入数据
	Parameters Parameters `json:"parameters"` // 请求参数
}

// Input 表示输入文本的集合
type Input struct {
	Texts []string `json:"texts"` // 待嵌入的文本列表
}

// Parameters 表示请求参数
type Parameters struct {
	TextType string `json:"text_type"` // 文本类型
}

// Embedding 表示单个文本的嵌入向量和索引
type Embedding struct {
	Vector []float64 `json:"embedding"`  // 嵌入向量
	Index  int       `json:"text_index"` // 输入文本索引
}

// Output 表示嵌入输出的集合
type Output struct {
	Embeddings []Embedding `json:"embeddings"` // 嵌入列表
}

// Usage 表示请求的使用情况
type Usage struct {
	TotalTokens int `json:"total_tokens"` // 请求使用的总标记数
}

// TextEmbeddingResponse 表示文本嵌入响应的结构
type TextEmbeddingResponse struct {
	StatusCode int    `json:"status_code"`       // 状态码
	RequestID  string `json:"request_id"`        // 请求 ID
	Code       string `json:"code,omitempty"`    // 错误代码，失败时使用
	Message    string `json:"message,omitempty"` // 错误信息，失败时使用
	Output     Output `json:"output"`            // 输出的嵌入信息
	Usage      Usage  `json:"usage"`             // 使用情况
}

// VectorQueryRequest 表示向量查询请求的结构
type VectorQueryRequest struct {
	Vector        []float64 `json:"vector"`         // 要查询的向量
	TopK          int       `json:"topk"`           // 返回的最近邻个数
	IncludeVector bool      `json:"include_vector"` // 是否在结果中包含向量
}

// VectorQueryResponse 表示向量查询响应的结构
type VectorQueryResponse struct {
	Code      int                 `json:"code"`       // 响应代码
	RequestID string              `json:"request_id"` // 请求 ID
	Message   string              `json:"message"`    // 响应消息
	Output    []VectorQueryOutput `json:"output"`     // 查询输出结果
}

// VectorQueryOutput 表示单个向量查询结果
type VectorQueryOutput struct {
	ID     string                 `json:"id"`               // 结果的唯一标识符
	Vector []float64              `json:"vector,omitempty"` // 匹配的向量
	Fields map[string]interface{} `json:"fields"`           // 其他相关字段
	Score  float64                `json:"score"`            // 匹配分数
}

// VectorInsertRequest 表示向量插入请求的结构
type VectorInsertRequest struct {
	Docs []Document `json:"docs"` // 要插入的文档列表
}

// Document 表示要插入的文档，包含向量和字段
type Document struct {
	Vector []float64         `json:"vector"` // 文档的向量表示
	Fields map[string]string `json:"fields"` // 文档的其他字段
}
