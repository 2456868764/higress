// File generated by hgctl. Modify as required.
// See: https://higress.io/zh-cn/docs/user/wasm-go#2-%E7%BC%96%E5%86%99-maingo-%E6%96%87%E4%BB%B6

package main

import (
	"ai-embedding/model"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/alibaba/higress/plugins/wasm-go/pkg/wrapper"
	"github.com/higress-group/proxy-wasm-go-sdk/proxywasm"
	"github.com/higress-group/proxy-wasm-go-sdk/proxywasm/types"
	"github.com/tidwall/gjson"
	"github.com/tidwall/resp"
)

const (
	CacheKeyContextKey       = "cacheKey"
	CacheContentContextKey   = "cacheContent"
	PartialMessageContextKey = "partialMessage"
	ToolCallsContextKey      = "toolCalls"
	StreamContextKey         = "stream"
	DefaultCacheKeyPrefix    = "higress-ai-embedding:"
	QueryEmbeddingKey        = "queryEmbedding"
)

func main() {
	wrapper.SetCtx(
		"ai-embedding",
		wrapper.ParseConfigBy(parseConfig),
		wrapper.ProcessRequestHeadersBy(onHttpRequestHeaders),
		wrapper.ProcessRequestBodyBy(onHttpRequestBody),
		wrapper.ProcessResponseHeadersBy(onHttpResponseHeaders),
		wrapper.ProcessStreamingResponseBodyBy(onHttpResponseBody),
	)
}

type RedisInfo struct {
	// @Title zh-CN redis 服务名称
	// @Description zh-CN 带服务类型的完整 FQDN 名称，例如 my-redis.dns、redis.my-ns.svc.cluster.local
	ServiceName string `required:"true" yaml:"serviceName" json:"serviceName"`
	// @Title zh-CN redis 服务端口
	// @Description zh-CN 默认值为6379
	ServicePort int `required:"false" yaml:"servicePort" json:"servicePort"`
	// @Title zh-CN 用户名
	// @Description zh-CN 登陆 redis 的用户名，非必填
	Username string `required:"false" yaml:"username" json:"username"`
	// @Title zh-CN 密码
	// @Description zh-CN 登陆 redis 的密码，非必填，可以只填密码
	Password string `required:"false" yaml:"password" json:"password"`
	// @Title zh-CN 请求超时
	// @Description zh-CN 请求 redis 的超时时间，单位为毫秒。默认值是1000，即1秒
	Timeout int `required:"false" yaml:"timeout" json:"timeout"`
}

type KVExtractor struct {
	// @Title zh-CN 从请求 Body 中基于 [GJSON PATH](https://github.com/tidwall/gjson/blob/master/SYNTAX.md) 语法提取字符串
	RequestBody string `required:"false" yaml:"requestBody" json:"requestBody"`
	// @Title zh-CN 从响应 Body 中基于 [GJSON PATH](https://github.com/tidwall/gjson/blob/master/SYNTAX.md) 语法提取字符串
	ResponseBody string `required:"false" yaml:"responseBody" json:"responseBody"`
}

type DashScopeInfo struct {
	ServiceName   string             `require:"true" yaml:"serviceName" json:"serviceName"`
	APIKey        string             `require:"true" yaml:"apiKey" json:"apiKey"`
	Host          string             `require:"true" yaml:"host" json:"host"`
	ContextKeyLen int                `required:"true" yaml:"contextKeyLen" json:"contextKeyLen"`
	Client        wrapper.HttpClient `yaml:"-" json:"-"`
}

type DashVectorInfo struct {
	ServiceName string `require:"true" yaml:"serviceName" jaon:"serviceName"`
	// sk-AjiyUN4bLEr1DvkpWvi9nD0UFodgO298C33BC546C11EF83429E3A4EE25FE6
	APIKey                string             `require:"true" yaml:"apiKey" json:"apiKey"`
	Host                  string             `require:"true" yaml:"host" json:"host"`
	Collection            string             `require:"true" yaml:"collection" json:"collection"`
	SimilarScoreThreshold float64            `required:"true" yaml:"similarScoreThreshold" json:"similarScoreThreshold"`
	Client                wrapper.HttpClient `yaml:"-" json:"-"`
}

type PluginConfig struct {
	// @Title zh-CN Redis 地址信息
	// @Description zh-CN 用于存储缓存结果的 Redis 地址
	RedisInfo      RedisInfo      `required:"true" yaml:"redis" json:"redis"`
	DashScopeInfo  DashScopeInfo  `required:"true" yaml:"dashScope" json:"dashScope"`
	DashVectorInfo DashVectorInfo `required:"true" yaml:"dashVector" json:"dashVector"`
	// @Title zh-CN 缓存 key 的来源
	// @Description zh-CN 往 redis 里存时，使用的 key 的提取方式
	CacheKeyFrom KVExtractor `required:"true" yaml:"cacheKeyFrom" json:"cacheKeyFrom"`
	// @Title zh-CN 缓存 value 的来源
	// @Description zh-CN 往 redis 里存时，使用的 value 的提取方式
	CacheValueFrom KVExtractor `required:"true" yaml:"cacheValueFrom" json:"cacheValueFrom"`
	// @Title zh-CN 流式响应下，缓存 value 的来源
	// @Description zh-CN 往 redis 里存时，使用的 value 的提取方式
	CacheStreamValueFrom KVExtractor `required:"true" yaml:"cacheStreamValueFrom" json:"cacheStreamValueFrom"`
	// @Title zh-CN 返回 HTTP 响应的模版
	// @Description zh-CN 用 %s 标记需要被 cache value 替换的部分
	ReturnResponseTemplate string `required:"true" yaml:"returnResponseTemplate" json:"returnResponseTemplate"`
	// @Title zh-CN 返回流式 HTTP 响应的模版
	// @Description zh-CN 用 %s 标记需要被 cache value 替换的部分
	ReturnStreamResponseTemplate string `required:"true" yaml:"returnStreamResponseTemplate" json:"returnStreamResponseTemplate"`
	// @Title zh-CN 缓存的过期时间
	// @Description zh-CN 单位是秒，默认值为0，即永不过期
	CacheTTL int `required:"false" yaml:"cacheTTL" json:"cacheTTL"`
	// @Title zh-CN Redis缓存Key的前缀
	// @Description zh-CN 默认值是"higress-ai-cache:"
	CacheKeyPrefix string              `required:"false" yaml:"cacheKeyPrefix" json:"cacheKeyPrefix"`
	redisClient    wrapper.RedisClient `yaml:"-" json:"-"`
}

func parseConfig(json gjson.Result, c *PluginConfig, log wrapper.Log) error {
	log.Infof("parseConfig()")
	c.RedisInfo.ServiceName = json.Get("redis.serviceName").String()
	if c.RedisInfo.ServiceName == "" {
		return errors.New("redis service name must not by empty")
	}
	c.RedisInfo.ServicePort = int(json.Get("redis.servicePort").Int())
	if c.RedisInfo.ServicePort == 0 {
		if strings.HasSuffix(c.RedisInfo.ServiceName, ".static") {
			// use default logic port which is 80 for static service
			c.RedisInfo.ServicePort = 80
		} else {
			c.RedisInfo.ServicePort = 6379
		}
	}
	c.RedisInfo.Username = json.Get("redis.username").String()
	c.RedisInfo.Password = json.Get("redis.password").String()
	c.RedisInfo.Timeout = int(json.Get("redis.timeout").Int())
	if c.RedisInfo.Timeout == 0 {
		c.RedisInfo.Timeout = 1000
	}
	c.CacheKeyFrom.RequestBody = json.Get("cacheKeyFrom.requestBody").String()
	if c.CacheKeyFrom.RequestBody == "" {
		c.CacheKeyFrom.RequestBody = "messages.@reverse.0.content"
	}
	c.CacheValueFrom.ResponseBody = json.Get("cacheValueFrom.responseBody").String()
	if c.CacheValueFrom.ResponseBody == "" {
		c.CacheValueFrom.ResponseBody = "choices.0.message.content"
	}
	c.CacheStreamValueFrom.ResponseBody = json.Get("cacheStreamValueFrom.responseBody").String()
	if c.CacheStreamValueFrom.ResponseBody == "" {
		c.CacheStreamValueFrom.ResponseBody = "choices.0.delta.content"
	}
	c.ReturnResponseTemplate = json.Get("returnResponseTemplate").String()
	if c.ReturnResponseTemplate == "" {
		c.ReturnResponseTemplate = `{"id":"from-cache","choices":[{"index":0,"message":{"role":"assistant","content":"%s"},"finish_reason":"stop"}],"model":"gpt-4o","object":"chat.completion","usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}`
	}
	c.ReturnStreamResponseTemplate = json.Get("returnStreamResponseTemplate").String()
	if c.ReturnStreamResponseTemplate == "" {
		c.ReturnStreamResponseTemplate = `data:{"id":"from-cache","choices":[{"index":0,"delta":{"role":"assistant","content":"%s"},"finish_reason":"stop"}],"model":"gpt-4o","object":"chat.completion","usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}` + "\n\ndata:[DONE]\n\n"
	}
	c.CacheKeyPrefix = json.Get("cacheKeyPrefix").String()
	if c.CacheKeyPrefix == "" {
		c.CacheKeyPrefix = DefaultCacheKeyPrefix
	}
	c.redisClient = wrapper.NewRedisClusterClient(wrapper.FQDNCluster{
		FQDN: c.RedisInfo.ServiceName,
		Port: int64(c.RedisInfo.ServicePort),
	})

	c.DashScopeInfo.ServiceName = json.Get("dashScope.serviceName").String()
	c.DashScopeInfo.APIKey = json.Get("dashScope.apiKey").String()
	c.DashScopeInfo.Host = json.Get("dashScope.host").String()
	c.DashScopeInfo.ContextKeyLen = int(json.Get("dashScope.contextKeyLen").Int())
	c.DashScopeInfo.Client = wrapper.NewClusterClient(wrapper.FQDNCluster{
		FQDN: c.DashVectorInfo.ServiceName,
		Port: 443,
		Host: c.DashScopeInfo.Host,
	})

	c.DashVectorInfo.ServiceName = json.Get("dashVector.serviceName").String()
	c.DashVectorInfo.Host = json.Get("dashVector.host").String()
	c.DashVectorInfo.Collection = json.Get("dashVector.collection").String()
	c.DashVectorInfo.APIKey = json.Get("dashVector.apiKey").String()
	c.DashVectorInfo.SimilarScoreThreshold = json.Get("dashVector.similarScoreThreshold").Float()
	c.DashVectorInfo.Client = wrapper.NewClusterClient(wrapper.FQDNCluster{
		FQDN: c.DashVectorInfo.ServiceName,
		Port: 443,
		Host: c.DashVectorInfo.Host,
	})

	log.Infof("parseConfig() result:%+v", c)
	return c.redisClient.Init(c.RedisInfo.Username, c.RedisInfo.Password, int64(c.RedisInfo.Timeout))
}

func onHttpRequestHeaders(ctx wrapper.HttpContext, config PluginConfig, log wrapper.Log) types.Action {
	log.Infof("onHttpRequestHeaders()")
	contentType, _ := proxywasm.GetHttpRequestHeader("content-type")
	// The request does not have a body.
	if contentType == "" {
		return types.ActionContinue
	}
	if !strings.Contains(contentType, "application/json") {
		log.Warnf("content is not json, can't process:%s", contentType)
		ctx.DontReadRequestBody()
		return types.ActionContinue
	}
	proxywasm.RemoveHttpRequestHeader("Accept-Encoding")
	// The request has a body and requires delaying the header transmission until a cache miss occurs,
	// at which point the header should be sent.
	return types.ActionContinue
}

func TrimQuote(source string) string {
	return strings.Trim(source, `"`)
}

func onHttpRequestBody(ctx wrapper.HttpContext, config PluginConfig, body []byte, log wrapper.Log) types.Action {
	log.Infof("onHttpRequestBody()")
	log.Infof("request body = %s", string(body))
	bodyJson := gjson.ParseBytes(body)
	// TODO: It may be necessary to support stream mode determination for different LLM providers.
	stream := false
	if bodyJson.Get("stream").Bool() {
		stream = true
		ctx.SetContext(StreamContextKey, struct{}{})
	} else if ctx.GetContext(StreamContextKey) != nil {
		stream = true
	}
	// 根据上下文获取缓存 Key
	key, err := getKeyFromContext(ctx, config, bodyJson, log)
	if err != nil {
		log.Infof("parse key from request body failed, err:%v", err)
		return types.ActionContinue
	}
	ctx.SetContext(CacheKeyContextKey, key)
	log.Infof("cache key: %s", key)
	redisKey := encodeKey(key)
	err2 := config.redisClient.Get(config.CacheKeyPrefix+redisKey, func(response resp.Value) {
		if err := response.Error(); err != nil {
			log.Errorf("redis get key:%s failed, err:%v", key, err)
			proxywasm.ResumeHttpRequest()
			return
		}
		if response.IsNull() {
			// 缓存 miss，需要向 LLM 发起请求
			handleRedisCacheMiss(ctx, key, config, key, stream, log)
		} else {
			// 缓存命中，直接返回缓存内容
			handleRedisCacheHit(ctx, key, config, response.String(), stream, log)
		}
	})
	if err2 != nil {
		log.Errorf("redis access failed, err:%v", err2)
		return types.ActionContinue
	}
	return types.ActionPause
}

func getKeyFromContext(ctx wrapper.HttpContext, config PluginConfig, json gjson.Result, log wrapper.Log) (string, error) {
	key := json.Get(config.CacheKeyFrom.RequestBody).String()
	return key, nil
}

func encodeKey(key string) string {
	return key
}

func handleRedisCacheHit(ctx wrapper.HttpContext, key string, config PluginConfig, cacheResponse string, stream bool, log wrapper.Log) {
	log.Infof("handleRedisCacheHit()")
	log.Infof("cache hit, key:%s", key)
	ctx.SetContext(CacheKeyContextKey, nil)
	if !stream {
		proxywasm.SendHttpResponseWithDetail(200, "ai-cache.hit", [][2]string{{"content-type", "application/json; charset=utf-8"}}, []byte(fmt.Sprintf(config.ReturnResponseTemplate, cacheResponse)), -1)
	} else {
		proxywasm.SendHttpResponseWithDetail(200, "ai-cache.hit", [][2]string{{"content-type", "text/event-stream; charset=utf-8"}}, []byte(fmt.Sprintf(config.ReturnStreamResponseTemplate, cacheResponse)), -1)
	}
}

func handleRedisCacheMiss(ctx wrapper.HttpContext, key string, config PluginConfig, query string, stream bool, log wrapper.Log) {
	log.Infof("handleRedisCacheMiss()")
	log.Infof("cache miss, key:%s", key)
	url := "/api/v1/services/embeddings/text-embedding/text-embedding"
	textEmbeddingRequest := model.TextEmbeddingRequest{
		Model: "text-embedding-v1",
		Input: model.Input{
			Texts: []string{query},
		},
		Parameters: model.Parameters{
			TextType: "query",
		},
	}
	requestBody, _ := json.Marshal(textEmbeddingRequest)
	headers := [][2]string{
		{"Content-Type", "application/json"},
		{"Authorization", "Bearer " + config.DashScopeInfo.APIKey},
	}
	err := config.DashScopeInfo.Client.Post(
		url,
		headers,
		requestBody,
		func(statusCode int, responseHeaders http.Header, responseBody []byte) {
			log.Infof("fetched embeddings for key: %s, and response: %s", key, string(responseBody))
			if statusCode != 200 {
				log.Errorf("failed to fetch embeddings, statusCode: %d, response: %s", statusCode, string(responseBody))
				ctx.SetContext(QueryEmbeddingKey, nil)
				proxywasm.ResumeHttpRequest()
				return
			}
			var resp model.TextEmbeddingResponse
			if err := json.Unmarshal(responseBody, &resp); err != nil {
				log.Errorf("failed to unmarshal, error:%v", err)
				ctx.SetContext(QueryEmbeddingKey, nil)
				proxywasm.ResumeHttpRequest()
				return
			}
			embedding := resp.Output.Embeddings[0].Vector
			ctx.SetContext(QueryEmbeddingKey, embedding)
			ctx.SetContext(CacheKeyContextKey, key)
			handleTextEmbeddings(ctx, config, key, embedding, stream, log)
		},
		5000)

	if err != nil {
		log.Errorf("DashScopeInfo access failed, err:%v", err)
		proxywasm.ResumeHttpRequest()
	}

}

func handleTextEmbeddings(ctx wrapper.HttpContext, config PluginConfig, key string, embedding []float64, stream bool, log wrapper.Log) {
	log.Infof("handleTextEmbeddings()")
	url := fmt.Sprintf("/v1/collections/%s/query", config.DashVectorInfo.Collection)
	requestData := model.VectorQueryRequest{
		Vector:        embedding,
		TopK:          1,
		IncludeVector: false,
	}
	requestBody, _ := json.Marshal(requestData)
	headers := [][2]string{
		{"Content-Type", "application/json"},
		{"dashvector-auth-token", config.DashVectorInfo.APIKey},
	}
	config.DashVectorInfo.Client.Post(
		url,
		headers,
		requestBody,
		func(statusCode int, responseHeaders http.Header, responseBody []byte) {
			log.Infof("fetch vector statusCode:%d, responseBody:%s", statusCode, string(responseBody))
			var resp model.VectorQueryResponse
			err := json.Unmarshal(responseBody, &resp)
			if err != nil {
				log.Errorf("failed to parse vector response: %v", err)
				proxywasm.ResumeHttpRequest()
				return
			}
			if len(resp.Output) < 1 {
				log.Infof("vector query response is empty")
				updateEmbedding(ctx, config, key, embedding, log)
				return
			}
			topSimilarKey := resp.Output[0].Fields["key"].(string)
			log.Infof("top similar key: %s", topSimilarKey)
			topSimilarScore := resp.Output[0].Score
			if topSimilarScore >= config.DashVectorInfo.SimilarScoreThreshold {
				ctx.SetContext(CacheKeyContextKey, nil)
				getRedisCacheByKey(ctx, config, topSimilarKey, false, log)
			} else {
				log.Infof("the top similar key score is too lower, key:%s, score:%f", topSimilarKey, topSimilarScore)
				updateEmbedding(ctx, config, key, embedding, log)
				proxywasm.ResumeHttpRequest()
				return
			}
		},
		100000)
}

func getRedisCacheByKey(ctx wrapper.HttpContext, config PluginConfig, key string, stream bool, log wrapper.Log) {
	log.Infof("getRedisCacheByKey()")
	err2 := config.redisClient.Get(config.CacheKeyPrefix+key, func(response resp.Value) {
		if err := response.Error(); err != nil {
			log.Errorf("redis get key:%s failed, err:%v", key, err)
			proxywasm.ResumeHttpRequest()
			return
		}
		if response.IsNull() {
			proxywasm.ResumeHttpRequest()
			return
		}
		handleRedisCacheHit(ctx, key, config, response.String(), stream, log)
	})
	if err2 != nil {
		log.Errorf("redis access failed, err:%v", err2)
		proxywasm.ResumeHttpRequest()
	}
}

func updateEmbedding(ctx wrapper.HttpContext, config PluginConfig, key string, embedding []float64, log wrapper.Log) {
	log.Infof("updateEmbedding()")
	url := fmt.Sprintf("/v1/collections/%s/docs", config.DashVectorInfo.Collection)
	headers := [][2]string{
		{"Content-Type", "application/json"},
		{"dashvector-auth-token", config.DashVectorInfo.APIKey},
	}
	redisKey := encodeKey(key)
	doc := model.Document{
		Vector: embedding,
		Fields: map[string]string{
			"key": redisKey,
		},
	}
	requestBody, _ := json.Marshal(model.VectorInsertRequest{Docs: []model.Document{doc}})
	err := config.DashVectorInfo.Client.Post(
		url,
		headers,
		requestBody,
		func(statusCode int, responseHeaders http.Header, responseBody []byte) {
			if statusCode != 200 {
				log.Errorf("fail to udpate key:%s embedding response: %s", key, responseBody)
			} else {
				log.Infof("successfully update query embedding for key: %s", key)
			}
			proxywasm.ResumeHttpRequest()
		},
		5000,
	)

	if err != nil {
		log.Errorf("Failed to upload query embedding: %v", err)
		proxywasm.ResumeHttpRequest()
	}
}

func processSSEMessage(ctx wrapper.HttpContext, config PluginConfig, sseMessage string, log wrapper.Log) string {
	log.Infof("processSSEMessage()")
	subMessages := strings.Split(sseMessage, "\n")
	var message string
	for _, msg := range subMessages {
		if strings.HasPrefix(msg, "data:") {
			message = msg
			break
		}
	}
	if len(message) < 6 {
		log.Errorf("invalid message:%s", message)
		return ""
	}
	// skip the prefix "data:"
	bodyJson := message[5:]
	if gjson.Get(bodyJson, config.CacheStreamValueFrom.ResponseBody).Exists() {
		tempContentI := ctx.GetContext(CacheContentContextKey)
		if tempContentI == nil {
			content := TrimQuote(gjson.Get(bodyJson, config.CacheStreamValueFrom.ResponseBody).Raw)
			ctx.SetContext(CacheContentContextKey, content)
			return content
		}
		append := TrimQuote(gjson.Get(bodyJson, config.CacheStreamValueFrom.ResponseBody).Raw)
		content := tempContentI.(string) + append
		ctx.SetContext(CacheContentContextKey, content)
		return content
	} else if gjson.Get(bodyJson, "choices.0.delta.content.tool_calls").Exists() {
		// TODO: compatible with other providers
		ctx.SetContext(ToolCallsContextKey, struct{}{})
		return ""
	}
	log.Debugf("unknown message:%s", bodyJson)
	return ""
}

func onHttpResponseHeaders(ctx wrapper.HttpContext, config PluginConfig, log wrapper.Log) types.Action {
	log.Infof("onHttpResponseHeaders()")
	contentType, _ := proxywasm.GetHttpResponseHeader("content-type")
	if strings.Contains(contentType, "text/event-stream") {
		ctx.SetContext(StreamContextKey, struct{}{})
	}
	return types.ActionContinue
}

func onHttpResponseBody(ctx wrapper.HttpContext, config PluginConfig, chunk []byte, isLastChunk bool, log wrapper.Log) []byte {
	log.Infof("onHttpResponseBody()")
	log.Infof("response chunk = %s", string(chunk))
	if ctx.GetContext(ToolCallsContextKey) != nil {
		// we should not cache tool call result
		return chunk
	}
	keyI := ctx.GetContext(CacheKeyContextKey)
	if keyI == nil {
		return chunk
	}
	if !isLastChunk {
		stream := ctx.GetContext(StreamContextKey)
		if stream == nil {
			tempContentI := ctx.GetContext(CacheContentContextKey)
			if tempContentI == nil {
				ctx.SetContext(CacheContentContextKey, chunk)
				return chunk
			}
			tempContent := tempContentI.([]byte)
			tempContent = append(tempContent, chunk...)
			ctx.SetContext(CacheContentContextKey, tempContent)
		} else {
			var partialMessage []byte
			partialMessageI := ctx.GetContext(PartialMessageContextKey)
			if partialMessageI != nil {
				partialMessage = append(partialMessageI.([]byte), chunk...)
			} else {
				partialMessage = chunk
			}
			messages := strings.Split(string(partialMessage), "\n\n")
			for i, msg := range messages {
				if i < len(messages)-1 {
					// process complete message
					processSSEMessage(ctx, config, msg, log)
				}
			}
			if !strings.HasSuffix(string(partialMessage), "\n\n") {
				ctx.SetContext(PartialMessageContextKey, []byte(messages[len(messages)-1]))
			} else {
				ctx.SetContext(PartialMessageContextKey, nil)
			}
		}
		return chunk
	}
	// last chunk
	key := keyI.(string)
	stream := ctx.GetContext(StreamContextKey)
	var value string
	if stream == nil {
		var body []byte
		tempContentI := ctx.GetContext(CacheContentContextKey)
		if tempContentI != nil {
			body = append(tempContentI.([]byte), chunk...)
		} else {
			body = chunk
		}
		bodyJson := gjson.ParseBytes(body)

		value = TrimQuote(bodyJson.Get(config.CacheValueFrom.ResponseBody).Raw)
		log.Infof("no stream value=%s", value)
		if value == "" {
			log.Warnf("parse value from response body failded, body:%s", body)
			return chunk
		}
	} else {
		if len(chunk) > 0 {
			var lastMessage []byte
			partialMessageI := ctx.GetContext(PartialMessageContextKey)
			if partialMessageI != nil {
				lastMessage = append(partialMessageI.([]byte), chunk...)
			} else {
				lastMessage = chunk
			}
			if !strings.HasSuffix(string(lastMessage), "\n\n") {
				log.Warnf("invalid lastMessage:%s", lastMessage)
				return chunk
			}
			// remove the last \n\n
			lastMessage = lastMessage[:len(lastMessage)-2]
			value = processSSEMessage(ctx, config, string(lastMessage), log)
		} else {
			tempContentI := ctx.GetContext(CacheContentContextKey)
			if tempContentI == nil {
				return chunk
			}
			value = tempContentI.(string)
		}
		log.Infof("stream value=%s", value)
	}
	config.redisClient.Set(config.CacheKeyPrefix+key, value, nil)
	if config.CacheTTL != 0 {
		config.redisClient.Expire(config.CacheKeyPrefix+key, config.CacheTTL, nil)
	}
	return chunk
}
