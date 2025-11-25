package vectordb

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/config"
	"github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/schema"

	// "github.com/milvus-io/milvus-sdk-go/v2/client"

	"github.com/milvus-io/milvus/client/v2/column"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/index"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

const (
	MILVUS_DUMMY_DIM     = 8
	MILVUS_PROVIDER_TYPE = "milvus"
)

var (
	STOPWORDS = []string{
		"a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
		"can", "could", "did", "do", "does", "doing", "done", "each", "few", "for",
		"from", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
		"him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its",
		"itself", "me", "more", "most", "my", "myself", "no", "nor", "not", "now",
		"of", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out",
		"over", "own", "same", "she", "should", "so", "some", "such", "than", "that",
		"the", "their", "theirs", "them", "themselves", "then", "there", "these",
		"they", "this", "those", "through", "to", "too", "under", "until", "up",
		"very", "was", "we", "were", "what", "when", "where", "which", "while",
		"who", "whom", "why", "will", "with", "would", "you", "your", "yours",
		"yourself", "yourselves",
	}
)

// MilvusProviderInitializer initializes the Milvus vector store provider
type milvusProviderInitializer struct{}

// InitConfig initializes the configuration with default values if not set
func (m *milvusProviderInitializer) InitConfig(cfg *config.VectorDBConfig) error {
	if cfg.Provider != MILVUS_PROVIDER_TYPE {
		return fmt.Errorf("provider type mismatch: expected %s, got %s", MILVUS_PROVIDER_TYPE, cfg.Provider)
	}

	// Set default values
	if cfg.Host == "" {
		cfg.Host = "localhost"
	}
	if cfg.Port == 0 {
		cfg.Port = 19530
	}
	if cfg.Database == "" {
		cfg.Database = "default"
	}

	if cfg.Collection == "" {
		cfg.Collection = schema.DEFAULT_DOCUMENT_COLLECTION
	}

	return nil
}

// ValidateConfig validates the configuration parameters
func (m *milvusProviderInitializer) ValidateConfig(cfg *config.VectorDBConfig) error {
	if cfg.Host == "" {
		return fmt.Errorf("milvus host is required")
	}
	if cfg.Port <= 0 {
		return fmt.Errorf("milvus port must be positive")
	}

	if cfg.Database == "" {
		return fmt.Errorf("milvus database is required")
	}

	if cfg.Collection == "" {
		return fmt.Errorf("milvus document collection is required")
	}
	return nil
}

// CreateProvider creates a new Milvus vector store provider instance
func (m *milvusProviderInitializer) CreateProvider(cfg *config.VectorDBConfig, dim int) (VectorStoreProvider, error) {
	if err := m.InitConfig(cfg); err != nil {
		return nil, err
	}
	if err := m.ValidateConfig(cfg); err != nil {
		return nil, err
	}
	provider, err := NewMilvusProvider(cfg, dim)
	return provider, err
}

// MilvusProvider implements the vector store provider interface for Milvus
type MilvusProvider struct {
	client     *milvusclient.Client
	config     *config.VectorDBConfig
	collection string
	mapper     VectorDBMapper
	dimensions int
}

// NewMilvusProvider creates a new instance of MilvusProvider
func NewMilvusProvider(cfg *config.VectorDBConfig, dimensions int) (VectorStoreProvider, error) {
	// Create Milvus client
	connectParam := milvusclient.ClientConfig{
		Address: fmt.Sprintf("%s:%d", cfg.Host, cfg.Port),
	}
	connectParam.DBName = cfg.Database
	// Add authentication if credentials are provided
	if cfg.Username != "" && cfg.Password != "" {
		connectParam.Username = cfg.Username
		connectParam.Password = cfg.Password
	}

	milvusClient, err := milvusclient.New(context.Background(), &connectParam)
	if err != nil {
		return nil, fmt.Errorf("failed to create milvus client: %w", err)
	}

	mapper, err := NewDefaultVectorDBMapper(MILVUS_PROVIDER_TYPE, cfg.Mapping)
	if err != nil {
		return nil, fmt.Errorf("failed to create default vector db mapper: %w", err)
	}

	provider := &MilvusProvider{
		client:     milvusClient,
		config:     cfg,
		collection: cfg.Collection,
		mapper:     mapper,
		dimensions: dimensions,
	}
	ctx := context.Background()
	if err := provider.CreateCollection(ctx, dimensions); err != nil {
		return nil, err
	}
	return provider, nil
}

func (m *MilvusProvider) buildSchema() (*entity.Schema, error) {
	// Create Milvus collection Schema
	idField, _ := m.mapper.GetIDField()
	isIDAuto := idField.IsAutoID()
	schema := entity.NewSchema().
		WithName(m.collection).
		WithDescription("Knowledge document collection").
		WithAutoID(isIDAuto).
		WithDynamicFieldEnabled(false)
	// Add fields
	var fieldEntity *entity.Field
	fieldMappings, _ := m.mapper.GetFieldMappings()
	for _, field := range fieldMappings {
		fieldEntity = nil
		maxLength := field.MaxLength()
		switch field.StandardName {
		case "id":
			isIDAuto := field.IsAutoID()
			fieldEntity = entity.NewField().
				WithName(field.RawName).
				WithDataType(entity.FieldTypeVarChar).
				WithMaxLength(int64(maxLength)).
				WithIsPrimaryKey(true)
			if isIDAuto {
				fieldEntity.WithIsAutoID(true)
			}
			schema.WithField(fieldEntity)
		case "content":
			fieldEntity = entity.NewField().
				WithName(field.RawName).
				WithDataType(entity.FieldTypeVarChar).
				WithMaxLength(int64(maxLength))

			if m.config.HybridSearch.Enabled {
				analyzerParams := map[string]any{
					"tokenizer": "standard",
					"filter": []any{
						"lowercase",
						map[string]any{
							"type":       "stop",
							"stop_words": STOPWORDS,
						},
					},
				}

				fieldEntity.WithEnableAnalyzer(true).WithAnalyzerParams(analyzerParams)
			}

			schema.WithField(fieldEntity)
		case "vector":
			fieldEntity = entity.NewField().
				WithName(field.RawName).
				WithDataType(entity.FieldTypeFloatVector).
				WithDim(int64(m.dimensions))
			schema.WithField(fieldEntity)
		case "metadata":
			fieldEntity = entity.NewField().
				WithName(field.RawName).
				WithDataType(entity.FieldTypeJSON)
			schema.WithField(fieldEntity)
		case "created_at":
			fieldEntity = entity.NewField().
				WithName(field.RawName).
				WithDataType(entity.FieldTypeInt64)
			schema.WithField(fieldEntity)
		}
	}
	if m.config.HybridSearch.Enabled {
		sparseVectorField, _ := m.mapper.GetSparseVectorField()
		textField, _ := m.mapper.GetRawField("content")
		if sparseVectorField != nil && textField != nil {
			function := entity.NewFunction().
				WithName("text_bm25_emb").
				WithInputFields(textField.RawName).
				WithOutputFields(sparseVectorField.RawName).
				WithType(entity.FunctionTypeBM25)
			// Add sparse vector field
			// Note: BM25 function configuration may need to be done separately via Milvus API
			sparseVectorField := entity.NewField().
				WithName(sparseVectorField.RawName).
				WithDataType(entity.FieldTypeSparseVector).
				WithDescription("BM25 sparse vector field for hybrid search")
			schema.WithField(sparseVectorField).WithFunction(function)
		}
	}
	return schema, nil
}

func (m *MilvusProvider) GetMetricType(metricType string) entity.MetricType {
	switch strings.ToUpper(metricType) {
	case "L2":
		return entity.L2
	case "IP":
		return entity.IP
	case "COSINE":
		return entity.COSINE
	case "HAMMING":
		return entity.HAMMING
	case "JACCARD":
		return entity.JACCARD
	case "TANIMOTO":
		return entity.TANIMOTO
	case "SUBSTRUCTURE":
		return entity.SUBSTRUCTURE
	case "SUPERSTRUCTURE":
		return entity.SUPERSTRUCTURE
	default:
		return entity.IP
	}
}

func (m *MilvusProvider) buildSparseIndex(dropRatio float64) (index.Index, error) {
	index := index.NewSparseInvertedIndex(entity.BM25, dropRatio)
	return index, nil
}

func (m *MilvusProvider) buildVectorIndex() (index.Index, error) {
	// Map index type
	indexConfig, _ := m.mapper.GetIndexConfig()
	searchConfig, _ := m.mapper.GetSearchConfig()
	// Map index parameters
	milvusIndexType := strings.ToUpper(indexConfig.IndexType)
	if milvusIndexType == "" {
		milvusIndexType = "HNSW"
	}
	metricType := m.GetMetricType(searchConfig.MetricType)
	switch milvusIndexType {
	case "FLAT":
		// FLAT index doesn't need additional parameters
		index := index.NewFlatIndex(metricType)
		return index, nil

	case "BIN_FLAT":
		// BIN_FLAT index doesn't need additional parameters
		// nlist := 128
		// if nlistVal, err := indexConfig.ParamsInt64("nlist"); err == nil {
		// 	nlist = int(nlistVal)
		// }
		index := index.NewBinFlatIndex(metricType)
		return index, nil

	case "IVF_FLAT":
		// Default parameters
		nlist := 128
		if nlistVal, err := indexConfig.ParamsInt64("nlist"); err == nil {
			nlist = int(nlistVal)
		}
		index := index.NewIvfFlatIndex(metricType, nlist)
		return index, nil

	case "BIN_IVF_FLAT":
		// Default parameters
		nlist := 128
		if nlistVal, err := indexConfig.ParamsInt64("nlist"); err == nil {
			nlist = int(nlistVal)
		}
		index := index.NewBinIvfFlatIndex(metricType, nlist)
		return index, nil

	case "IVF_SQ8":
		// Default parameters
		nlist := 128
		if nlistVal, err := indexConfig.ParamsInt64("nlist"); err == nil {
			nlist = int(nlistVal)
		}
		index := index.NewIvfSQ8Index(metricType, nlist)
		return index, nil

	case "IVF_PQ":
		// Default parameters
		nlist := 128
		m := 4
		nbits := 8

		if nlistVal, err := indexConfig.ParamsInt64("nlist"); err == nil {
			nlist = int(nlistVal)
		}
		if mVal, err := indexConfig.ParamsFloat64("m"); err == nil {
			m = int(mVal)
		}
		if nbitsVal, err := indexConfig.ParamsInt64("nbits"); err == nil {
			nbits = int(nbitsVal)
		}

		index := index.NewIvfPQIndex(metricType, nlist, m, nbits)
		return index, nil

	case "HNSW":
		// Default parameters
		m := 8
		efConstruction := 64
		if mVal, err := indexConfig.ParamsInt64("M"); err == nil {
			m = int(mVal)
		}
		if efConstructionVal, err := indexConfig.ParamsInt64("efConstruction"); err == nil {
			efConstruction = int(efConstructionVal)
		}
		index := index.NewHNSWIndex(metricType, m, efConstruction)
		return index, nil

	case "DISKANN":
		// DISKANN index parameters
		index := index.NewDiskANNIndex(metricType)

		return index, nil

	case "SCANN":
		// SCANN index parameters
		nlist := 128
		with_raw_data := false
		if nlistVal, err := indexConfig.ParamsInt64("nlist"); err == nil {
			nlist = int(nlistVal)
		}
		if with_raw_dataVal, err := indexConfig.ParamsBool("with_raw_data"); err == nil {
			with_raw_data = with_raw_dataVal
		}
		index := index.NewSCANNIndex(metricType, nlist, with_raw_data)
		return index, nil

	case "AUTOINDEX":
		// Auto index
		index := index.NewAutoIndex(metricType)
		return index, nil

	default:
		return nil, fmt.Errorf("unsupported index type: %s", milvusIndexType)
	}
}

// CreateCollection creates a new collection with the specified dimension
func (m *MilvusProvider) CreateCollection(ctx context.Context, dim int) error {
	// Check if collection exists
	document_exists, err := m.client.HasCollection(context.Background(), milvusclient.NewHasCollectionOption(m.collection))
	if err != nil {
		return fmt.Errorf("failed to check %s collection existence: %w", m.collection, err)
	}

	if !document_exists {
		fmt.Printf("create collection %s\n", m.collection)
		// Create schema
		schema, err := m.buildSchema()
		if err != nil {
			return fmt.Errorf("failed to build schema: %w", err)
		}
		// Create collection
		err = m.client.CreateCollection(context.Background(), milvusclient.NewCreateCollectionOption(m.collection, schema))
		if err != nil {
			return fmt.Errorf("failed to create collection: %w", err)
		}
		// Create vector index
		vectorIndex, err := m.buildVectorIndex()
		vectorField, _ := m.mapper.GetVectorField()
		if err != nil {
			return fmt.Errorf("failed to create vector index: %w", err)
		}
		_, err = m.client.CreateIndex(context.Background(), milvusclient.NewCreateIndexOption(m.collection, vectorField.RawName, vectorIndex))
		if err != nil {
			return fmt.Errorf("failed to create vector index: %w", err)
		}
		// create sparse index if hybrid search is enabled
		if m.config.HybridSearch.Enabled {
			sparseIndex, _ := m.buildSparseIndex(0.2)
			sparseField, _ := m.mapper.GetSparseVectorField()

			if sparseField == nil {
				return fmt.Errorf("sparse field not found")
			}
			_, err = m.client.CreateIndex(context.Background(), milvusclient.NewCreateIndexOption(m.collection, sparseField.RawName, sparseIndex))
			if err != nil {
				return fmt.Errorf("failed to create sparse index: %w", err)
			}
		}
	}
	// Load collection
	_, err2 := m.client.LoadCollection(context.Background(), milvusclient.NewLoadCollectionOption(m.collection))
	if err2 != nil {
		return fmt.Errorf("failed to load document collection: %w", err2)
	}
	return nil
}

// DropCollection removes the collection from the database
func (m *MilvusProvider) DropCollection(ctx context.Context) error {
	// Check if collection exists
	exists, err := m.client.HasCollection(context.Background(), milvusclient.NewHasCollectionOption(m.collection))
	if err != nil {
		return fmt.Errorf("failed to check %s collection existence: %w", m.collection, err)
	}
	if !exists {
		return fmt.Errorf("collection %s does not exist", m.collection)
	}
	// Drop collection
	err = m.client.DropCollection(context.Background(), milvusclient.NewDropCollectionOption(m.collection))
	if err != nil {
		return fmt.Errorf("failed to drop collection: %w", err)
	}
	return nil
}

// AddDoc adds documents to the vector database
func (m *MilvusProvider) AddDoc(ctx context.Context, docs []schema.Document) error {
	if len(docs) == 0 {
		return nil
	}

	// Get field mappings
	fieldMappings, err := m.mapper.GetFieldMappings()
	if err != nil {
		return fmt.Errorf("failed to get field mappings: %w", err)
	}
	// Prepare data and columns
	columns := make([]column.Column, 0, len(fieldMappings))
	// Create corresponding column data for each field
	for _, field := range fieldMappings {
		// Skip ID field if configured as auto ID
		if field.IsPrimaryKey() && field.IsAutoID() {
			continue
		}
		switch field.StandardName {
		case "id":
			// Handle string type fields
			values := make([]string, len(docs))
			for i, doc := range docs {
				values[i] = doc.ID
			}
			columns = append(columns, column.NewColumnVarChar(field.RawName, values))
		case "content":
			values := make([]string, len(docs))
			for i, doc := range docs {
				values[i] = doc.Content
			}
			columns = append(columns, column.NewColumnVarChar(field.RawName, values))

		case "vector":
			// Handle vector fields
			vectors := make([][]float32, len(docs))
			for i, doc := range docs {
				vectors[i] = doc.Vector
			}
			columns = append(columns, column.NewColumnFloatVector(field.RawName, len(vectors[0]), vectors))
		case "metadata":
			// Handle JSON type fields (like metadata)
			values := make([][]byte, len(docs))
			for i, doc := range docs {
				// Serialize metadata
				metadataBytes, err := json.Marshal(doc.Metadata)
				if err != nil {
					return fmt.Errorf("failed to marshal metadata for doc %s: %w", doc.ID, err)
				}
				values[i] = metadataBytes
			}
			columns = append(columns, column.NewColumnJSONBytes(field.RawName, values))
		case "created_at":
			// Handle integer type fields
			values := make([]int64, len(docs))
			for i, doc := range docs {
				values[i] = doc.CreatedAt.UnixMilli()
			}
			columns = append(columns, column.NewColumnInt64(field.RawName, values))
		}
	}
	// Insert data
	_, err = m.client.Insert(ctx, milvusclient.NewColumnBasedInsertOption(m.collection, columns...))
	if err != nil {
		return fmt.Errorf("failed to insert documents: %w", err)
	}
	// Flush data
	_, err = m.client.Flush(ctx, milvusclient.NewFlushOption(m.collection))
	if err != nil {
		return fmt.Errorf("failed to flush collection: %w", err)
	}

	return nil
}

// DeleteDoc deletes a document by its ID
func (m *MilvusProvider) DeleteDoc(ctx context.Context, id string) error {
	// Get ID field
	idField, _ := m.mapper.GetIDField()
	// Build delete expression using the RawName of ID field
	expr := fmt.Sprintf(`%s == "%s"`, idField.RawName, id)

	// Delete data
	_, err := m.client.Delete(ctx, milvusclient.NewDeleteOption(m.collection).WithExpr(expr))
	if err != nil {
		return fmt.Errorf("failed to delete documents for id %s: %w", id, err)
	}

	// Flush data
	_, err = m.client.Flush(ctx, milvusclient.NewFlushOption(m.collection))
	if err != nil {
		return fmt.Errorf("failed to flush collection after delete: %w", err)
	}

	return nil
}

// UpdateDoc updates documents by first deleting existing ones and then adding new ones
func (m *MilvusProvider) UpdateDoc(ctx context.Context, docs []schema.Document) error {
	// Delete existing documents
	ids := make([]string, len(docs))
	for i, doc := range docs {
		ids[i] = doc.ID
	}
	if err := m.DeleteDocs(ctx, ids); err != nil {
		return fmt.Errorf("failed to delete existing documents: %w", err)
	}
	// Add new documents
	if err := m.AddDoc(ctx, docs); err != nil {
		return fmt.Errorf("failed to add new documents: %w", err)
	}

	return nil
}

func (m *MilvusProvider) buildSearchParam() (map[string]string, error) {
	// Get index configuration
	// indexConfig, err := m.mapper.GetIndexConfig()
	// if err != nil {
	// 	return nil, fmt.Errorf("failed to get index config: %w", err)
	// }
	// Get search configuration
	searchConfig, err := m.mapper.GetSearchConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to get search config: %w", err)
	}

	searchParam := make(map[string]string)
	searchParam["metric_type"] = strings.ToUpper(searchConfig.MetricType)

	return searchParam, nil

	// // Choose appropriate search parameters based on index type
	// milvusIndexType := strings.ToUpper(indexConfig.IndexType)
	// if milvusIndexType == "" {
	// 	milvusIndexType = "HNSW" // Default to HNSW index
	// }

	// switch milvusIndexType {
	// case "FLAT":
	// 	// FLAT and BIN_FLAT indices don't need additional search parameters
	// 	return searchParam, nil
	// case "BIN_FLAT", "IVF_FLAT", "BIN_IVF_FLAT", "IVF_SQ8":
	// 	// Search parameters for IVF series indices
	// 	nprobe := 16 // Default value
	// 	if nprobeVal, err := searchConfig.ParamsFloat64("nprobe"); err == nil {
	// 		nprobe = int(nprobeVal)
	// 	}
	// 	searchParam["nprobe"] = strconv.Itoa(nprobe)
	// 	return searchParam, nil

	// case "IVF_PQ":
	// 	// Search parameters for IVF_PQ index
	// 	nprobe := 16 // Default value
	// 	if nprobeVal, err := searchConfig.ParamsFloat64("nprobe"); err == nil {
	// 		nprobe = int(nprobeVal)
	// 	}
	// 	searchParam["nprobe"] = strconv.Itoa(nprobe)
	// 	return searchParam, nil

	// case "HNSW":
	// 	// Search parameters for HNSW index
	// 	efSearch := 16 // Default value
	// 	if efSearchVal, err := searchConfig.ParamsFloat64("ef"); err == nil {
	// 		efSearch = int(efSearchVal)
	// 	}
	// 	searchParam["ef"] = strconv.Itoa(efSearch)
	// 	return searchParam, nil

	// case "IVF_HNSW":
	// 	// Search parameters for IVF_HNSW index
	// 	nprobe := 16   // Default value
	// 	efSearch := 64 // Default value
	// 	if nprobeVal, err := searchConfig.ParamsFloat64("nprobe"); err == nil {
	// 		nprobe = int(nprobeVal)
	// 	}
	// 	if efSearchVal, err := searchConfig.ParamsFloat64("ef"); err == nil {
	// 		efSearch = int(efSearchVal)
	// 	}
	// 	searchParam["nprobe"] = strconv.Itoa(nprobe)
	// 	searchParam["ef"] = strconv.Itoa(efSearch)
	// 	return searchParam, nil

	// case "SCANN":
	// 	// Search parameters for SCANN index
	// 	nprobe := 16 // Default value
	// 	reorder_k := 64
	// 	if nprobeVal, err := searchConfig.ParamsFloat64("nprobe"); err == nil {
	// 		nprobe = int(nprobeVal)
	// 	}
	// 	if reorderKVal, err := searchConfig.ParamsInt64("reorder_k"); err == nil {
	// 		reorder_k = int(reorderKVal)
	// 	}
	// 	searchParam["nprobe"] = strconv.Itoa(nprobe)
	// 	searchParam["reorder_k"] = strconv.Itoa(reorder_k)
	// 	return searchParam, nil

	// case "DISKANN":
	// 	// Search parameters for DISKANN index
	// 	search_list := 100 // Default value
	// 	if searchListVal, err := searchConfig.ParamsInt64("search_list"); err == nil {
	// 		search_list = int(searchListVal)
	// 	}
	// 	searchParam["search_list"] = strconv.Itoa(search_list)
	// 	return searchParam, nil

	// case "AUTOINDEX":
	// 	level := 8
	// 	if levelVal, err := searchConfig.ParamsInt64("level"); err == nil {
	// 		level = int(levelVal)
	// 	}
	// 	searchParam["level"] = strconv.Itoa(level)
	// 	return searchParam, nil
	// default:
	// 	// Default to using HNSW search parameters
	// 	searchParam["ef"] = strconv.Itoa(16)
	// 	return searchParam, nil
	// }
}

// SearchDocs performs similarity search for documents
func (m *MilvusProvider) SearchDocs(ctx context.Context, query string, vector []float32, options *schema.SearchOptions) ([]schema.SearchResult, error) {
	if options == nil {
		options = &schema.SearchOptions{TopK: 10, Threshold: 0.0}
	}

	// Build search parameters
	sp, err := m.buildSearchParam()
	if err != nil {
		return nil, fmt.Errorf("failed to build search param: %w", err)
	}

	outputFields, _ := m.mapper.GetOutputFields()
	vectorField, _ := m.mapper.GetVectorField()

	searchOption := milvusclient.NewSearchOption(m.collection, options.TopK, []entity.Vector{entity.FloatVector(vector)}).WithANNSField(vectorField.RawName).WithOutputFields(outputFields...)
	for key, value := range sp {
		searchOption = searchOption.WithSearchParam(key, value)
	}

	// fmt.Printf("search option: %+v\n", searchOption)

	// Build filter expression
	searchResults, err := m.client.Search(ctx, searchOption)
	if err != nil {
		return nil, fmt.Errorf("failed to search documents: %w", err)
	}

	// Parse results
	var results []schema.SearchResult
	for _, result := range searchResults {
		for i := 0; i < result.ResultCount; i++ {
			id, _ := result.IDs.Get(i)
			score := result.Scores[i]
			// Get field data
			var content string
			var metadata map[string]interface{}
			for _, field := range result.Fields {
				fieldMapping, err := m.mapper.GetField(field.Name())
				if err != nil {
					continue
				}
				fieldName := strings.ToLower(fieldMapping.StandardName)
				switch fieldName {
				case "content":
					if contentCol, ok := field.(*column.ColumnVarChar); ok {
						if contentVal, err := contentCol.Get(i); err == nil {
							if contentStr, ok := contentVal.(string); ok {
								content = contentStr
							}
						}
					}
				case "metadata":
					if metaCol, ok := field.(*column.ColumnJSONBytes); ok {
						if metaVal, err := metaCol.Get(i); err == nil {
							if metaBytes, ok := metaVal.([]byte); ok {
								if err := json.Unmarshal(metaBytes, &metadata); err != nil {
									metadata = make(map[string]interface{})
								}
							}
						}
					}
				}
			}
			searchResult := schema.SearchResult{
				Document: schema.Document{
					ID:       fmt.Sprintf("%s", id),
					Content:  content,
					Metadata: metadata,
				},
				Score: float64(score),
			}

			// Filter results by threshold
			if searchResult.Score < options.Threshold {
				continue
			}

			results = append(results, searchResult)
		}
	}
	return results, nil
}

// SearchDocs performs similarity search for documents
func (m *MilvusProvider) SearchHybridDocs(ctx context.Context, query string, vector []float32, options *schema.SearchOptions) ([]schema.SearchResult, error) {
	if options == nil {
		options = &schema.SearchOptions{TopK: 10, Threshold: 0.0}
	}

	// Build search parameters
	sp, err := m.buildSearchParam()
	if err != nil {
		return nil, fmt.Errorf("failed to build search param: %w", err)
	}

	outputFields, _ := m.mapper.GetOutputFields()
	vectorField, _ := m.mapper.GetVectorField()
	sparseVectorField, _ := m.mapper.GetSparseVectorField()

	if sparseVectorField == nil {
		return nil, fmt.Errorf("sparse vector field not found for hybrid search")
	}

	// Build vector search request
	request1 := milvusclient.NewAnnRequest(vectorField.RawName, options.TopK, entity.FloatVector(vector))
	for key, value := range sp {
		request1 = request1.WithSearchParam(key, value)
	}

	// Build sparse vector search request
	annParam := index.NewSparseAnnParam()
	annParam.WithDropRatio(0.2)
	request2 := milvusclient.NewAnnRequest(sparseVectorField.RawName, options.TopK, entity.Text(query)).
		WithAnnParam(annParam)

	// Build reranker based on configuration
	var reranker milvusclient.Reranker
	switch m.config.HybridSearch.Ranker {
	case config.WeightedRanker:
		// Use weighted reranker with configured vector weight
		vectorWeight := m.config.HybridSearch.VectorWeight
		if vectorWeight <= 0 {
			vectorWeight = 0.5 // Default weight
		}
		reranker = milvusclient.NewWeightedReranker([]float64{vectorWeight, 1.0 - vectorWeight})
	default:
		// Default to RRF reranker
		reranker = milvusclient.NewRRFReranker()
	}

	// Perform hybrid search
	searchResults, err := m.client.HybridSearch(ctx, milvusclient.NewHybridSearchOption(
		m.collection,
		options.TopK,
		request1,
		request2,
	).WithReranker(reranker).WithOutputFields(outputFields...))

	if err != nil {
		return nil, fmt.Errorf("failed to perform hybrid search: %w", err)
	}

	// Parse results
	var results []schema.SearchResult
	for _, result := range searchResults {
		for i := 0; i < result.ResultCount; i++ {
			id, _ := result.IDs.Get(i)
			score := result.Scores[i]
			// Get field data
			var content string
			var metadata map[string]interface{}
			for _, field := range result.Fields {
				fieldMapping, err := m.mapper.GetField(field.Name())
				if err != nil {
					continue
				}
				fieldName := strings.ToLower(fieldMapping.StandardName)
				switch fieldName {
				case "content":
					if contentCol, ok := field.(*column.ColumnVarChar); ok {
						if contentVal, err := contentCol.Get(i); err == nil {
							if contentStr, ok := contentVal.(string); ok {
								content = contentStr
							}
						}
					}
				case "metadata":
					if metaCol, ok := field.(*column.ColumnJSONBytes); ok {
						if metaVal, err := metaCol.Get(i); err == nil {
							if metaBytes, ok := metaVal.([]byte); ok {
								if err := json.Unmarshal(metaBytes, &metadata); err != nil {
									metadata = make(map[string]interface{})
								}
							}
						}
					}
				}
			}
			searchResult := schema.SearchResult{
				Document: schema.Document{
					ID:       fmt.Sprintf("%s", id),
					Content:  content,
					Metadata: metadata,
				},
				Score: float64(score),
			}
			// Filter results by threshold
			if searchResult.Score < options.Threshold {
				continue
			}
			results = append(results, searchResult)
		}
	}
	return results, nil
}

// DeleteDocs deletes multiple documents by their IDs
func (m *MilvusProvider) DeleteDocs(ctx context.Context, ids []string) error {
	if len(ids) == 0 {
		return nil
	}

	// Build delete expression
	// Milvus expects string values to be quoted within the expression, otherwise the parser will
	// treat the hyphen inside UUID as a minus operator and raise a parse error.
	quotedIDs := make([]string, len(ids))
	for i, id := range ids {
		quotedIDs[i] = fmt.Sprintf("\"%s\"", id)
	}

	idField, _ := m.mapper.GetIDField()
	expr := fmt.Sprintf("%s in [%s]", idField.RawName, strings.Join(quotedIDs, ","))

	// Delete data
	_, err := m.client.Delete(ctx, milvusclient.NewDeleteOption(m.collection).WithExpr(expr))
	if err != nil {
		return fmt.Errorf("failed to delete documents: %w", err)
	}
	// Flush data
	_, err = m.client.Flush(ctx, milvusclient.NewFlushOption(m.collection))
	if err != nil {
		return fmt.Errorf("failed to flush collection after delete: %w", err)
	}

	return nil
}

// ListDocs retrieves all documents with optional limit
func (m *MilvusProvider) ListDocs(ctx context.Context, limit int) ([]schema.Document, error) {
	// Build query expression
	// Query all relevant documents
	outputFields, _ := m.mapper.GetOutputFields()
	queryOption := milvusclient.NewQueryOption(m.collection).WithOutputFields(outputFields...).WithOffset(0).WithLimit(limit)
	queryResult, err := m.client.Query(ctx, queryOption)

	if err != nil {
		return nil, fmt.Errorf("failed to query documents: %w", err)
	}

	rowCount := queryResult.ResultCount
	if rowCount <= 0 {
		return []schema.Document{}, nil
	}

	documents := make([]schema.Document, 0, rowCount)

	// Parse query results
	for i := 0; i < rowCount; i++ {
		var (
			id        string
			content   string
			metadata  map[string]interface{}
			createdAt int64
		)

		for _, col := range queryResult.Fields {
			fieldMapping, err := m.mapper.GetField(col.Name())
			if err != nil {
				continue
			}
			fieldName := strings.ToLower(fieldMapping.StandardName)
			switch fieldName {
			case "id":
				if v, err := col.(*column.ColumnVarChar).Get(i); err == nil {
					id = v.(string)
				}
			case "content":
				if v, err := col.(*column.ColumnVarChar).Get(i); err == nil {
					content = v.(string)
				}
			case "metadata":
				if v, err := col.(*column.ColumnJSONBytes).Get(i); err == nil {
					if bytes, ok := v.([]byte); ok {
						_ = json.Unmarshal(bytes, &metadata)
					}
				}
			case "created_at":
				if v, err := col.(*column.ColumnInt64).Get(i); err == nil {
					createdAt = v.(int64)
				}
			}
		}

		doc := schema.Document{
			ID:        id,
			Content:   content,
			Metadata:  metadata,
			CreatedAt: time.UnixMilli(createdAt),
		}
		documents = append(documents, doc)
	}
	return documents, nil
}

// GetProviderType returns the provider type identifier
func (m *MilvusProvider) GetProviderType() string {
	return MILVUS_PROVIDER_TYPE
}

// Close closes the connection to the Milvus server
func (m *MilvusProvider) Close() error {
	if m.client != nil {
		return m.client.Close(context.Background())
	}
	return nil
}
