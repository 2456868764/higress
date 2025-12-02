from __future__ import annotations
from typing import List
import hashlib
import uuid
import operator
import os
import re

from pymilvus import MilvusClient, FieldSchema, DataType, Function, FunctionType, CollectionSchema, Collection, AnnSearchRequest, RRFRanker
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pymilvus import WeightedRanker


from base import VectorStoreBase, CorpusFile
from config import config, VectorStoreConfig
from utils import logger

# Get KB_ROOT_PATH from config
KB_ROOT_PATH = config.vector_store.kb_root_path
MILVUS_PERSISTENT_PATH = os.path.join(KB_ROOT_PATH, 'milvus_persistent')


def md5_encryption(data):
    md5 = hashlib.md5()
    md5.update(data.encode('utf-8'))
    return md5.hexdigest()


def get_stopwords():
    """
    Get stopwords list for text analyzer.
    Returns Chinese stopwords by default.
    """
    # Default Chinese stopwords list
    # In production, you might want to load from a file or use a library
    return [
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
		"yourself", "yourselves"
    ]

def clean_text(text: str) -> str:
    """清洗文本"""
    # 统一换行
    text = re.sub(r'\r\n|\r', '\n', text)
    # 去除多余空白
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    # 去除特殊字符
    return text.strip()

class MilvusHybridVectorStore(VectorStoreBase):

    def __init__(self,
                 embedding_model: Embeddings,
                 config: VectorStoreConfig):
        self.embeddings = embedding_model
        self.knowledge_base_name = config.collection_name
        self.collection_name = config.collection_name
        self.config = config
        self.reranker_type = config.reranker_type
        self.reranker_weight = config.reranker_weight
        self._load_milvus()

    def _load_milvus(self):
        connection_args = {
            "host": self.config.host,
            "port": self.config.port,
            "user": self.config.user,
            "password": self.config.password,
            "secure": False,
            "db_name": self.config.name,
        }
        index_params = self.config.kwargs.get("index_params", None)
        search_params = self.config.kwargs.get("search_params", None)
        if self.config.host == "" and self.config.port == "":
            # milvus lite
            if not os.path.exists(MILVUS_PERSISTENT_PATH):
                os.makedirs(MILVUS_PERSISTENT_PATH)
            db_path = os.path.join(MILVUS_PERSISTENT_PATH, self.config.name + ".db")
            print(f"milvus lite db path:{db_path}")
            self.pyclient = MilvusClient(db_path)
        else:
            self.pyclient = MilvusClient(
                uri="http://"+self.config.host+":"+self.config.port,
                db_name=self.config.name,
                user=self.config.user,
                password=self.config.password
            )
        self.create_collection()

    def create_vectorstore(self):
        pass

    def drop_vectorstore(self):
        if self.pyclient.has_collection(self.collection_name):
            self.pyclient.release_collection(self.collection_name)
            self.pyclient.drop_collection(self.collection_name)

    def clear_vectorstore(self):
        if self.pyclient.has_collection(self.collection_name):
            self.pyclient.release_collection(self.collection_name)
            self.pyclient.drop_collection(self.collection_name)
            self._load_milvus()

    def create_collection(self):
        if not self.pyclient.has_collection(self.collection_name):
            embedding = self.embeddings.embed_documents(["初始化"])
            dense_dim = len(embedding[0])
            schema = self.pyclient.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
            )
            schema.add_field("id", DataType.VARCHAR,is_primary=True, auto_id=False, max_length=256)
            schema.add_field("metadata", DataType.JSON)
            STOPWORDS = get_stopwords()
            analyzer_params = {
                "tokenizer": "standard",
                "filter": [
                    "lowercase",  # 内置过滤器
                    {
                        "type": "stop", 
                        "stop_words": STOPWORDS,
                    }
                ]
            }
            schema.add_field("content", DataType.VARCHAR, max_length=2048, enable_analyzer=True, analyzer_params=analyzer_params)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dense_dim)
            schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
            # schema.add_field("chunk_type", DataType.VARCHAR, max_length=256)

            bm25_function = Function(
                name="bm25",
                input_field_names=["content"],
                output_field_names=["sparse_vector"],
                function_type=FunctionType.BM25,
            )

            schema.add_function(bm25_function)
            index_params = self.pyclient.prepare_index_params()

            index_params.add_index(
                field_name="vector",
                metric_type="IP", # COSINE, IP
                index_type="HNSW", # FLAT,HNSW
                index_name="vector_index",
                params={"params": {"M": 8, "efConstruction": 64}}
            )

            index_params.add_index(
                field_name="sparse_vector",
                index_name="text_sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
                params={"inverted_index_algo": "DAAT_MAXSCORE"}, # or "DAAT_WAND" or "TAAT_NAIVE"
            )

            self.pyclient.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params,
            )
        else:
            res = self.pyclient.describe_index(
                collection_name=self.collection_name,
                index_name="vector_index"
            )
            print("index info:", res)

    
    def delete_doc(self, filename):
        """
        删除指定文件的所有chunk记录
        :param filename:
        :return:
        """
        if self.pyclient.has_collection(self.collection_name):
            delete_list = [item.get("id") for item in
                           self.pyclient.query(collection_name=self.collection_name,
                                               filter=f'metadata["md5source"] == "{md5_encryption(filename)}"',
                                               output_fields=["id"])]

            if len(delete_list) > 0:
                self.pyclient.delete(collection_name=self.collection_name,
                                     filter=f'id in {delete_list}')
                logger.warning(f"成功删除文件 {filename} {str(len(delete_list))} 条记录")
            else:
                logger.warning(f"vs中不存在文件 {filename} 相关的记录，不需要删除")
        else:
            logger.warning(f"vs为空，没有可删除的记录")

    def exist_doc(self, filename):
        """
        判断指定文件是否存在于向量数据库中
        :param filename:
        :return:
        """
        if self.pyclient.has_collection(self.collection_name):
            md5filename = md5_encryption(filename)
            filter = f'metadata["md5source"] == "{md5filename}"'
            items = self.pyclient.query(collection_name=self.collection_name,
                                               filter=filter,
                                               output_fields=["id"])
            delete_list = [item.get("id") for item in items]
            if len(delete_list) > 0:
                print(f"文件 {filename} 存在于向量数据库中, 相关记录数: {str(len(delete_list))}")
                logger.warning(f"文件 {filename} {str(len(delete_list))} 条记录")
                return True
            else:
                print(f"文件 {filename} 不存在于向量数据库中")
                logger.warning(f"vs中不存在文件 {filename} 相关的记录")
                return False
        else:
            logger.warning(f"vs为空，没有可查询的记录")
            return False
                    

    def update_doc(self, file: CorpusFile, docs: List[Document]):
        """
        插入/更新向量数据库中的记录
        更新：若该文件的chunk已经存在，需要先将原信息删除后重新插入
        :param file:
        :param docs:
        :return:
        """
        filename = file.metadata.get("url")
        self.delete_doc(filename)
        return self.add_doc(file, docs=docs)

    def add_doc(self, file: CorpusFile, docs, **kwargs):
        """
        将chunks插入vs
        :param file:
        :param docs:
        :return:
        """
        # 将所有信息存储到metadata
        logger.info(f"开始添加记录：{file.filename}")
        doc_ids = []
        data = []
        filename = file.metadata.get("url")
        for doc in docs:
            doc_id = doc.metadata.get("id", str(uuid.uuid4()))
            metadata = doc.metadata
            append_metadata = {"md5source": md5_encryption(filename)}
            for k, v in append_metadata.items():
                metadata[k] = str(v)
            for k, v in file.metadata.items():
                metadata[k] = str(v)
               
            doc_ids.append(doc_id)
            # 为了避免不同文档内容相同导致的向量重复，在embedding时增加文档名称
            page_content = clean_text(doc.page_content)
            vector = self.embeddings.embed_documents([page_content])[0]
            data.append({"id": doc_id, "metadata": metadata, "content": page_content, "vector": vector})


        # logger.info(f"insert data: {data}")    

        self.pyclient.insert(collection_name=self.collection_name, data=data)
        self.pyclient.flush(self.collection_name)
        doc_infos = [{"id": id, "metadata": doc.metadata} for id, doc in zip(doc_ids, docs)]
        logger.info(f"添加记录结束：{file.filename}，一共{len(docs)} 条记录")
        return doc_infos


    def search_hybrid_docs(self, text, top_k, threshold, **kwargs):
        """
        :param text:
        :param top_k:
        :param threshold:
        :return: List[Tuple[Document, float]]: Result doc and score.
        """
        # 从 kwargs 中提取过滤参数
        filter_expr = None        
        # Generate dense embeddings for the query text.
        embedding = self.embeddings.embed_documents([text])
        search_embedding_param = {
            "data": embedding,
            "anns_field": "vector",
            "param": {
                "metric_type": "IP",
            },
            "limit": top_k
        }

        if filter_expr:
            search_embedding_param["expr"] = filter_expr
        
        request_embedding = AnnSearchRequest(**search_embedding_param)
        # Generate sparse embeddings for the query text.
       # full-text search (sparse)
        search_param_2 = {
            "data": [text],
            "anns_field": "sparse_vector",
            "param": {"drop_ratio_search": 0.2},
            "limit": top_k
        }

        if filter_expr:
            search_param_2["expr"] = filter_expr

        request_sparse = AnnSearchRequest(**search_param_2)
        # Store these two requests as a list in `reqs`
        reqs = [request_embedding, request_sparse]
        # Use WeightedRanker or RRFRanker to combine results
        if self.reranker_type == "weighted":
            weight = self.reranker_weight
            rerank = WeightedRanker(weight, 1-weight)
        else:
            rerank = RRFRanker()
        # hybrid search
        output_fields = ["id", "content", "metadata"]
        # 过滤表达式需要在顶层传入 hybrid_search，而不是 AnnSearchRequest
        res = self.pyclient.hybrid_search(self.collection_name, reqs, rerank, limit=top_k, output_fields=output_fields)
        # Organize results.
        docs = []
        for result in res[0]:
            data = {x: result['entity'].get(x) for x in output_fields}
            doc = self._parse_document(data)
            pair = (doc, result['distance'])
            docs.append(pair)

        if threshold is not None:
            docs = self._score_threshold_process(docs, threshold, top_k)

        # 召回父文档
        parent_doc_map = {}
        for i, tp in enumerate(docs):
            parent_id = tp[0].metadata.get("parent_id")
            if parent_id is not None: parent_doc_map[i] = parent_id

        if len(parent_doc_map) > 0:
            try:
                ids = list(set(parent_doc_map.values()))
                parent_docs = {}        # parent_id: parent_doc
                for p_doc in self.pyclient.get(collection_name=self.collection_name,
                                               ids=ids,
                                               output_fields=output_fields):
                    parent_docs[p_doc["id"]] = Document(page_content=p_doc["text"],
                                                        metadata=p_doc["metadata"])
                for doc_index in parent_doc_map:
                    docs[doc_index] = tuple([parent_docs[parent_doc_map[doc_index]], docs[doc_index][1]])

            except Exception as e:
                msg = f"路由到parent chunk失败：{e}"
                logger.error(f'{e.__class__.__name__}: {msg}', exc_info=e)    
        
        # 根据 metadata 中的 ID 去重文档
        docs = self._deduplicate_docs_by_id(docs)        
        return docs

    def search_docs(self, text, top_k, threshold, **kwargs):
        """
        :param text:
        :param top_k:
        :param threshold:
        :param kwargs: 支持 chunk_types (list) 和 doc_names (list) 过滤参数
        :return: List[Tuple[Document, float]]: Result doc and score.
        """
        # Generate dense embeddings for the query text.
        output_fields = ["id", "content", "metadata"]
        embedding = self.embeddings.embed_documents([text])
        filter_expr = None
        search_embedding_param = {
            "data": embedding,
            "anns_field": "vector",
            "search_params": {
                "metric_type": "IP",
            },
            "limit": top_k,
            "output_fields": output_fields
        }
        
        # 如果有过滤条件，添加到搜索参数中
        if filter_expr:
            search_embedding_param["filter"] = filter_expr
        
        res = self.pyclient.search(self.collection_name, **search_embedding_param)
        # Organize results.
        docs = []
        for result in res[0]:
            data = {x: result['entity'].get(x) for x in output_fields}
            doc = self._parse_document(data)
            pair = (doc, result['distance'])
            docs.append(pair)

        if threshold is not None:
            docs = self._score_threshold_process(docs, threshold, top_k)
        
        # 召回父文档
        parent_doc_map = {}
        for i, tp in enumerate(docs):
            parent_id = tp[0].metadata.get("parent_id")
            if parent_id is not None: parent_doc_map[i] = parent_id

        if len(parent_doc_map) > 0:
            try:
                ids = list(set(parent_doc_map.values()))
                parent_docs = {}        # parent_id: parent_doc
                for p_doc in self.pyclient.get(collection_name=self.collection_name,
                                               ids=ids,
                                               output_fields=output_fields):
                    parent_docs[p_doc["id"]] = Document(page_content=p_doc["text"],
                                                        metadata=p_doc["metadata"])
                for doc_index in parent_doc_map:
                    docs[doc_index] = tuple([parent_docs[parent_doc_map[doc_index]], docs[doc_index][1]])

            except Exception as e:
                msg = f"路由到parent chunk失败：{e}"
                logger.error(f'{e.__class__.__name__}: {msg}', exc_info=e)

        # 根据 metadata 中的 ID 去重文档
        docs = self._deduplicate_docs_by_id(docs)        
        return docs

    def _deduplicate_docs_by_id(self, docs):
        """
        根据 metadata 中的 ID 去重文档
        
        Args:
            docs: 包含 (Document, score) 元组的列表
            
        Returns:
            list: 去重后的文档列表，保留每个ID的最高分数文档
        """
        if not docs:
            return docs
            
        # 使用字典存储每个ID的最佳文档（最高分数）
        id_to_best_doc = {}
        for doc, score in docs:
            # 获取文档的ID，优先使用 'id' 字段，其次使用 'pk' 字段
            doc_id = doc.metadata.get('id')
            if doc_id not in id_to_best_doc or score > id_to_best_doc[doc_id][1]:
                id_to_best_doc[doc_id] = (doc, score)
        
        # 按原始顺序返回去重后的文档，保持分数排序
        deduplicated_docs = list(id_to_best_doc.values())
        # 按分数降序排序
        deduplicated_docs.sort(key=lambda x: x[1], reverse=True)
        return deduplicated_docs    

    def _parse_document(self, data: dict) -> Document:
        return Document(
            page_content=data['content'],
            metadata=data['metadata'],
        )

    def _score_threshold_process(self, docs, score_threshold, k):
        if score_threshold is not None:
            cmp = (
                operator.ge
            )
            docs = [
                (doc, similarity)
                for doc, similarity in docs
                if cmp(similarity, score_threshold)
            ]
        return docs[:k]

    def _build_filter_expression(self, chunk_types=None, doc_names=None, exclude_ids=None):
        """
        构建 Milvus 过滤表达式，支持 chunk_types 和 doc_names 数组过滤
        
        Args:
            chunk_types (list): 要过滤的 chunk_type 列表
            doc_names (list): 要过滤的 doc_name 列表
            exclude_ids (list): 要排除的文档ID列表
        
        Returns:
            str: Milvus 过滤表达式，如果没有过滤条件则返回 None
        """
        filter_conditions = []
        
        # 处理 chunk_types 过滤
        if chunk_types and isinstance(chunk_types, list) and len(chunk_types) > 0:
            # 过滤掉空值
            valid_chunk_types = [ct for ct in chunk_types if ct is not None and str(ct).strip()]
            if valid_chunk_types:
                # 构建 IN 表达式：metadata.chunk_type in ["type1", "type2"]
                chunk_type_values = ', '.join([f'"{ct}"' for ct in valid_chunk_types])
                filter_conditions.append(f'metadata["chunk_type"] in [{chunk_type_values}]')
        
        # 处理 doc_names 过滤
        if doc_names and isinstance(doc_names, list) and len(doc_names) > 0:
            # 过滤掉空值
            valid_doc_names = [dn for dn in doc_names if dn is not None and str(dn).strip()]
            if valid_doc_names:
                # 构建 IN 表达式：metadata.doc_name in ["doc1", "doc2"]
                doc_name_values = ', '.join([f'"{dn}"' for dn in valid_doc_names])
                filter_conditions.append(f'metadata["doc_name"] in [{doc_name_values}]')
        
        # 处理 exclude_ids 过滤（排除指定主键ID）
        if exclude_ids and isinstance(exclude_ids, list) and len(exclude_ids) > 0:
            valid_exclude_ids = [eid for eid in exclude_ids if eid is not None and str(eid).strip()]
            if valid_exclude_ids:
                exclude_id_values = ', '.join([f'"{eid}"' for eid in valid_exclude_ids])
                filter_conditions.append(f'pk not in [{exclude_id_values}]')
        
        # 如果有过滤条件，根据条件数量决定是否需要 AND 连接
        if len(filter_conditions) == 1:
            return filter_conditions[0]
        elif len(filter_conditions) > 1:
            return ' and '.join(filter_conditions)
        
        return None

    def _deduplicate_docs_by_id(self, docs):
        """
        根据 metadata 中的 ID 去重文档
        
        Args:
            docs: 包含 (Document, score) 元组的列表
            
        Returns:
            list: 去重后的文档列表，保留每个ID的最高分数文档
        """
        if not docs:
            return docs
            
        # 使用字典存储每个ID的最佳文档（最高分数）
        id_to_best_doc = {}
        for doc, score in docs:
            # 获取文档的ID，优先使用 'id' 字段，其次使用 'pk' 字段
            doc_id = doc.metadata.get('id')
            if doc_id not in id_to_best_doc or score > id_to_best_doc[doc_id][1]:
                id_to_best_doc[doc_id] = (doc, score)
        
        # 按原始顺序返回去重后的文档，保持分数排序
        deduplicated_docs = list(id_to_best_doc.values())
        # 按分数降序排序
        deduplicated_docs.sort(key=lambda x: x[1], reverse=True)
        return deduplicated_docs
