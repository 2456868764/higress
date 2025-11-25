from typing import List, Dict, Tuple, Type, Union
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import List
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModel
)

from langchain_core.documents import Document



class Reranker(ABC):
    """
    Reranker接口定义，所有重排序器实现都应该继承此接口
    """
    
    @abstractmethod
    def rank(self,
             query: str,
             docs: List[Document],
             top_k: int = None,
             batch_size: int = 16,
             return_documents: bool = True,
             threshold: float = 0.0):
        """
        对文档进行重排序
        
        Args:
            query: 查询文本
            docs: 待排序的文档列表
            top_k: 返回的文档数量
            batch_size: 批处理大小
            return_documents: 是否返回文档对象
            threshold: 分数阈值，低于此值的文档将被过滤
            
        Returns:
            排序后的文档列表
        """
        pass

class DefaultReranker(Reranker):

    """
    support to use rerank models like:
    1. FlagEmbedding: https://github.com/FlagOpen/FlagEmbedding
    2. BCEEmbedding: https://github.com/netease-youdao/BCEmbedding
    3. https://github.com/UKPLab/sentence-transformers/blob/737353354fbdf1a419eee864f998ffe9fdf3b682/sentence_transformers/cross_encoder/CrossEncoder.py#L20
    """

    def __init__(
            self,
            model_name_or_path: str = None,
            use_fp16: bool = False,
            max_length: int = 512,
            device: str = "auto"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, trust_remote_code=True)
        if max_length == 0:
            max_length = 512
        self.max_length = max_length
        self.need_activate = True if "bce" in model_name_or_path.lower() else False
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
                use_fp16 = False
        else:
            self.device = device

        if use_fp16:
            self.model.half()

        self.model = self.model.to(self.device)
        self.model.eval()

        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if self.num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)

    @torch.no_grad()
    def compute_score(self,
                      sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
                      batch_size: int = 256,
                      max_length: int = 512,
                      enable_tqdm: bool=False,) -> List[float]:
        if self.num_gpus > 0:
            batch_size = batch_size * self.num_gpus

        assert isinstance(sentence_pairs, list)
        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]

        all_scores = []
        for start_index in tqdm(range(0, len(sentence_pairs), batch_size), desc="Compute Scores", disable=not enable_tqdm):
            sentences_batch = sentence_pairs[start_index:start_index + batch_size]
            inputs = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=max_length,
            ).to(self.device)

            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
            if self.need_activate:
                scores = torch.sigmoid(scores)
            all_scores.extend(scores.cpu().numpy().tolist())

        # if len(all_scores) == 1:
        #     return all_scores[0]
        return all_scores

    def rank(self,
             query: str,
             docs: List[Document],
             top_k: int = None,
             batch_size: int = 16,
             return_documents: bool = True,
             threshold: float = 0.0,
             ):
        query_doc_pairs = [(query, doc.page_content) for doc in docs]
        scores = self.compute_score(
            query_doc_pairs,
            batch_size=batch_size,
            max_length=self.max_length,
        )
        # print("rank docs==================")
        # print(docs)
        # print("rank scores:===============")
        # print(scores)
        results = []
        for i in range(len(scores)):
            docs[i].metadata['rank_score'] = scores[i]
            # 根据 threshold 过滤低分文档
            if scores[i] >= threshold:
                if return_documents:
                    results.append({"corpus_id": i, "score": scores[i], "document": docs[i]})
                else:
                    results.append({"corpus_id": i, "score": scores[i]})
        # print("after rerank============================================")
        # print(docs)
        # print("after rerank============================================")
        # 如何没有满足条件的文档，则返回第一个文档
        if len(results) == 0:
            if return_documents:
                results.append({"corpus_id": 0, "score": scores[0], "document": docs[0]})
            else:
                results.append({"corpus_id": 0, "score": scores[0]})

        results = sorted(results, key=lambda x: x["score"], reverse=True)
        results_topk = results[:top_k]
        return results_topk



class QwenReranker(Reranker):
    """
    使用Qwen3-Reranker-0.6B模型进行文档重排序
    """
    def __init__(
            self,
            model_name_or_path: str = "Qwen/Qwen3-Reranker-0.6B",
            use_fp16: bool = False,
            max_length: int = 4096,
            instruction: str = None,
            device: str = "auto"
    ):

        if device == "auto":
            # 设置设备
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
                use_fp16 = False
        else:
            self.device = device   

        # 在 mps/cpu 上禁用 fp16，避免底层内核崩溃
        if self.device in ("mps", "cpu"):
            use_fp16 = False
        
        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side='left')
        
        # 优化模型加载配置以减少内存使用
        model_kwargs = {
            "device_map": self.device,  # 直接指定device参数
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,  # 减少CPU内存使用
            # "attn_implementation": "flash_attention_2" if torch.cuda.is_available() else None,  # 使用flash attention减少内存
        }
        if "cuda" in self.device and use_fp16:
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float32

        # 在加载模型前清理GPU内存
        if torch.cuda.is_available():
            import gc
            torch.cuda.empty_cache()
            gc.collect()
            
            # 检查加载前的内存状态
            allocated_before = torch.cuda.memory_allocated(self.device) / 1024**3
            print(f"加载模型前GPU内存 - 已分配: {allocated_before:.2f}GB")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs
        ).eval()
        
        # 模型加载后再次清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            allocated_after = torch.cuda.memory_allocated(self.device) / 1024**3
            print(f"加载模型后GPU内存 - 已分配: {allocated_after:.2f}GB (增加了 {allocated_after - allocated_before:.2f}GB)")
        
        # 设置token ID
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        
        # 设置最大长度
        self.max_length = max_length
        
        # 设置前缀和后缀
        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        
        # 设置指令
        self.instruction = instruction
        if self.instruction is None:
            self.instruction = '评估文档与查询的相关性，判断文档是否包含查询所需的关键信息，是否能够准确、完整地回答查询问题，并考虑文档的可靠性和信息价值'
            

    def format_instruction(self, query, doc):
        """格式化输入指令"""
        output = f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}"
        return output
        
    def process_inputs(self, pairs):
        """处理输入"""
        try:
            # 检查输入是否为空
            if not pairs:
                return None
                
            inputs = self.tokenizer(
                pairs, 
                padding=False, 
                truncation='longest_first',
                return_attention_mask=False, 
                max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
            )
            
            # 添加前缀和后缀tokens
            for i, ele in enumerate(inputs['input_ids']):
                inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
                
            # 填充到相同长度
            inputs = self.tokenizer.pad(
                inputs, 
                padding=True, 
                return_tensors="pt", 
                max_length=self.max_length
            )
            
            # 移动到指定设备
            for key in inputs:
                if hasattr(inputs[key], 'to'):
                    inputs[key] = inputs[key].to(self.device)
                    
            return inputs
            
        except Exception as e:
            print(f"处理输入时发生错误: {e}")
            return None
        
    @torch.no_grad()
    def compute_logits(self, inputs):
        """计算相关性分数"""
        try:
            # 确保输入在正确的设备上
            for key in inputs:
                if hasattr(inputs[key], 'to'):
                    inputs[key] = inputs[key].to(self.device)
            
            # 前向传播
            outputs = self.model(**inputs)
            batch_scores = outputs.logits[:, -1, :]
            
            # 提取true和false向量
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]
            
            # 计算分数
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().cpu().tolist()  # 移到CPU并转换为列表
            
            # 立即清理中间结果
            del outputs, batch_scores, true_vector, false_vector
            
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return scores
            
        except Exception as e:
            print(f"计算logits时发生错误: {e}")
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # 返回默认分数
            batch_size = inputs['input_ids'].size(0) if 'input_ids' in inputs else 1
            return [0.5] * batch_size
        
    def rank(self,
             query: str,
             docs: List[Document],
             top_k: int = None,
             batch_size: int = 8,
             return_documents: bool = True,
             threshold: float = 0.0):
        """
        对文档进行重排序
        
        Args:
            query: 查询文本
            docs: 待排序的文档列表
            top_k: 返回的文档数量
            batch_size: 批处理大小（默认为2，避免内存问题）
            return_documents: 是否返回文档对象
            threshold: 分数阈值，低于此值的文档将被过滤
            
        Returns:
            排序后的文档列表
        """
        if not docs:
            return []

        print(f"reranker_input_docs : {len(docs)} documents")    
        # 构建查询-文档对
        query_doc_pairs = [(query, doc.page_content) for doc in docs]
        
        # 计算相关性分数 - 减少批处理大小以避免内存问题
        all_scores = []

        if torch.cuda.is_available():
            import gc
            torch.cuda.empty_cache()
            gc.collect()
        
        for start_index in range(0, len(query_doc_pairs), batch_size):
            # 打印当前批次信息
            print(f"Processing batch {start_index // batch_size + 1} with {min(batch_size, len(query_doc_pairs) - start_index)} documents")

            batch_pairs = query_doc_pairs[start_index:start_index + batch_size]
            
            try:
                # 格式化输入
                formatted_pairs = [self.format_instruction(query, doc) for query, doc in batch_pairs]
                
                # 处理输入
                inputs = self.process_inputs(formatted_pairs)
                if inputs is None:
                    print(f"跳过批次 {start_index // batch_size + 1}：输入处理失败")
                    # 为失败的批次添加默认分数
                    all_scores.extend([0.5] * len(batch_pairs))
                    continue
                
                # 计算分数
                scores = self.compute_logits(inputs)
                all_scores.extend(scores)
                
                # 在每个批次后清理内存
                del inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    
            except Exception as e:
                print(f"处理批次 {start_index // batch_size + 1} 时发生错误: {e}")
                # 为失败的批次添加默认分数
                all_scores.extend([0.5] * len(batch_pairs))
                # 清理内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                continue
        
        # 打印所有文档的分数
        print(f"All documents scores: {all_scores}")

        # 构建结果列表
        results = []
        for i in range(len(all_scores)):
            docs[i].metadata['rank_score'] = all_scores[i]
            # 根据 threshold 过滤低分文档
            if all_scores[i] >= threshold:
                if return_documents:
                    results.append({"corpus_id": i, "score": all_scores[i], "document": docs[i]})
                else:
                    results.append({"corpus_id": i, "score": all_scores[i]})
        
        # 如果没有满足条件的文档，则返回第一个文档
        if len(results) == 0 and len(docs) > 0:
            if return_documents:
                results.append({"corpus_id": 0, "score": all_scores[0], "document": docs[0]})
            else:
                results.append({"corpus_id": 0, "score": all_scores[0]})
        
        # 按分数降序排序
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        
        # 截取前top_k个结果
        if top_k is not None:
            results = results[:top_k]
        return results
        
        
        
def test_qwen_reranker():
    """
    测试QwenReranker的功能
    """
    print("开始测试QwenReranker...")
    
    # 初始化QwenReranker
    print("初始化QwenReranker...")
    model_name_or_path = "/root/autodl-tmp/models/Qwen/Qwen3-Reranker-4B"
    reranker = QwenReranker(model_name_or_path=model_name_or_path, device="cuda:3", use_fp16=True)
    
    # 准备测试数据
    query = "中国的首都是哪里？"
    docs = [
        Document(page_content="中国的首都是北京。", metadata={"source": "wiki", "index": 0}),
        Document(page_content="重力是一种使两个物体相互吸引的力。", metadata={"source": "science", "index": 0}),
        Document(page_content="北京是中国的政治和文化中心。", metadata={"source": "wiki", "index": 1}),
        Document(page_content="上海是中国最大的城市。", metadata={"source": "wiki", "index": 2})
    ]
    
    # 执行重排序
    print("执行文档重排序...")
    results = reranker.rank(query=query, docs=docs, top_k=3)
    
    # 打印结果
    print("\n重排序结果:")
    for i, result in enumerate(results):
        print(f"排名 {i+1}:")
        print(f"  文档内容: {result['document'].page_content}")
        print(f"  相关性分数: {result['score']:.4f}")
        print(f"  来源: {result['document'].metadata.get('source')}")
        print(f"  索引: {result['document'].metadata.get('index')}")
        print()
    
    print("QwenReranker测试完成!")
    
    return results


if __name__ == "__main__":
    # 测试QwenReranker
    test_qwen_reranker()