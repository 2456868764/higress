import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from util import rm_file, save_list_to_json
from tqdm import tqdm
from openai import OpenAI

prefix = "Below is a question followed by some context from different sources. Please answer the question based on the context. The answer to the question is a word or entity. If the provided information is insufficient to answer the question, respond 'Insufficient Information'. Answer directly without explanation."


def remove_think_content(text: str) -> str:
    """
    移除响应中的 think 内容
    支持多种格式：
    - <think>...</think>
    - <thinking>...</thinking>
    - 思考：... 或 Thinking: ...
    - [think]...[/think]
    """
    if not text:
        return text
    
    # 移除 <think>...</think> 标签（不区分大小写，支持多行）
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # 移除 [think]...[/think] 标签
    text = re.sub(r'\[think\].*?\[/think\]', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'\[thinking\].*?\[/thinking\]', '', text, flags=re.IGNORECASE | re.DOTALL)
    # 清理多余的空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def query_bot(
            client,
            model_name,
            messages,
            temperature=0.3,
            max_new_tokens=512,
            **kwargs,
    ):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": messages},
                ],
                temperature=temperature,
                max_tokens=max_new_tokens,
                **kwargs,
            )
            raw_response = response.choices[0].message.content.strip()
            # 移除 think 内容
            cleaned_response = remove_think_content(raw_response)
            return cleaned_response
        except Exception as e:
            print(f"OpenAI API error: {e}")
            raise


def process_single_item(item, client, model_name, temperature, max_tokens, prefix):
    """
    处理单个数据项，调用 LLM 生成答案
    
    Args:
        item: 单个数据项字典
        client: OpenAI 客户端
        model_name: 模型名称
        temperature: 温度参数
        max_tokens: 最大 token 数
        prefix: 提示词前缀
    
    Returns:
        处理后的结果字典
    """
    try:
        retrieval_list = item['retrieval_list']
        context = '--------------'.join(e['text'] for e in retrieval_list)
        # retrieval_list = item['evidence_list']
        # context = '--------------'.join(e['fact'] for e in retrieval_list)
        prompt = f"{prefix}\n\nQuestion:{item['query']}\n\nContext:\n\n{context}"
        response = query_bot(client, model_name, prompt, 
                           temperature=temperature, 
                           max_new_tokens=max_tokens)
        
        result = {
            'query': item['query'],
            'model_answer': response,
            'gold_answer': item['answer'],
            'question_type': item['question_type']
        }
        return result
    except Exception as e:
        print(f"Error processing query '{item.get('query', 'unknown')}': {e}")
        # 返回错误结果，避免整个流程中断
        return {
            'query': item.get('query', ''),
            'model_answer': f'Error: {str(e)}',
            'gold_answer': item.get('answer', ''),
            'question_type': item.get('question_type', 'unknown')
        }

def main():
    parser = argparse.ArgumentParser(description='QA evaluation using OpenAI compatible API')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input JSON file path (e.g., toy_data/step1_data.json)')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output JSON file path (e.g., qa_output/llama.json)')
    parser.add_argument('--api-key', type=str, default=None,
                        help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--base-url', type=str, default=None,
                        help='OpenAI compatible API base URL (or set OPENAI_BASE_URL env var)')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name (or set OPENAI_MODEL env var, default: gpt-4o)')
    parser.add_argument('--temperature', type=float, default=0.3,
                        help='Temperature for generation (default: 0.3)')
    parser.add_argument('--max-tokens', type=int, default=1024,
                        help='Maximum tokens to generate (default: 1024)')
    parser.add_argument('--max-workers', type=int, default=4,
                        help='Maximum number of concurrent workers (default: 10)')
    
    args = parser.parse_args()
    
    # Configuration: can be set via command line arguments or environment variables
    api_key = args.api_key or os.getenv("OPENAI_API_KEY","sk-0d9dd773c0e24c169b113d10f46656ca")
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL","https://dashscope.aliyuncs.com/compatible-mode/v1")
    model_name = args.model or os.getenv("OPENAI_MODEL", "qwen-plus")

    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # Load input data
    with open(args.input, 'r') as file:
        doc_data = json.load(file)
    
    # Process data with concurrent execution
    rm_file(args.output)
    save_list = []
    
    # 使用线程池并发处理
    # with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
    #     # 提交所有任务
    #     future_to_item = {
    #         executor.submit(
    #             process_single_item, 
    #             item, 
    #             client, 
    #             model_name, 
    #             args.temperature, 
    #             args.max_tokens,
    #             prefix
    #         ): item for item in doc_data
    #     }
        
    #     # 使用 tqdm 显示进度
    #     with tqdm(total=len(doc_data), desc="Processing") as pbar:
    #         # 按完成顺序收集结果
    #         for future in as_completed(future_to_item):
    #             result = future.result()
    #             save_list.append(result)
    #             pbar.update(1)
    
    # 保持原始顺序（如果需要的话）
    # 如果需要保持顺序，可以使用以下代码：
    save_list = [None] * len(doc_data)
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_index = {
            executor.submit(
                process_single_item, 
                item, 
                client, 
                model_name, 
                args.temperature, 
                args.max_tokens,
                prefix
            ): idx for idx, item in enumerate(doc_data)
        }
        with tqdm(total=len(doc_data), desc="Processing") as pbar:
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                result = future.result()
                save_list[idx] = result
                pbar.update(1)
    
    save_list_to_json(save_list, args.output)
    print(f"Results saved to {args.output}")
    print(f"Processed {len(save_list)} items with {args.max_workers} concurrent workers")


if __name__ == "__main__":
    main()

