# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

from typing import List
import torch
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from utils import logger

class LocalEmbeddings(Embeddings):

    def __init__(self,
                 model_name_or_path: str,
                 model_engine: str = "huggingface",
                 dim: int = 768,
                 device: str = "cpu"):
        self.model_name_or_path = model_name_or_path
        self.model_engine = model_engine
        self.dim = dim
        self.device = device

        self._init_embedding_model()

    def _init_embedding_model(self):
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        model_kwargs = {"device": self.device}
        logger.info(f"Using {self.model_engine} as model engine to load embeddings")
        if self.model_engine == "qwen":
            print(f"Using device {self.device} to load Qwen embedding model")
            # Load Qwen embedding model using SentenceTransformer
            model_kwargs = {}
            # if torch.cuda.is_available():
            #     model_kwargs["attn_implementation"] = "flash_attention_2"
            tokenizer_kwargs = {"padding_side": "left"}
            self.st_model = SentenceTransformer(
                self.model_name_or_path,
                model_kwargs=model_kwargs,
                device = self.device,
                tokenizer_kwargs=tokenizer_kwargs,
            )
            # self.st_model.to(self.device)
            logger.info(f"Loaded Qwen embedding model from {self.model_name_or_path}")
        else:
            pass

    def embed_documents(self, docs: List[str]):
        # 获取嵌入表示并根据 self.dim 限制维度
        if self.model_engine == "qwen":
            embeddings = self.st_model.encode(docs)
            return [embedding[:self.dim] for embedding in embeddings]
        else:
            embeddings = self.embeddings.embed_documents(docs)
            return [embedding[:self.dim] for embedding in embeddings]

    def embed_query(self, query: str):
        # 获取查询嵌入并根据 self.dim 限制维度
        if self.model_engine == "qwen":
            # Use the "query" prompt for queries as recommended in the demo
            embedding = self.st_model.encode(query, prompt_name="query")
            return embedding[:self.dim]
        else:
            embedding = self.embeddings.embed_query(query)
            return embedding[:self.dim]

