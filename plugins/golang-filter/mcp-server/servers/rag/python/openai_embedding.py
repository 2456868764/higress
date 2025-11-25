import os
from typing import List

from openai._types import NOT_GIVEN
from langchain_core.embeddings import Embeddings
from openai import OpenAI

OPENAI_MODEL_DIM_MAP = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "qwen3-embedding-8b": 4096,
    "qwen3-embedding-4b": 2560,
    "qwen3-embedding-0.6b": 1024,
}

class OpenAIEmbedding(Embeddings):
    """
    OpenAI embedding model implementation.

    This class provides an interface to the OpenAI embedding API, which offers
    various embedding models for text processing.

    For more information, see:
    https://platform.openai.com/docs/guides/embeddings/use-cases
    """

    def __init__(self, model: str = "text-embedding-ada-002", **kwargs):
        """
        Initialize the OpenAI embedding model.

        Args:
            model (str): The model identifier to use for embeddings. Default is "text-embedding-ada-002".
            **kwargs: Additional keyword arguments.
                - api_key (str, optional): The OpenAI API key. If not provided,
                  it will be read from the OPENAI_API_KEY environment variable.
                - base_url (str, optional): The base URL for the OpenAI API. If not provided,
                  it will be read from the OPENAI_BASE_URL environment variable.
                - model_name (str, optional): Alternative way to specify the model.
                - dimension (int, optional): The dimension of the embedding vectors.
                  If not provided, the default dimension for the model will be used.
                - azure_endpoint (str, optional): If provided, use Azure OpenAI instead.
                - api_version (str, optional): Azure API version to use. Default is "2023-05-15".

        Notes:
            Available models:
                - 'text-embedding-ada-002': No dimension needed, default is 1536
                - 'text-embedding-3-small': dimensions from 512 to 1536, default is 1536
                - 'text-embedding-3-large': dimensions from 1024 to 3072, default is 3072
        """
        # Extract Azure-specific parameters
        api_version = kwargs.pop("api_version", "2023-05-15")
        self.is_qwen = model.startswith("qwen3-")
        # Extract standard parameters (keep original behavior)
        if "api_key" in kwargs:
            api_key = kwargs.pop("api_key")
        else:
            api_key = os.getenv("OPENAI_API_KEY")

        if "base_url" in kwargs:
            base_url = kwargs.pop("base_url")
        else:
            base_url = os.getenv("OPENAI_BASE_URL")

        if "model_name" in kwargs and (not model or model == "text-embedding-ada-002"):
            model = kwargs.pop("model_name")

        if "dimension" in kwargs:
            dimension = kwargs.pop("dimension")
        else:
            dimension = OPENAI_MODEL_DIM_MAP.get(model, 1536)

        print(f"OpenAI model: {model}, dimension: {dimension}")
        self.dim = dimension
        self.model = model
        # Initialize the appropriate client based on parameters
        self.client = OpenAI(api_key=api_key, base_url=base_url, **kwargs)

    def _get_dim(self):
        """
        Get the dimension parameter for the API call.

        Returns:
            int or NOT_GIVEN: The dimension to use for the embedding, or NOT_GIVEN
            if using text-embedding-ada-002 which doesn't support custom dimensions.
        """
        return self.dim if self.model != "text-embedding-ada-002" else NOT_GIVEN

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text (str): The query text to embed.

        Returns:
            List[float]: A list of floats representing the embedding vector.
        """
        if self.is_qwen:
            response = self.client.embeddings.create(
                input=[text], model=self.model
            )
        else:
            response = self.client.embeddings.create(
                input=[text], model=self.model, dimensions=self._get_dim()
            )

        return response.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of document texts.

        Args:
            texts (List[str]): A list of document texts to embed.

        Returns:
            List[List[float]]: A list of embedding vectors, one for each input text.
        """
        if self.is_qwen:
            response = self.client.embeddings.create(
                input=texts, model=self.model
            )
        else:
            response = self.client.embeddings.create(
                input=texts, model=self.model, dimensions=self._get_dim()
            )

        return [r.embedding for r in response.data]

    @property
    def dimension(self) -> int:
        """
        Get the dimensionality of the embeddings for the current model.

        Returns:
            int: The number of dimensions in the embedding vectors.
        """
        return self.dim

