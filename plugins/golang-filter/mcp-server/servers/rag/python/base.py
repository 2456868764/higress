
from abc import ABC, abstractmethod


class CorpusFile(ABC):
    """
        "title": "200+ of the best deals from Amazon's Cyber Monday sale",
        "author": null,
        "source": "Mashable",
        "published_at": "2023-11-27T08:45:59+00:00",
        "category": "entertainment",
        "url": "https://mashable.com/article/cyber-monday-deals-amazon-2023",
    """
    metadata: dict
    

class VectorStoreBase(ABC):
    """base class for vector store implementations"""

    @abstractmethod
    def create_vectorstore(self):
        pass

    @abstractmethod
    def drop_vectorstore(self):
        pass

    @abstractmethod
    def clear_vectorstore(self):
        pass

    @abstractmethod
    def add_doc(self, file, docs):
        pass

    @abstractmethod
    def delete_doc(self, filename):
        pass

    @abstractmethod
    def exist_doc(self, filename):
        pass

    @abstractmethod
    def update_doc(self, file, docs):
        pass

    @abstractmethod
    def search_docs(self, text, top_k, threshold, **kwargs):
        pass


    @abstractmethod
    def search_hybrid_docs(self, text, top_k, threshold, **kwargs):
        pass


