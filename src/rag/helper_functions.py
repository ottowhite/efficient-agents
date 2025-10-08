from langchain_community.document_loaders import PyPDFLoader
from langchain.storage import LocalFileStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import InMemoryByteStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from openai import RateLimitError
from typing import List
from rank_bm25 import BM25Okapi
import fitz
import asyncio
import random
import textwrap
import numpy as np
from enum import Enum
import sys
import time

#store = LocalFileStore("/tmp/rag_cache/")
store = LocalFileStore("./cache/")

def replace_t_with_space(list_of_documents):
    """
    Replaces all tab characters ('\t') with spaces in the page content of each document

    Args:
        list_of_documents: A list of document objects, each with a 'page_content' attribute.

    Returns:}
        The modified list of documents with tab characters replaced by spaces.
    """

    for doc in list_of_documents:
        doc.page_content = doc.page_content.replace('\t', ' ')  # Replace tabs with spaces
    return list_of_documents

async def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    """
    Encodes a PDF book into a vector store using embeddings.

    Args:
        path: The path to the PDF file.
        chunk_size: The desired size of each text chunk.
        chunk_overlap: The amount of overlap between consecutive chunks.

    Returns:
        A FAISS vector store containing the encoded book content.
    """

    # Load PDF documents
    loader = PyPDFLoader(path)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    # Create embeddings and vector store
    #embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings()

    start_time = time.time()
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)
    end_time = time.time() - start_time
    #print(list(store.yield_keys()))
    print("Creating embedding time:", end_time)

    return vectorstore


async def retrieve_context_per_question(question, chunks_query_retriever):
    """
    Retrieves relevant context and unique URLs for a given question using the chunks query retriever.

    Args:
        question: The question for which to retrieve context and URLs.

    Returns:
        A tuple containing:
        - A string with the concatenated content of relevant documents.
        - A list of unique URLs from the metadata of the relevant documents.
    """

    # Retrieve relevant documents for the given question
    #docs = chunks_query_retriever.get_relevant_documents(question)
    docs = chunks_query_retriever.invoke(question)

    # Concatenate document content
    # context = " ".join(doc.page_content for doc in docs)
    context = [doc.page_content for doc in docs]

    return context


def show_context(context):
    """
    Display the contents of the provided context list.

    Args:
        context (list): A list of context items to be displayed.

    Prints each context item in the list with a heading indicating its position.
    """
    for i, c in enumerate(context):
        print(f"Context {i + 1}:")
        print(c)
        print("\n")

# Enum class representing different embedding providers
class EmbeddingProvider(Enum):
    OPENAI = "openai"
    COHERE = "cohere"
    AMAZON_BEDROCK = "bedrock"
    HUGGINGFACE = "huggingface"

# Enum class representing different model providers
class ModelProvider(Enum):
    OPENAI = "openai"
    GROQ = "groq"
    ANTHROPIC = "anthropic"
    AMAZON_BEDROCK = "bedrock"
    HUGGING_FACE="huggingface"


def get_langchain_embedding_provider(provider: EmbeddingProvider, model_id: str = None):
    """
    Returns an embedding provider based on the specified provider and model ID.

    Args:
        provider (EmbeddingProvider): The embedding provider to use.
        model_id (str): Optional -  The specific embeddings model ID to use .

    Returns:
        A LangChain embedding provider instance.

    Raises:
        ValueError: If the specified provider is not supported.
    """
    if provider == EmbeddingProvider.OPENAI:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings()
    elif provider == EmbeddingProvider.COHERE:
        from langchain_cohere import CohereEmbeddings
        return CohereEmbeddings()
    elif provider == EmbeddingProvider.AMAZON_BEDROCK:
        from langchain_community.embeddings import BedrockEmbeddings
        return BedrockEmbeddings(model_id=model_id) if model_id else BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
    elif provider == EmbeddingProvider.HUGGINGFACE:
        embedding = HuggingFaceEmbeddings()
        #store = InMemoryByteStore()

        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
                embedding, store, namespace="haggingface",
                query_embedding_cache=True)
        return cached_embedder 
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")
