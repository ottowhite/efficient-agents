import sys
import asyncio
import argparse

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from helper_functions import (EmbeddingProvider,
                              retrieve_context_per_question,
                              replace_t_with_space,
                              get_langchain_embedding_provider,
                              show_context)

from langchain_community.vectorstores import FAISS

async def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    """
    Encodes a PDF book into a vector store using OpenAI embeddings.

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

    # Create embeddings (Tested with OpenAI and Amazon Bedrock)
    embeddings = get_langchain_embedding_provider(EmbeddingProvider.HUGGINGFACE)
    #embeddings = get_langchain_embedding_provider(EmbeddingProvider.AMAZON_BEDROCK)

    # Create vector store
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    return vectorstore

async def main(args):
    path = args.path
    query = args.query
    chunks_vector_store = await encode_pdf(path, chunk_size=1000, chunk_overlap=200)
    chunks_query_retriever = chunks_vector_store.as_retriever(search_kwargs={"k": 2})
    context = retrieve_context_per_question(query, chunks_query_retriever)
    show_context(context)

def parse_args():
    parser = argparse.ArgumentParser(description="Encode a PDF document and test a simple RAG.")
    parser.add_argument("--path", type=str, default="data/Understanding_Climate_Change.pdf",
                        help="Path to the PDF file to encode.")
    parser.add_argument("--query", type=str, default="What is the main cause of climate change?",
                        help="Query to test the retriever (default: 'What is the main cause of climate change?').")

    # Parse and validate arguments
    return parser.parse_args()

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()

    args = parse_args()
    asyncio.run(main(args))
