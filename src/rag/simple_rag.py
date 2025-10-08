import sys
import asyncio
import argparse

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from helper_functions import (EmbeddingProvider,
                              retrieve_context_per_question,
                              replace_t_with_space,
                              get_langchain_embedding_provider,
                              show_context, 
                              encode_pdf)

from langchain_community.vectorstores import FAISS

async def main(args):
    path = args.path
    query = args.query
    chunks_vector_store = await encode_pdf(path, chunk_size=1000, chunk_overlap=200)
    chunks_query_retriever = chunks_vector_store.as_retriever(search_kwargs={"k": 2})
    context = await retrieve_context_per_question(query, chunks_query_retriever)
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
