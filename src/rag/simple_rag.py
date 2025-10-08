import sys
import asyncio
import argparse
import time

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
    start_time = time.time()
    chunks_vector_store = await encode_pdf(path, chunk_size=1000, chunk_overlap=200)
    encode_time = time.time() - start_time
    chunks_query_retriever = chunks_vector_store.as_retriever(search_kwargs={"k": 2})


    start_time = time.time()
    context = await retrieve_context_per_question(query, chunks_query_retriever)
    ret_time = time.time() - start_time
    show_context(context)

    print("encode pdf time", encode_time)
    print("retrieve context time", ret_time)

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
