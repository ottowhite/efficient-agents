import asyncio
import os
import random
import string
import time
from datasets import load_dataset
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr
from openai import OpenAI
from langchain_classic.storage import LocalFileStore
from langchain_classic.embeddings import CacheBackedEmbeddings

def test_embedding_latency(embedding_model, length=500):
	"""Generate a random string and embed it twice, timing each embedding."""
	random_string = ''.join(random.choices(string.ascii_letters, k=length))
	print(f"Random string (length {length}): {random_string[:50]}")
	
	# First embedding
	start = time.perf_counter()
	embedding1 = embedding_model.embed_query(random_string)
	latency1 = time.perf_counter() - start
	print(f"First embedding latency: {latency1*1000:.2f}ms")
	
	# Second embedding
	start = time.perf_counter()
	embedding2 = embedding_model.embed_query(random_string)
	latency2 = time.perf_counter() - start
	print(f"Second embedding latency: {latency2*1000:.2f}ms")
	
	return embedding1, embedding2, latency1, latency2


class EmbeddingModelWrapper:
	def __init__(self, embedding_model, client_type: str):
		self.embedding_model = embedding_model
		self.client_type = client_type
	
	def embed_query(self, query):
		if self.client_type == "openai":
			return self.embedding_model.create(
				model="intfloat/e5-base-v2",
				input=query,
			)
		elif self.client_type == "langchain":
			return self.embedding_model.embed_query(query)
		else:
			raise ValueError(f"Invalid client type: {self.client_type}")


async def main():
	# dev_dataset = load_dataset("RUC-NLPIR/FlashRAG_datasets", "hotpotqa", split="dev")
	# train_dataset = load_dataset("RUC-NLPIR/FlashRAG_datasets", "hotpotqa", split="train") 
	# print(f"dev_dataset: {len(dev_dataset)}")
	# print(f"train_dataset: {len(train_dataset)}")

	openai_client = OpenAI(
		base_url="http://localhost:30080/v1",
		api_key=""
	)

	langchain_embedding_model_raw = OpenAIEmbeddings(
		model="intfloat/e5-base-v2",
		base_url="http://localhost:30080/v1",
		api_key=SecretStr(""),
		# This also happens to skip tokenisation
		check_embedding_ctx_length=False
	)

	cache_dir = "./cache/"
	if not os.path.exists(cache_dir):
		os.makedirs(cache_dir)
	store = LocalFileStore(cache_dir)
	cached_embedding_model_langchain = CacheBackedEmbeddings.from_bytes_store(
		langchain_embedding_model_raw,
		document_embedding_cache=store,
		query_embedding_cache=store,
		namespace=langchain_embedding_model_raw.model
	)

	print(list(store.yield_keys()))

	embedding_model_openai = EmbeddingModelWrapper(openai_client.embeddings, "openai")
	embedding_model_langchain = EmbeddingModelWrapper(langchain_embedding_model_raw, "langchain")
	embedding_model_langchain_cached = EmbeddingModelWrapper(cached_embedding_model_langchain, "langchain")

	for i in range(10):
		test_embedding_latency(embedding_model_langchain_cached)




if __name__ == "__main__":
	asyncio.run(main())
