from dataclasses import dataclass

class AttentionBlockConfig:
	sliding_window_size: int | None

	grouped_query_attention: bool
	num_kv_heads: int | None

	num_attention_heads: int
	attention_head_dim: int

	sequence_length: int

	def __init__(self,
		sliding_window_size: int | None,
		grouped_query_attention: bool,
		num_kv_heads: int | None,
		num_attention_heads: int,
		attention_head_dim: int
	):
		self.sliding_window_size = sliding_window_size
		self.grouped_query_attention = grouped_query_attention
		self.num_kv_heads = num_kv_heads
		self.num_attention_heads = num_attention_heads
		self.attention_head_dim = attention_head_dim
	
	def get_kv_size(self, sequence_length: int) -> int:
		
		if self.grouped_query_attention:
			num_kv_caches = self.num_kv_heads
		else:
			num_kv_caches = self.num_attention_heads
		assert num_kv_caches is not None

		if self.sliding_window_size is not None:
			effective_sequence_length = min(sequence_length, self.sliding_window_size)
		else:
			effective_sequence_length = sequence_length

		return self.attention_head_dim * effective_sequence_length * 2 * num_kv_caches

class ModelConfig:
	attention_blocks: list[AttentionBlockConfig]

	def __init__(self,
		attention_blocks: list[AttentionBlockConfig]
	):
		self.attention_blocks = attention_blocks
	
	def get_kv_size(self, sequence_length: int) -> int:
		return sum(block.get_kv_size(sequence_length) for block in self.attention_blocks)

def main():
	num_sliding_layers = 18
	num_full_attention_layers = 18
	kv_heads_per_layer = 8
	num_attention_heads = 64
	attention_head_dim = 64
	sliding_window_size = 128
	sequence_length = 1_000_000
	bytes_per_float = 0.5
	
	sliding_attention_blocks = [
		AttentionBlockConfig(
			sliding_window_size=sliding_window_size,
			grouped_query_attention=True,
			num_kv_heads=kv_heads_per_layer,
			num_attention_heads=num_attention_heads,
			attention_head_dim=attention_head_dim
		)
	] * num_sliding_layers 

	full_attention_blocks = [
		AttentionBlockConfig(
			sliding_window_size=None,
			grouped_query_attention=True,
			num_kv_heads=kv_heads_per_layer,
			num_attention_heads=num_attention_heads,
			attention_head_dim=attention_head_dim
		)
	] * num_full_attention_layers

	attention_blocks = sliding_attention_blocks + full_attention_blocks

	model_config = ModelConfig(attention_blocks=attention_blocks)

	kv_size_floats = model_config.get_kv_size(sequence_length)
	kv_size_gb = kv_size_floats * bytes_per_float / 1024 / 1024 / 1024
	print(f"KV size for {sequence_length} tokens: {kv_size_gb:.3f} GB")

if __name__ == "__main__":
	main()