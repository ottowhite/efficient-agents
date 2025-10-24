from enum import Enum


class Unit(Enum):
	FLOATS = "floats"
	BYTES = "bytes"
	KB = "KB"
	MB = "MB"
	GB = "GB"
	
	@property
	def bytes(self) -> float:
		"""Returns the number of bytes for this unit (1 for FLOATS as base unit)"""
		return {
			Unit.FLOATS: None,
			Unit.BYTES: 1,
			Unit.KB: 1024,
			Unit.MB: 1024 ** 2,
			Unit.GB: 1024 ** 3,
		}[self]


class UnitConverter:
	def __init__(self, bytes_per_float: float):
		self.bytes_per_float = bytes_per_float
	
	def get(self, num_floats: float, unit: Unit) -> float:
		if unit == Unit.FLOATS:
			return num_floats
		elif unit == Unit.BYTES:
			return num_floats * self.bytes_per_float
		elif unit == Unit.KB:
			return num_floats * self.bytes_per_float / 1024
		elif unit == Unit.MB:
			return num_floats * self.bytes_per_float / 1024 / 1024
		elif unit == Unit.GB:
			return num_floats * self.bytes_per_float / 1024 / 1024 / 1024
		else:
			raise ValueError(f"Unknown unit: {unit}")

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
		attention_head_dim: int,
		bytes_per_float: float
	):
		self.sliding_window_size = sliding_window_size
		self.grouped_query_attention = grouped_query_attention
		self.num_kv_heads = num_kv_heads
		self.num_attention_heads = num_attention_heads
		self.attention_head_dim = attention_head_dim
		self.converter = UnitConverter(bytes_per_float)
	
	def get_kv_size(self, sequence_length: int, unit: Unit) -> float:
		if self.grouped_query_attention:
			num_kv_caches = self.num_kv_heads
		else:
			num_kv_caches = self.num_attention_heads
		assert num_kv_caches is not None

		if self.sliding_window_size is not None:
			effective_sequence_length = min(sequence_length, self.sliding_window_size)
		else:
			effective_sequence_length = sequence_length

		num_floats = self.attention_head_dim * effective_sequence_length * 2 * num_kv_caches
		return self.converter.get(num_floats, unit)

class LayerConfig:
	attention_block: AttentionBlockConfig
	num_params: int

	def __init__(self, attention_block: AttentionBlockConfig, num_params: int):
		self.attention_block = attention_block
		self.num_params = num_params
	
	def get_kv_size(self, sequence_length: int, unit: Unit) -> float:
		return self.attention_block.get_kv_size(sequence_length, unit)
	
	def get_params_size(self, unit: Unit) -> float:
		return self.attention_block.converter.get(self.num_params, unit)

class ModelConfig:
	layers: list[LayerConfig]

	def __init__(self, layers: list[LayerConfig]):
		self.layers = layers
	
	def get_kv_size(self, sequence_length: int, unit: Unit) -> float:
		return sum(layer.get_kv_size(sequence_length, unit) for layer in self.layers)
	
	def get_params_size(self, unit: Unit) -> float:
		return sum(layer.get_params_size(unit) for layer in self.layers)

def print_transfer_times(name: str, gb_size: float, bandwidth_gbps: float) -> None:
	transfer_time_ms = (gb_size / bandwidth_gbps) * 1000
	print(f"{name} transfer time: {transfer_time_ms:.3f} ms")

def print_all_transfer_times(gb_size: float) -> None:
	print_transfer_times("PCIe 3.0", gb_size, 16)
	print_transfer_times("PCIe 4.0", gb_size, 32)
	print_transfer_times("NVLink 4", gb_size, 900)
	print_transfer_times("NVLink 5", gb_size, 1800)
	print_transfer_times("H100 Memory Bandwidth", gb_size, 3350)

def create_gpt_oss():
	num_sliding_layers = 18
	num_full_attention_layers = 18
	kv_heads_per_layer = 8
	num_attention_heads = 64
	attention_head_dim = 64
	sliding_window_size = 128
	bytes_per_float = 0.5
	num_params_per_layer = int(120_000_000_000 / (num_sliding_layers + num_full_attention_layers))
	
	sliding_layers = [
		LayerConfig(
			attention_block=AttentionBlockConfig(
				sliding_window_size=sliding_window_size,
				grouped_query_attention=True,
				num_kv_heads=kv_heads_per_layer,
				num_attention_heads=num_attention_heads,
				attention_head_dim=attention_head_dim,
				bytes_per_float=bytes_per_float
			),
			num_params=num_params_per_layer
		)
	] * num_sliding_layers 

	full_attention_layers = [
		LayerConfig(
			attention_block=AttentionBlockConfig(
				sliding_window_size=None,
				grouped_query_attention=True,
				num_kv_heads=kv_heads_per_layer,
				num_attention_heads=num_attention_heads,
				attention_head_dim=attention_head_dim,
				bytes_per_float=bytes_per_float
			),
			num_params=num_params_per_layer
		)
	] * num_full_attention_layers

	layers = sliding_layers + full_attention_layers

	return ModelConfig(layers=layers)

def main():
	gpt_oss = create_gpt_oss()	
	sequence_length = 1_000_000

	h100_memory_capacity_gb = 80

	kv_size_gb = gpt_oss.get_kv_size(sequence_length, Unit.GB)
	params_size_gb = gpt_oss.get_params_size(Unit.GB)

	h100_memory_remaining_gb = h100_memory_capacity_gb - params_size_gb
	h100_num_tokens = h100_memory_remaining_gb / gpt_oss.get_kv_size(1, Unit.GB)
	print(f"GPT-OSS 120B parameters size: {params_size_gb:.3f} GB")
	print(f"H100 memory capacity: {h100_memory_capacity_gb:.3f} GB")
	print(f"H100 memory remaining: {h100_memory_remaining_gb:.3f} GB")
	print(f"H100 number of tokens: {h100_num_tokens:.3f}")

	print(f"KV size for {sequence_length} tokens: {kv_size_gb:.3f} GB")
	print_all_transfer_times(kv_size_gb)

if __name__ == "__main__":
	main()