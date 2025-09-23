import itertools
import os
import shutil

class ProgramConfig:
	def __init__(self, config_json: dict, experiment_name: str | None = None):
		self.command = config_json["command"]
		self.args = config_json["args"]
		
		if experiment_name is not None:
			assert not self.is_grid(), "Config cannot be a grid if it has an experiment name"
			self.experiment_name = experiment_name

	
	def is_grid(self) -> bool:
		for key, value in self.args.items():
			if isinstance(value, list):
				return True
		return False
	
	def generate_configs_from_grid(self) -> list["ProgramConfig"]:
		assert self.is_grid(), "Config must be a grid to generate configs from grid"
		
		# Separate list values from non-list values
		keys_with_multi_values = []
		value_lists = []
		non_list_args = {}
		
		for key, value in self.args.items():
			if isinstance(value, list):
				keys_with_multi_values.append(key)
				value_lists.append(value)
			else:
				non_list_args[key] = value
		
		# Generate all combinations of list values
		combinations = itertools.product(*value_lists)
		
		# Create Config objects for each combination
		configs = []
		for combo in combinations:
			# Create new args dict with current combination
			new_args = non_list_args.copy()
			experiment_name = "exp"
			for key, value in zip(keys_with_multi_values, combo):
				experiment_name += f"_{key}={value}"
				new_args[key] = value
			
			# Create new config with same command but updated args
			new_config_json = {
				"command": self.command,
				"args": new_args
			}
			configs.append(ProgramConfig(new_config_json, experiment_name))
		
		return configs
	
	def args_to_string(self) -> str:
		arg_prefix = "--"
		arg_key_value_separator = " "
		arg_list_separator = " "

		arg_string = ""
		for key, value in self.args.items():
			arg_string += f"{arg_prefix}{key}{arg_key_value_separator}{value}{arg_list_separator}"

		return arg_string

class ExperimentConfig:
	def __init__(self, experiment_name: str, run_nsys: bool, result_path: str, artifact_retrievals: dict[str, str]):
		self.experiment_name = experiment_name
		self.run_nsys = run_nsys
		self.result_path = result_path
		self.artifact_retrievals = artifact_retrievals

		for _, destination_dir in self.artifact_retrievals.items():
			os.makedirs(destination_dir, exist_ok=True)

class Experiment:
	def __init__(self, config: ProgramConfig, experiment_config: ExperimentConfig):
		assert not config.is_grid(), "Config must not be a grid"
		self.config = config
		self.experiment_config = experiment_config
	
	def start(self):
		command = self.config.command
		full_command = f"{command} {self.config.args_to_string()}"

		if self.experiment_config.run_nsys:
			os.system("make nsys/start")

		print("Running command: ", full_command)
		os.system(full_command)

		if self.experiment_config.run_nsys:
			os.system("make nsys/stop")
		
		self.retrieve_artifacts()

		print(f"Save to {self.config.experiment_name}.txt")
		pass

	def retrieve_artifacts(self):
		for artifact_path, destination_dir in self.experiment_config.artifact_retrievals.items():
			if not os.path.exists(artifact_path):
				print("Warning: Artifact path does not exist: ", artifact_path)
				continue

			src_filename = os.path.basename(artifact_path)
			destination = f"{destination_dir}/{self.config.experiment_name}_{src_filename}"

			shutil.copy(artifact_path, destination)


# Test the implementation
if __name__ == "__main__":
	grid_config = ProgramConfig(
		config_json={
			"command": "python3 -m src.experiments.vllm_trigger",
			"args": {
				"batch_size": [1, 10, 50, 100],
				"input_length": 0,
				"output_length": [5000],
				"num_trials": 1,
				"results_path": "results.csv",
			}
		})
	program_configs = grid_config.generate_configs_from_grid()
	
	experiment_config = ExperimentConfig(
		experiment_name="beam_search",
		run_nsys=False,
		result_path="data/nsys-reports",
		artifact_retrievals={
			"llama1b-llm.nsys-rep": "data/nsys-reports",
			"llama8b-prm.nsys-rep": "data/nsys-reports",
			"results.csv": "data/latencies"
		}
	)

	for cfg in program_configs:
		experiment = Experiment(cfg, experiment_config)
		experiment.start()
