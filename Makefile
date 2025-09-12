PHONY: jinja compose/up

jinja:
	jinja2 compose.yaml.j2 vars.yaml -o compose.yaml

agent-server/build:
	docker buildx build -f docker/Dockerfile.api . -t agent-server

docker/build/ncu_vllm:
	docker buildx build -f docker/Dockerfile.ncu_vllm . -t ncu_vllm

docker/run/ncu_vllm:
	docker run --privileged --env-file .env --gpus device=0 -p 9999:8000 -p 2222:22 --volume $(HOME)/.cache/huggingface:/root/.cache/huggingface -t ncu_vllm:latest tail -f 2>&1 > /dev/null &

docker/broadcast/ncu_vllm:
	docker ps --filter ancestor=ncu_vllm:latest --format '{{.ID}}' | xargs -P 0 -I{} docker exec -t {} $(COMMAND)

nsys/start:
	COMMAND="nsys start --gpu-metrics-devices=0 --output=report.nsys-rep --force-overwrite=true" $(MAKE) docker/broadcast/ncu_vllm

nsys/stop:
	COMMAND="nsys stop" $(MAKE) docker/broadcast/ncu_vllm
	docker cp $(shell docker ps --filter "name=llama1b-llm" --format "{{.ID}}"):/vllm-workspace/report.nsys-rep data/nsys-reports/llama1b-llm.nsys-rep
	docker cp $(shell docker ps --filter "name=llama8b-prm" --format "{{.ID}}"):/vllm-workspace/report.nsys-rep data/nsys-reports/llama8b-prm.nsys-rep

compose/up: jinja
	docker compose up -d --build --remove-orphans

work-dispatcher/run: compose/up
	sleep 2
	$(eval BASE_SERVER_PORT=$(shell yq '.base_port' vars.yaml))
	$(eval NUM_REPLICAS=$(shell yq '.replicas' vars.yaml))
	$(eval NUM_PROBLEMS=$(shell yq '.num_problems' vars.yaml))
	$(eval CONCURRENT_PROBLEMS=$(shell yq '.concurrent_problems' vars.yaml))
	BASE_SERVER_PORT=$(BASE_SERVER_PORT) NUM_REPLICAS=$(NUM_REPLICAS) NUM_PROBLEMS=$(NUM_PROBLEMS) CONCURRENT_PROBLEMS=$(CONCURRENT_PROBLEMS) python3 -m src.examples.beam_search.work_dispatcher

work-dispatcher/nsys-run: compose/up
	$(eval BASE_SERVER_PORT=$(shell yq '.base_port' vars.yaml))
	$(eval NUM_REPLICAS=$(shell yq '.replicas' vars.yaml))
	$(eval NUM_PROBLEMS=$(shell yq '.num_problems' vars.yaml))
	$(eval CONCURRENT_PROBLEMS=$(shell yq '.concurrent_problems' vars.yaml))

	$(MAKE) nsys/start
	BASE_SERVER_PORT=$(BASE_SERVER_PORT) NUM_REPLICAS=$(NUM_REPLICAS) NUM_PROBLEMS=$(NUM_PROBLEMS) CONCURRENT_PROBLEMS=$(CONCURRENT_PROBLEMS) python3 -m src.examples.beam_search.work_dispatcher
	$(MAKE) nsys/stop
