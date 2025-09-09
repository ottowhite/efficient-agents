PHONY: jinja compose/up

jinja:
	jinja2 compose.yaml.j2 vars.yaml -o compose.yaml

agent-server/build:
	docker buildx build -f docker/Dockerfile.api . -t agent-server

compose/up: jinja
	docker compose up -d --build --remove-orphans

work-dispatcher/run: compose/up
	sleep 2
	$(eval BASE_SERVER_PORT=$(shell yq '.base_port' vars.yaml))
	$(eval NUM_REPLICAS=$(shell yq '.replicas' vars.yaml))
	$(eval NUM_PROBLEMS=$(shell yq '.num_problems' vars.yaml))
	$(eval CONCURRENT_PROBLEMS=$(shell yq '.concurrent_problems' vars.yaml))
	BASE_SERVER_PORT=$(BASE_SERVER_PORT) NUM_REPLICAS=$(NUM_REPLICAS) NUM_PROBLEMS=$(NUM_PROBLEMS) CONCURRENT_PROBLEMS=$(CONCURRENT_PROBLEMS) python3 -m src.examples.beam_search.work_dispatcher