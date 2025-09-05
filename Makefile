jinja:
	jinja2 compose.yaml.j2 vars.yaml -o compose.yaml

agent-server/build:
	docker buildx build -f docker/Dockerfile.api . -t agent-server

compose/up:
	docker compose up -d --build