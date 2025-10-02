CONFIG = $(CURDIR)/kompose

default: status

status:
	kubectl get ns

up:
	kubectl apply -f $(CONFIG)

down:
	kubectl delete -f $(CONFIG)

config:
	mkdir -p $(CONFIG) && cd $(CONFIG) && kompose convert -f ../../compose.yaml
