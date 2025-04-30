SHELL := /usr/bin/env bash
PORT ?= 8400

.DEFAULT_GOAL := help
.PHONY: help
help: $(MAKEFILE_LIST)
	@sed -n 's/^##//p' $(MAKEFILE_LIST)

## install: create uv environment and add requirements
.PHONY: install
install:
	uv pip install -r requirements.txt
	huggingface-cli 

## model: get modes from Huggingface face
.PHONY: models
models:
	for model in anemll-DeepSeekR1-8B-ctx1024_0.1.1 anemll-Meta-Llama-3.2-1B-ctx2048_0.1.2; do \
		huggingface-cli download anemll/$$model --local-dir models/$$model && \
		unzip -n models/$$model/\*.zip -d models/$$model; \
	done

## run the server expecting $MODEL_DIR to be set
.PHONY: server
server:
	if ! lsof -i:$(PORT) -sTCP:LISTEN; then \
		PORT=$(PORT) python server.py; \
	fi
