SHELL := /usr/bin/env bash
PORT ?= 8400

.DEFAULT_GOAL := help
.PHONY: help
help: $(MAKEFILE_LIST)
	@sed -n 's/^##//p' $(MAKEFILE_LIST)

## install: create uv environment and add requirements from pyproject.toml
.PHONY: install
install:
	uv pip install .

MODEL_DIR ?= ./models
# anemll-Meta-Llama-3.2-1B-ctx2048_0.1.2
MODELS ?= anemll-DeepSeekR1-8B-ctx1024_0.2.0 \
	anemll-dwq-llama-3.2-1B-4b-pf6b-ctx1024_0.3.0
	

## model: get modes from Huggingface face
.PHONY: models
models:
	for model in $(MODELS); do \
		huggingface-cli download anemll/$$model --local-dir $(MODEL_DIR)/$$model && \
		unzip -n models/$$model/\*.zip -d models/$$model; \
	done

## run: run the server expecting $MODEL_DIR defaults to ./models has downloaded models
.PHONY: run
run:
	if ! lsof -i:$(PORT) -sTCP:LISTEN; then \
		PORT=$(PORT) python anemll-server.py; \
	fi

## chat: run simple chat to test a running server
.PHONY: chat
chat:
	./chat_full.py

## test-segfault: run segmentation fault trigger tests (requires running server)
.PHONY: test-segfault
test-segfault:
	python trigger_segfault_test.py

## test-recovery: run comprehensive segfault recovery tests (starts/stops server)
.PHONY: test-recovery
test-recovery:
	python test_segfault_recovery.py

## test-enhanced: run enhanced segfault tests with CoreML error classification
.PHONY: test-enhanced
test-enhanced:
	python enhanced_segfault_tests.py

## test-classifier: run CoreML error classifier demo
.PHONY: test-classifier
test-classifier:
	python coreml_classifier_demo.py

## test-all: run all dynamic loading and segfault tests
.PHONY: test-all
test-all:
	python test_dynamic_loading.py
	python test_segfault_recovery.py

## test-all-enhanced: run all tests including enhanced CoreML error classification
.PHONY: test-all-enhanced
test-all-enhanced:
	python test_dynamic_loading.py
	python test_segfault_recovery.py
	python enhanced_segfault_tests.py
