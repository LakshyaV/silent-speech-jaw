.PHONY: install lint format test check firmware config

install:
	pip install -e ".[all]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

test:
	pytest

check: lint test

# Compile the streamer for the Arduino UNO Q with pinned core and library versions.
firmware:
	arduino-cli compile --profile unoq --warnings all firmware/ssj_streamer

# Regenerate configs/default.yaml after changing a configuration dataclass.
config:
	python tools/write_default_config.py
