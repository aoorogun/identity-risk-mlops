.PHONY: all test lint
all:
	prembly-risk generate
	prembly-risk train
	prembly-risk scenario --file configs/scenario_high_risk_launch.json
	prembly-risk monitor
test:
	python -m unittest discover -s tests -v
lint:
	ruff check src tests

