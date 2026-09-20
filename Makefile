.PHONY: data-check test

data-check:
	python scripts/data_check.py

test:
	pytest tests/unit
