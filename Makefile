all:
	pip install cython
	pip install pytest
	pip install funcy
	python setup.py build_ext --inplace

test:
	pytest tests/test_bonsai.py

clean::
	@find . -name \*~ -exec rm -v '{}' +
	@find . -name \*.pyc -exec rm -v '{}' +
	@find . -name __pycache__ -prune -exec rm -vfr '{}' +
	@find . -name \*.so -prune -exec rm -vfr '{}' +
	@rm -rf build bdist dist sdist
	@rm -rf .tox .eggs
	@rm -rf venv
	@find . \( -name \*.orig -o -name \*.bak -o -name \*.rej \) -exec rm -v '{}' +
	@rm -rf distribute-* *.egg *.egg-info *.tar.gz junit.xml .cache .mypy_cache .pytest_cache
	@rm -rf tmp/*

.PHONY: clean test