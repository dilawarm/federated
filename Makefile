clean:
	rm -rf .coverage
	rm -rf .pytest_cache
	rm -rf docs/_build/
	rm -rf federated/__pycache__
	rm -rf federated/*/__pycache__
	rm -rf .ipynb_checkpoints
	rm -rf notebooks/.ipynb_checkpoints
	rm -rf history
	rm -rf htmlcov
	rm -rf venv

docker:
	make clean
	docker build --tag federated:latest .
	docker run -t -i federated:latest bash

help:
	python -m federated.main --help

install:
	pip install -r requirements.txt

tests: 
	python -m pytest --cov=federated/ federated/tests/

test_data:
	python -m pytest --cov=federated/data/ federated/tests/data_preprocessing_test.py

test_models:
	python -m pytest --cov=federated/models/ federated/tests/models_test.py

test_training:
	python -m pytest --cov=federated/utils/training_loops.py federated/tests/training_loops_test.py

test_rfa:
	python -m pytest --cov=federated/utils/rfa.py federated/tests/rfa_test.py
