install:
	pip3 install -r requirements.txt

test: 
	python3 -m pytest --cov=federated/ federated/tests/

test_data:
	python3 -m federated.tests.data_preprocessing_test

test_models:
	python3 -m federated.tests.models_test

test_training:
	python3 -m federated.tests.training_loops_test

test_rfa:
	python3 -m federated.tests.rfa_test