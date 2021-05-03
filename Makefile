init:
	pip3 install -r requirements.txt

test: test_data test_models test_training test_rfa

test_data:
	python3 -m federated.tests.data_preprocessing_test

test_models:
	python3 -m federated.tests.models_test

test_training:
	python3 -m federated.tests.training_loops_test

test_rfa:
	python3 -m federated.tests.rfa_test

run_federated:
	python3 -m federated.optimization.federated

run_centralized:
	python3 -m federated.optimization.centralized
