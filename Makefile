init:
	pip3 install -r requirements.txt

test_data:
	python3 -m federated.data.mitbih_data_preprocessing_test

test_models:
	python3 -m federated.models.mitbih_model_test

test_training:
	python3 -m federated.utils.training_loops_test