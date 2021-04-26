from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib

eps, _ = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(100000, 32, 0.1, 1, 1e-5)

print(eps)
