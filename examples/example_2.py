from tmcomposite.tuner import TMCompositeTuner
from tmu.data.cifar10 import CIFAR10
import numpy as np


def create_subset(X, Y, samples_per_class):
    subset_X, subset_Y = [], []

    for i in range(10):  # 10 classes in CIFAR-10
        mask = (Y == i).reshape(-1)
        subset_X.append(X[mask][:samples_per_class])
        subset_Y.append(Y[mask][:samples_per_class])

    subset_X, subset_Y = np.vstack(subset_X), np.vstack(subset_Y)
    indices = np.arange(subset_X.shape[0])
    np.random.shuffle(indices)

    return subset_X[indices], subset_Y[indices]


if __name__ == "__main__":
    data = CIFAR10().get()
    X_train_org = data["x_train"]
    Y_train = data["y_train"]
    X_test_org = data["x_test"]
    Y_test = data["y_test"]

    """data_train = dict(
        X=X_train_org,
        Y=Y_train
    )

    data_test = dict(
        X=X_test_org,
        Y=Y_test
    )"""

    percentage = 0.1
    X_train_subset, Y_train_subset = create_subset(X_train_org, Y_train, int(5000 * percentage))
    X_test_subset, Y_test_subset = create_subset(X_test_org, Y_test, int(1000 * percentage))
    Y_test_subset = Y_test_subset.reshape(Y_test_subset.shape[0])
    Y_train_subset = Y_train_subset.reshape(Y_train_subset.shape[0])
    data_train = dict(X=X_train_subset, Y=Y_train_subset)
    data_test = dict(X=X_test_subset, Y=Y_test_subset)

    # Instantiate tuner
    tuner = TMCompositeTuner(
        data_train=data_train,
        data_test=data_test,
        n_jobs=1  # for parallelization; set to 1 for no parallelization
    )

    # Specify number of trials (iterations of the tuning process)
    n_trials = 100

    # Run the tuner
    best_params, best_value = tuner.tune(n_trials=n_trials)

    # Print out the results
    print("Best Parameters:", best_params)
    print("Best Value:", best_value)
