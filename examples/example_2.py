from tmcomposite.tuner import TMCompositeTuner
from keras.datasets import cifar10

if __name__ == "__main__":

    (X_train_org, Y_train), (X_test_org, Y_test) = cifar10.load_data()

    data_train = dict(
        X=X_train_org,
        Y=Y_train
    )

    data_test = dict(
        X=X_test_org,
        Y=Y_test
    )


    # Instantiate tuner
    tuner = TMCompositeTuner(
        data_train=data_train,
        data_test=data_test,
        n_jobs=8  # for parallelization; set to 1 for no parallelization
    )

    # Specify number of trials (iterations of the tuning process)
    n_trials = 100

    # Run the tuner
    best_params, best_value = tuner.tune(n_trials=n_trials)

    # Print out the results
    print("Best Parameters:", best_params)
    print("Best Value:", best_value)
