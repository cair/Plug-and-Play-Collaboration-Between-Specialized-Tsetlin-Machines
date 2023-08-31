   

from tmcomposite.composite import TMComposite
from tmcomposite.components.adaptive_thresholding import AdaptiveThresholdingComponent
from tmcomposite.components.color_thermometer_scoring import ColorThermometerComponent
from tmcomposite.components.histogram_of_gradients import HistogramOfGradientsComponent
from tmcomposite.config import TMClassifierConfig
from keras.datasets import cifar10
from tmu.models.classification.vanilla_classifier import TMClassifier


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

    epochs = 100

    # Define the composite model
    composite = TMComposite(
        components=[
            AdaptiveThresholdingComponent(TMClassifier, TMClassifierConfig(
                num_clauses=2000,
                T=500, 
                s=10.0,
                max_included_literals=32,
                device="GPU",
                weighted_clauses=True,
                patch_dim=(10, 10),
            ), epochs=epochs),
            ColorThermometerComponent(TMClassifier, TMClassifierConfig(
                num_clauses=2000,
                T=1500,
                s=2.5,
                max_included_literals=32,
                device="GPU",
                weighted_clauses=True,
                patch_dim=(3, 3),
            ), resolution=8, epochs=epochs),

            HistogramOfGradientsComponent(TMClassifier, TMClassifierConfig(
                num_clauses=2000,
                T=50,
                s=10.0,
                max_included_literals=32,
                device="GPU",
                weighted_clauses=False
            ), epochs=epochs)
        ],
        use_multiprocessing=True
    )

    # Train the composite model
    composite.fit(data=data_train)

    preds = composite.predict(data=data_test)

    print("Team Accuracy: %.1f" % (100*(preds == data_test["Y"]).mean()))

    composite.save_model("test_model.pkl")
    composite.load_model("test_model.pkl")


