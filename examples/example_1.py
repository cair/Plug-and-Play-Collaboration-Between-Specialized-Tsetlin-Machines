from tmcomposite.callbacks.base import TMCompositeCallback
from tmcomposite.composite import TMComposite
from tmcomposite.components.adaptive_thresholding import AdaptiveThresholdingComponent
from tmcomposite.components.color_thermometer_scoring import ColorThermometerComponent
from tmcomposite.components.histogram_of_gradients import HistogramOfGradientsComponent
from tmcomposite.config import TMClassifierConfig
from keras.datasets import cifar10
from tmu.models.classification.vanilla_classifier import TMClassifier
import pathlib


if __name__ == "__main__":

    epochs = 100
    checkpoint_path = pathlib.Path("checkpoints")
    checkpoint_path.mkdir(exist_ok=True)

    composite_path = checkpoint_path / "composite"
    composite_path.mkdir(exist_ok=True)

    component_path = checkpoint_path / "components"
    component_path.mkdir(exist_ok=True)

    (X_train_org, Y_train), (X_test_org, Y_test) = cifar10.load_data()
    Y_test = Y_test.reshape(Y_test.shape[0])
    Y_train = Y_train.reshape(Y_train.shape[0])

    data_train = dict(
        X=X_train_org,
        Y=Y_train
    )

    data_test = dict(
        X=X_test_org,
        Y=Y_test
    )


    class TMCompositeCheckpointCallback(TMCompositeCallback):

        def on_epoch_component_begin(self, component, epoch, logs=None):
            pass

        def on_epoch_component_end(self, component, epoch, logs=None):
            component.save(component_path / f"{component}-{epoch}.pkl")

    class TMCompositeEvaluationCallback(TMCompositeCallback):

        def __init__(self, data):
            super().__init__()
            self.best_acc = 0.0
            self.data = data

       # def on_epoch_end(self, composite, epoch, logs=None):
       #     preds = composite.predict(data=self.data)
       #     print("Team Accuracy: %.1f" % (100 * (preds == self.data["Y"]).mean()))

    # Define the composite model
    composite_model = TMComposite(
        components=[
            AdaptiveThresholdingComponent(TMClassifier, TMClassifierConfig(
                num_clauses=2000,
                T=500,
                s=10.0,
                max_included_literals=32,
                platform="CPU",
                weighted_clauses=True,
                patch_dim=(10, 10),
            ), epochs=epochs),

            ColorThermometerComponent(TMClassifier, TMClassifierConfig(
                num_clauses=2000,
                T=1500,
                s=2.5,
                max_included_literals=32,
                platform="CPU",
                weighted_clauses=True,
                patch_dim=(3, 3),
            ), resolution=8, epochs=epochs),

            ColorThermometerComponent(TMClassifier, TMClassifierConfig(
                num_clauses=2000,
                T=1500,
                s=2.5,
                max_included_literals=32,
                platform="CPU",
                weighted_clauses=True,
                patch_dim=(4, 4),
            ), resolution=8, epochs=epochs),

            HistogramOfGradientsComponent(TMClassifier, TMClassifierConfig(
                num_clauses=2000,
                T=50,
                s=10.0,
                max_included_literals=32,
                platform="CPU",
                weighted_clauses=False
            ), epochs=epochs)
        ],
        use_multiprocessing=True
    )

    # Train the composite model
    composite_model.fit(
        data=data_train,
        callbacks=[
            TMCompositeCheckpointCallback(),
            TMCompositeEvaluationCallback(data=data_test)
        ]
    )

    preds = composite_model.predict(data=data_test)

    y_true = data_test["Y"].flatten()
    for k, v in preds.items():
        print(f"{k} Accuracy: %.1f" % (100 * (v == y_true).mean()))
