from typing import Tuple
import numpy as np
import cv2
from tmu.models.classification.vanilla_classifier import TMClassifier

from tmcomposite.components.base import TMComponent

class AdaptiveThresholdingComponent(TMComponent):
            
            def __init__(self, model_cls, model_config, **kwargs) -> None:
                super().__init__(model_cls=model_cls, model_config=model_config, **kwargs)
    
            def preprocess(self, data: dict):
                super().preprocess(data=data)

                X_org = data["X"]
                X = np.copy(data["X"])
                Y = np.copy(data["Y"])
                
    
                Y = Y.reshape(Y.shape[0])

                for i in range(X.shape[0]):
                        for j in range(X.shape[3]):
                                X[i,:,:,j] = cv2.adaptiveThreshold(X_org[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
   
                return dict(
                    X=X,
                    Y=Y,
                )
    
            def fit(self, data: dict) -> None:
                X_train, Y_train = data["X"], data["Y"]
                self.model_instance.fit(X_train, Y_train)

            def predict(self, data: dict) -> Tuple[np.array, np.array]:
                X_test = data["X"]
                return self.model_instance.predict(X_test, return_class_sums=True)