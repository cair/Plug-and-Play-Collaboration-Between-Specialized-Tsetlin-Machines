from typing import Tuple
import numpy as np
from tmcomposite.components.base import TMComponent

class ColorThermometerComponent(TMComponent):
            
            def __init__(self, model_cls, model_config, resolution=8, **kwargs) -> None:
                super().__init__(model_cls=model_cls, model_config=model_config, **kwargs)
                self.resolution = resolution
    
            def preprocess(self, data: dict):
                super().preprocess(data=data)

                X_org = data["X"]
                X = np.copy(data["X"])
                Y = np.copy(data["Y"])

                X = np.empty((X_org.shape[0], X_org.shape[1], X_org.shape[2], X_org.shape[3], self.resolution), dtype=np.uint8)
                for z in range(self.resolution):
                    X[:,:,:,:,z] = X_org[:,:,:,:] >= (z+1)*255/(self.resolution+1)


                X = X.reshape((X_org.shape[0], X_org.shape[1], X_org.shape[2], 3*self.resolution))
                Y=Y.reshape(Y.shape[0])

                return dict(
                    X=X,
                    Y=Y,
                )
    
            def fit(self, data: dict) -> None:
                X, Y = data["X"], data["Y"]
                self.model_instance.fit(X, Y)

            def predict(self, data: dict) -> Tuple[np.array, np.array]:
                X_test = data["X"]
                return self.model_instance.predict(X_test, return_class_sums=True)