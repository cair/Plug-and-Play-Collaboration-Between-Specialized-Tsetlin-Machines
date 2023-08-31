from typing import Tuple
import numpy as np
import cv2

from tmcomposite.components.base import TMComponent

class HistogramOfGradientsComponent(TMComponent):
            
    def __init__(
            self, 
            model_cls,
            model_config, 
            resolution=8,
            block_size = 12,
            block_stride = 4,
            cell_size = 4,
            nbins = 18,
            deriv_aperture = 1,
            win_sigma = -1.,
            histogram_norm_nype = 0,
            L2_hys_threshold = 0.2,
            gamma_correction = True,
            n_levels = 64,
            signed_gradient = True,
            **kwargs
    ) -> None:
        super().__init__(model_cls=model_cls, model_config=model_config, **kwargs)
        self.resolution = resolution
        self.block_size = block_size
        self.block_stride = block_stride
        self.cell_size = cell_size
        self.nbins = nbins
        self.deriv_aperture = deriv_aperture
        self.win_sigma = win_sigma
        self.histogram_norm_nype = histogram_norm_nype
        self.L2_hys_threshold = L2_hys_threshold
        self.gamma_correction = gamma_correction
        self.n_levels = n_levels
        self.signed_gradient = signed_gradient

    def preprocess(self, data: dict):
        super().preprocess(data=data)

        X_org = data["X"]
        X = np.copy(data["X"])
        Y = np.copy(data["Y"])


        win_size = X_org.shape[1]   
        hog = cv2.HOGDescriptor(
            (win_size, win_size),
            (self.block_size, self.block_size),
            (self.block_stride, self.block_stride),
            (self.cell_size, self.cell_size), 
            self.nbins, 
            self.deriv_aperture, 
            self.win_sigma,
            self.histogram_norm_nype,
            self.L2_hys_threshold,
            self.gamma_correction,
            self.n_levels, 
            self.signed_gradient
        )



        Y=Y.reshape(Y.shape[0])
        
        fd = hog.compute(X_org[0])
        X = np.empty((X_org.shape[0], fd.shape[0]), dtype=np.uint32)
        for i in range(X_org.shape[0]):
            fd = hog.compute(X_org[i]) 
            X[i] = fd >= 0.1

        return dict(
            X=X,
            Y=Y
        )

    def fit(self, data: dict) -> None:
        X, Y = data["X"], data["Y"]
        self.model_instance.fit(X, Y)

    def predict(self, data: dict) -> Tuple[np.array, np.array]:
        X_test = data["X"]
        return self.model_instance.predict(X_test, return_class_sums=True)