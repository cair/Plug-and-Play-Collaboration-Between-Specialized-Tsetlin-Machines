from typing import Optional, Type, Union, List
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm

from tmcomposite.components.base import TMComponent

class TMComposite:

    def __init__(self, components: Optional[list[TMComponent]] = None,
                 use_multiprocessing: bool = False) -> None:
        self.components: List[TMComponent] = components or []

        self.use_multiprocessing = use_multiprocessing

    def _mp_fit(self, args: tuple) -> None:
        idx, component, data = args
        
        data_preprocessed = component.preprocess(data)

        epochs = component.epochs
        pbar = tqdm(total=epochs, position=idx)
        pbar.set_description(f"Component {idx}: {type(component).__name__}")
        for _ in range(epochs):
            component.fit(data=data_preprocessed)
            pbar.update(1)
        return component
        

    def fit(self, data: dict) -> None:
        if self.use_multiprocessing:

            with Pool() as pool:
                self.components = pool.map(self._mp_fit, ((idx, component, data) for idx, component in enumerate(self.components)))
        else:
            data_preprocessed = [component.preprocess(data) for component in self.components]
            epochs_left = [component.epochs for component in self.components]
            pbars = [tqdm(total=component.epochs) for component in self.components]
            for idx, (pbar, component) in enumerate(zip(pbars, self.components)):
                pbar.set_description(f"Component {idx}: {type(component).__name__}")

            while any(epochs_left):
                for idx, component in enumerate(self.components):
                    if epochs_left[idx] > 0:
                        component.fit(data=data_preprocessed[idx])
                        pbars[idx].update(1)
                        epochs_left[idx] -= 1

    def predict(self, data: dict) -> np.array:
        votes = None

        for component in self.components:
            data_preprocessed = component.preprocess(data)
            _, scores = component.predict(data_preprocessed)
            if votes is None:
                votes = np.zeros_like(scores, dtype=np.float32)

            max_score = np.max(scores, axis=1, keepdims=True)
            min_score = np.min(scores, axis=1, keepdims=True)
            denom = max_score - min_score
            denom[denom == 0] = 1.0  # Avoid division by zero element-wise

            normalized_scores = scores / denom
            votes += normalized_scores

        return votes.argmax(axis=1)

    def save_model(self, path: Union[Path, str], format="pkl") -> None:
        path = Path(path) if isinstance(path, str) else path

        if format == "pkl":
            import pickle
            with open(path, "wb") as f:
                pickle.dump(self, f)
        else:
            raise NotImplementedError(f"Format {format} not supported")

    def load_model(self, path: Union[Path, str], format="pkl") -> None:
        path = Path(path) if isinstance(path, str) else path

        if format == "pkl":
            import pickle
            with open(path, "rb") as f:
                loaded_model = pickle.load(f)
                self.__dict__.update(loaded_model.__dict__)
        else:
            raise NotImplementedError(f"Format {format} not supported")