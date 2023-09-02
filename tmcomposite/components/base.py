import abc
import pickle
from pathlib import Path
from typing import Union


class TMComponent(abc.ABC):

    def __init__(self, model_cls, model_config, epochs=1, **kwargs) -> None:
        self.model_cls = model_cls
        self.model_config = model_config
        self.epochs = epochs

        # Warn about unused kwargs
        if kwargs:
            print(f"Warning: unused keyword arguments: {kwargs}")

        self.model_instance = self.model_cls(
            **self.model_config.dict()
        )

    def model(self):
        raise NotImplementedError

    def preprocess(self, data: dict) -> dict:
        assert "X" in data, "X must be in data"
        assert "Y" in data, "Y must be in data"

        # Check if this is called from a class that inherits from TMComponent
        if not isinstance(self, TMComponent):
            raise TypeError(f"{type(self).__name__} does not inherit from TMComponent")

        return data

    def fit(self, data: dict) -> None:
        pass

    @abc.abstractmethod
    def predict(self, data: dict):
        pass

    def save(self, path: Union[Path, str], format="pkl") -> None:
        path = Path(path) if isinstance(path, str) else path

        if format == "pkl":
            import pickle
            with open(path, "wb") as f:
                pickle.dump(self, f)


    def __str__(self):
        params = '-'.join([f"{k}={v}" for k, v in self.model_config.dict().items()])
        return f"{type(self).__name__}-{self.model_cls.__name__}-{params})"
