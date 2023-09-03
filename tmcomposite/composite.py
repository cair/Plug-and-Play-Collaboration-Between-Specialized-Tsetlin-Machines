import re
import threading
from collections import defaultdict
from os import cpu_count
from typing import Optional, Type, Union, List
from pathlib import Path
from multiprocessing import Pool, Queue, Manager
import numpy as np
from tqdm import tqdm

from tmcomposite.callbacks.base import TMCompositeCallbackProxy
from tmcomposite.components.base import TMComponent
from tmcomposite.gating.base import BaseGate
from tmcomposite.gating.linear_gate import LinearGate


class TMComposite:

    def __init__(
            self, 
            components: Optional[list[TMComponent]] = None,
            gate_function: Optional[Type[BaseGate]] = None,
            gate_function_params: Optional[dict] = None,
            use_multiprocessing: bool = False
    ) -> None:
        self.components: List[TMComponent] = components or []
        self.use_multiprocessing = use_multiprocessing

        if gate_function_params is None:
            gate_function_params = dict()

        self.gate_function_instance = gate_function(self, **gate_function_params) if gate_function else LinearGate(self, **gate_function_params)

    def _listener(self, queue, callbacks):
        while True:
            item = queue.get()
            if item == 'DONE':
                break
            method, *args = item
            for callback in callbacks:
                getattr(callback, method)(*args)

    @staticmethod
    def _mp_fit(args: tuple) -> None:
        idx, component, data_preprocessed, proxy_callback = args

        if proxy_callback:
            proxy_callback.on_train_composite_begin(composite=component)

        epochs = component.epochs
        pbar = tqdm(total=epochs, position=idx)
        pbar.set_description(f"Component {idx}: {type(component).__name__}")
        for epoch in range(epochs):
            if proxy_callback:
                proxy_callback.on_epoch_component_begin(component=component, epoch=epoch)
            component.fit(data=data_preprocessed)
            pbar.update(1)
            if proxy_callback:
                proxy_callback.on_epoch_component_end(component=component, epoch=epoch)

        if proxy_callback:
            proxy_callback.on_train_composite_end(composite=component)
        return component

    def fit(self, data: dict, callbacks: Optional[list] = None) -> None:

        if self.use_multiprocessing:
            with Manager() as manager:

                if callbacks:
                    callback_queue = manager.Queue()  # Create a queue with the manager
                    callback_proxy = TMCompositeCallbackProxy(callback_queue)

                    # Start listener thread
                    listener_thread = threading.Thread(target=self._listener, args=(callback_queue, callbacks))
                    listener_thread.start()
                else:
                    callback_proxy = None

                with Pool() as pool:
                    data_preprocessed = [component.preprocess(data) for component in self.components]
                    self.components = pool.map(self._mp_fit,
                                               ((idx, component, data_preprocessed[idx], callback_proxy) for idx, component in
                                                enumerate(self.components)))

                if callbacks:
                    callback_queue.put('DONE')  # Send done signal to listener
                    listener_thread.join()  # Wait for listener to process all logs


        else:
            data_preprocessed = [component.preprocess(data) for component in self.components]
            epochs_left = [component.epochs for component in self.components]
            pbars = [tqdm(total=component.epochs) for component in self.components]
            for idx, (pbar, component) in enumerate(zip(pbars, self.components)):
                pbar.set_description(f"Component {idx}: {type(component).__name__}")

            [callback.on_train_begin(composite=self) for callback in callbacks]
            epoch = 0
            while any(epochs_left):
                for idx, component in enumerate(self.components):
                    if epochs_left[idx] > 0:
                        component.fit(data=data_preprocessed[idx])
                        pbars[idx].update(1)
                        epochs_left[idx] -= 1
                [callback.on_epoch_end(composite=self, epoch=epoch) for callback in callbacks]
                epoch += 1

            [callback.on_train_enc(composite=self) for callback in callbacks]

    def _component_predict(self, component, data):
        data_preprocessed = component.preprocess(data)
        _, scores = component.predict(data_preprocessed)

        votes = dict()
        votes["composite"] = np.zeros_like(scores, dtype=np.float32)
        votes[str(component)] = np.zeros_like(scores, dtype=np.float32)

        for i in range(scores.shape[0]):
            denominator = np.max(scores[i]) - np.min(scores[i])
            score = 1.0 * scores[i] / denominator if denominator != 0 else 0
            votes["composite"][i] += score
            votes[str(component)][i] += score

        return votes

    def predict(self, data: dict, use_multiprocessing: bool = True) -> np.array:
        votes = dict()

        # Gating Mechanism
        gating_mask: np.ndarray = self.gate_function_instance.predict(data)
        assert gating_mask.shape[1] == len(self.components)
        assert gating_mask.shape[0] == data["Y"].shape[0]

        if use_multiprocessing:
            # Determine number of processes based on available CPU cores
            n_processes = min(cpu_count(), len(self.components))

            with Pool(n_processes) as pool:
                results = pool.starmap(self._component_predict, [
                    (component, data) for i, component in enumerate(self.components)
                ])

                # Aggregate results from each process
                for i, result in enumerate(results):
                    for key, score in result.items():

                        # Apply gating mask
                        masked_score = score * gating_mask[:, i]

                        if key not in votes:
                            votes[key] = masked_score
                        else:
                            votes[key] += masked_score
        else:
            for i, component in tqdm(enumerate(self.components)):
                component_votes = self._component_predict(component, data)
                for key, score in component_votes.items():
                        
                    # Apply gating mask
                    masked_score = score * gating_mask[:, i]

                    if key not in votes:
                        votes[key] = masked_score
                    else:
                        votes[key] += masked_score

        return {k: v.argmax(axis=1) for k, v in votes.items()}

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

    def load_model_from_components(self, path: Union[Path, str], format="pkl") -> None:
        path = Path(path) if isinstance(path, str) else path

        if not path.is_dir():
            raise ValueError(f"{path} is not a directory.")

        # Get all files with the desired format
        files = [f for f in path.iterdir() if f.is_file() and f.suffix == f".{format}"]

        # Group files by component details
        component_groups = defaultdict(list)
        for file in files:
            parts = file.stem.split('-')
            epoch = int(parts[-1])
            component_name = '-'.join(parts[:-1])
            component_groups[component_name].append((epoch, file))

        # Sort files in each group based on the epoch and get the highest epoch file
        latest_files = [sorted(group, key=lambda x: x[0], reverse=True)[0][1] for group in component_groups.values()]

        # Load all components and add to components list
        self.components = []
        if format == "pkl":
            import pickle
            for component_path in latest_files:
                with open(component_path, "rb") as f:
                    loaded_component = pickle.load(f)
                    self.components.append(loaded_component)
        else:
            raise NotImplementedError(f"Format {format} not supported")
