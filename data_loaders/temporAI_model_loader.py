import pickle
import io
import torch
from tempor.utils.serialization import load_from_file


def load_model(model_path):
    """
    Loads a TemporAI model from the specified path and converts it to CPU if necessary.

    Args:
        model_path (str): The path to the model file.

    Returns:
        The loaded model.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        model = CPU_Unpickler(open(model_path, 'rb')).load()
    else:
        model = load_from_file(model_path)
    return model


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


