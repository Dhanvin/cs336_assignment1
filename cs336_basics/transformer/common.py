import torch
from enum import Enum
import numpy as np
import numpy.typing as npt


def get_device():
    """Try to use the GPU if possible, otherwise, use CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


# Create a dictionary to map NumPy dtypes to PyTorch dtypes
np_torch_type_mapping = {
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.uint8: torch.uint8,
}

# Unsigned numpy dtypes are not compatible with
# Torch. First convert them to an appropriate type
torch_compatible_dtype_map = {np.uint16: np.int32, np.uint32: np.int64}


def np_array_to_tensor(np_data: npt.NDArray, device: str):
    compatible_np_data = np_data
    if np_data.dtype.type in torch_compatible_dtype_map:
        compatible_np_data = np_data.astype(
            torch_compatible_dtype_map[np_data.dtype.type]
        )
    return torch.from_numpy(compatible_np_data).to(device)


class SamplingStrategy(Enum):
    RANDOM = 1
    SEQ_NON_OVERLAPPING = 2
