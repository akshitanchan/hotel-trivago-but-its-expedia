import numpy as np
import lightgbm as lgb


def get_lgb_device_params(verbose: bool = True) -> dict:
    """Return LightGBM GPU params when GPU training is available."""
    try:
        dataset = lgb.Dataset(
            np.array([[0, 1], [1, 0]], dtype=np.float32),
            label=np.array([0, 1], dtype=np.int8),
        )
        lgb.train(
            {"objective": "binary", "device": "gpu", "verbose": -1, "num_threads": 1},
            dataset,
            num_boost_round=1,
        )
    except Exception:
        if verbose:
            print("CPU mode - no compatible LightGBM GPU detected")
        return {}

    if verbose:
        print("GPU mode enabled - using LightGBM GPU acceleration")
    return {"device": "gpu", "gpu_use_dp": False}
