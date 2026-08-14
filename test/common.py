import numpy as np


def assertNumpyEqual(a: np.ndarray, b: np.ndarray):
    return np.testing.assert_array_equal(a, b, verbose=True, strict=True)


def gen_docs(ns: list) -> list[dict]:
    return [{f"k{n}": f"v{n}"} for n in ns]
