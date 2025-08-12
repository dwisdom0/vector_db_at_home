import sqlite3
import warnings
import numpy as np
from pathlib import Path
from contextlib import contextmanager


class VectorStore:
    def __init__(self, db_path: str | Path, dim: int):
        self.db_path = db_path
        self.dim = dim
        self.numpy_dtype = np.float32

        self.allowed_input_types = set(
            [
                np.dtype(x)
                for x in (
                    np.bool_,
                    np.int_,
                    np.int8,
                    np.int16,
                    np.int32,
                    np.int64,
                    np.float16,
                    np.float16,
                    np.float32,
                    np.float64,
                    np.uint,
                    np.uint8,
                    np.uint16,
                    np.uint32,
                    np.uint64,
                )
            ]
        )

        with open("schema.sql", "r") as f:
            schema_sql = f.read()

        with self.connect() as con:
            con.executescript(schema_sql)

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        con.execute("pragma foreign_keys=on;")
        try:
            yield con
        finally:
            con.commit()
            con.close()

    def coerce_to_float32(self, arr: np.ndarray):
        if arr.dtype not in self.allowed_input_types:
            raise ValueError(f"input vectors of dtype {arr.dtype} are not supported")

        if arr.dtype != self.numpy_dtype:
            warnings.warn(
                f"Expected an array with a dtype of {self.numpy_dtype}, but got an array of {arr.dtype}. Attempting to coerce to {self.numpy_dtype}"
            )
        return arr.astype(np.float32)

    def blobs_to_ndarray(self, blobs: list[bytes]) -> np.ndarray | None:
        if len(blobs) == 0:
            return None

        return np.concat(
            [np.frombuffer(blob, dtype=self.numpy_dtype) for blob in blobs]
        )

    def ndarray_to_blobs(self, arr: np.ndarray) -> list[bytes]:
        if len(arr.shape) == 2:
            return [self.coerce_to_float32(a).tobytes() for a in arr]
        return [self.coerce_to_float32(arr).tobytes()]

    def count(self):
        # TODO: maybe manage this myself with a class variable
        with self.connect() as con:
            res = con.execute("SELECT count(*) FROM vector;").fetchone()
        return res[0]

    # TODO: also tail probably
    def head(self, n: int) -> np.ndarray | None:
        if self.count() == 0 or n == 0:
            return None
        with self.connect() as con:
            rows = con.execute(
                "SELECT * FROM vector ORDER BY id LIMIT ?", (n,)
            ).fetchall()
        return self.blobs_to_ndarray([row["vec"] for row in rows])

    def insert(self, arr: np.ndarray):
        if len(arr.shape) == 2 and arr.shape[1] != self.dim:
            raise ValueError(
                f"Cannot insert a vector shaped like {arr.shape} into a store that only holds vectors with {self.dim} elements"
            )
        elif len(arr.shape) == 1 and arr.shape[0] != self.dim:
            raise ValueError(
                f"Cannot insert a vector shaped like {arr.shape} into a store that only holds vectors with {self.dim} elements"
            )
        elif len(arr.shape) >= 3:
            raise ValueError(
                f"Expected a 2D array of row vectors to insert like (n, {self.dim})"
            )

        to_insert = [{"vec": v} for v in self.ndarray_to_blobs(arr)]
        with self.connect() as con:
            con.executemany("INSERT INTO vector (vec) VALUES (:vec)", to_insert)
