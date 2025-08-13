import json
import os
import sqlite3
import warnings
import numpy as np
from faiss import IndexFlatIP, IndexIDMap
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

        # TODO: maybe remove FAISS and just do the brute-force knn search myself
        # https://gist.github.com/mdouze/a8c914eb8c5c8306194ea1da48a577d2

        # IndexFlat* doesn't support add_with_ids
        # IndexIVF* does if I end up switching to that
        # https://github.com/facebookresearch/faiss/wiki/Pre--and-post-processing#faiss-id-mapping
        self.faiss_index = IndexIDMap(IndexFlatIP(self.dim))

        if os.path.exists(self.db_path):
            self.load_from_existing()

        else:
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

    def load_from_existing(self):
        with self.connect() as con:
            rows = con.execute("SELECT id, vec FROM vector;").fetchall()
        ids = [row["id"] for row in rows]
        arr = self.blobs_to_ndarray([row["vec"] for row in rows])
        self.faiss_index.add_with_ids(arr, ids)  # type: ignore

    def float32_row_vecs(self, arr: np.ndarray):
        if arr.dtype not in self.allowed_input_types:
            raise ValueError(f"input vectors of dtype {arr.dtype} are not supported")

        if arr.dtype != self.numpy_dtype:
            warnings.warn(
                f"Expected an array with a dtype of {self.numpy_dtype}, but got an array of {arr.dtype}. Coercing to {self.numpy_dtype}"
            )
        return arr.reshape(-1, self.dim).astype(self.numpy_dtype)

    def blobs_to_ndarray(self, blobs: list[bytes]) -> np.ndarray | None:
        if len(blobs) == 0:
            return None

        return np.concat(
            [np.frombuffer(blob, dtype=self.numpy_dtype) for blob in blobs]
        ).reshape(-1, self.dim)

    def ndarray_to_blobs(self, arr: np.ndarray) -> list[bytes]:
        if len(arr.shape) == 2:
            return [self.float32_row_vecs(a).tobytes() for a in arr]
        return [self.float32_row_vecs(arr).tobytes()]

    @staticmethod
    def parse_json(s: str | None) -> dict:
        try:
            return json.loads(s)  # type: ignore
        except TypeError:
            return dict()

    def count(self):
        # TODO: maybe manage this myself with a class variable
        with self.connect() as con:
            res = con.execute("SELECT count(*) FROM vector;").fetchone()
        return res[0]

    # TODO: also tail probably
    def head(self, n: int) -> list[dict]:
        if self.count() == 0 or n == 0:
            return list()
        with self.connect() as con:
            rows = con.execute(
                "SELECT * FROM vector ORDER BY id LIMIT ?", (n,)
            ).fetchall()
        to_return = []
        for row in rows:
            to_return.append(
                {
                    "id": row["id"],
                    "vec": self.blobs_to_ndarray([row["vec"]]),
                    "doc": self.parse_json(row["doc"]),
                }
            )
        return to_return

    def insert(self, arr: np.ndarray, docs: list[dict] | None = None):
        vecs = self.float32_row_vecs(arr)
        if vecs.shape[1] != self.dim:
            raise ValueError(
                f"Cannot insert a vector shaped like {arr.shape} into a store that only holds vectors with {self.dim} elements"
            )

        if docs is not None and len(docs) != vecs.shape[0]:
            raise ValueError(
                f"The number of vectors ({vecs.shape[0]}) does not match the number of documents ({len(docs)})"
            )

        with self.connect() as con:
            row = con.execute(
                "SELECT id FROM vector ORDER BY id DESC LIMIT 1;"
            ).fetchone()
            # want the ids to be 0-indexed
            if row is None:
                start_id = 0
            else:
                start_id = row["id"] + 1

        # can't use RETURNING inside executemany()
        # https://docs.python.org/3/library/sqlite3.html#sqlite3.Cursor.executemany
        #
        # the other option is to chunk into batches of 999
        # (or look up max_variable_number for our current sqlite)
        # and build a single insert for each batch with multiple VALUES like
        # insert into vector (vec) values (?), (?), (?) ... returning id;
        #
        # or I manually insert ids like range(max_id, max_id + num_vecs)
        # which will leave holes in the id column if things get deleted but that's fine
        if docs is None:
            to_insert = [
                {"id": i, "vec": v, "doc": None}
                for i, v in enumerate(self.ndarray_to_blobs(vecs), start=start_id)
            ]
        else:
            to_insert = [
                {"id": i, "vec": z[0], "doc": json.dumps(z[1])}
                for i, z in enumerate(
                    zip(self.ndarray_to_blobs(vecs), docs), start=start_id
                )
            ]

        with self.connect() as con:
            con.executemany(
                "INSERT INTO vector (id, vec, doc) VALUES (:id, :vec, :doc);",
                to_insert,
            )

        self.faiss_index.add_with_ids(
            vecs,
            [values["id"] for values in to_insert],
        )  # type: ignore

    # TODO: maybe I just shouldn't support delete()
    def delete(self, ids: list[int]):
        with self.connect() as con:
            con.executemany("DELETE FROM vector WHERE id = ?", [(i,) for i in ids])
        self.faiss_index.remove_ids(ids)

    def search(
        self, query: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        q_vecs = self.float32_row_vecs(query)
        # Since we're using FlatIndexIP (inner product),
        # we want to maximize the distance metric rather than minimize it.
        # It makes more sense to call it "similarity"
        # as in "cosine similarity"
        # Also the function signatures on FAISS's self-modifying python wrapper classes are too complicated
        # for the python type checker to figure out
        similarities: np.ndarray
        ids: np.ndarray
        similarities, ids = self.faiss_index.search(q_vecs, k)  # type: ignore

        ids_list = ids.flatten().tolist()
        placeholders = f"{','.join(['?' for _ in ids_list])}"

        with self.connect() as con:
            rows = con.execute(
                f"select id, doc from vector where id in ({placeholders})", ids_list
            ).fetchall()
        docs = [self.parse_json(row["doc"]) for row in rows]

        return similarities, ids, docs
