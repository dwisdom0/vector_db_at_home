import json
import os
import sqlite3
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SelectRecord:
    id: int
    vec: np.ndarray
    doc: dict


@dataclass
class SearchRecord:
    id: int
    vec: np.ndarray
    doc: dict
    distance: float


class VectorStore:
    def __init__(self, db_path: str | Path, dim: int):
        self.db_path = db_path
        self.dim = dim
        self.vec_dtype = np.float32

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

        # https://numpy.org/doc/stable/user/basics.rec.html#structured-arrays
        # I'm going to go with the structured array approach
        # maybe I should do polars dataframe or pandas dataframe instead
        # the docs suggest xarray, pandas, and DataArray
        # I would think vaex as well. I don't get why no one ever shouts out vaex.
        # I mean, I've never used it so maybe there's some reason.
        # I'll just do it this way and I can change it if it becomes a problem
        self.structured_dtype = np.dtype(
            [("id", np.uint64), ("vec", self.vec_dtype, self.dim)]
        )
        self.index = np.empty((0,), dtype=self.structured_dtype)

        if os.path.exists(self.db_path):
            self.load_from_existing()

        else:
            with open(os.path.join(os.path.dirname(__file__), "schema.sql"), "r") as f:
                schema_sql = f.read()

            with self.connect() as con:
                con.executescript(schema_sql)

    def __repr__(self):
        return f"VectorStore(db_path={self.db_path}, dim={self.dim})"

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
        self.index = np.array(
            [
                (row["id"], np.frombuffer(row["vec"], dtype=self.vec_dtype))
                for row in rows
            ],
            dtype=self.structured_dtype,
        )

    def float32_row_vecs(self, arr: np.ndarray):
        if arr.dtype not in self.allowed_input_types:
            raise ValueError(f"input vectors of dtype {arr.dtype} are not supported")

        if arr.dtype != self.vec_dtype:
            warnings.warn(
                f"Expected an array with a dtype of {self.vec_dtype}, but got an array of {arr.dtype}. Coercing to {self.vec_dtype}"
            )
        return arr.reshape(-1, self.dim).astype(self.vec_dtype)

    def blobs_to_ndarray(self, blobs: list[bytes]) -> np.ndarray:
        if len(blobs) == 0:
            return np.empty((0, 0), dtype=self.vec_dtype)

        return np.concat(
            [np.frombuffer(blob, dtype=self.vec_dtype) for blob in blobs]
        ).reshape(-1, self.dim)

    def ndarray_to_blobs(self, arr: np.ndarray) -> list[bytes]:
        return [self.float32_row_vecs(a).tobytes() for a in arr]

    @staticmethod
    def json_parse(s: str | None) -> dict:
        try:
            return json.loads(s)  # type: ignore
        except TypeError:
            return dict()

    @staticmethod
    def json_dump(d: dict | None) -> str:
        try:
            return json.dumps(d)  # type: ignore
        except TypeError:
            return "{}"

    def count(self):
        with self.connect() as con:
            res = con.execute("SELECT count(*) FROM vector;").fetchone()
        return res[0]

    # TODO: tail
    def head(self, n: int = 5) -> list[dict]:
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
                    "doc": self.json_parse(row["doc"]),
                }
            )
        return to_return

    def insert_dicts(self, ds: list[dict]):
        # expect certain keys
        # {
        #   "vec": np.ndarray
        #   "doc": optional dict with whatever the user wants, must be json-seralizable
        # }
        vecs = []
        docs = []
        for d in ds:
            vecs.append(d.get("vec", None))
            # assert that the doc is json-serializable
            try:
                _ = json.dumps(d["doc"])
            except TypeError as e:
                raise TypeError(f"docs must be JSON serializable: {e}")
            except KeyError:
                pass
            docs.append(d.get("doc", None))
        self.insert(np.stack(vecs), docs)

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
        # so I manually insert ids like range(max_id, max_id + num_vecs)
        # which will leave holes in the id column if things get deleted but that's fine

        blobs = self.ndarray_to_blobs(vecs)
        ids = list(range(start_id, start_id + len(blobs)))
        if docs is None:
            docs = [dict()] * len(ids)
        dumped_docs = [self.json_dump(doc) for doc in docs]

        to_insert = [
            {"id": i, "vec": v, "doc": d} for i, v, d in zip(ids, blobs, dumped_docs)
        ]

        with self.connect() as con:
            con.executemany(
                "INSERT INTO vector (id, vec, doc) VALUES (:id, :vec, :doc);",
                to_insert,
            )

        self.index = np.concat(
            [
                self.index,
                np.array(
                    [(i, vec) for i, vec in zip(ids, vecs)], dtype=self.structured_dtype
                ),
            ]
        )

    def delete(self, ids: list[int]):
        with self.connect() as con:
            # https://sqlite.org/limits.html
            # SQLite limits the number of placeholders
            # some versions limit it to 999
            # others ~32k
            # can check SQLITE_MAX_VARIABLE_NUMBER if we want to be super safe
            # other wise I think it's fine to just let it error
            placeholders = ",".join(["?" for _ in ids])
            count_result = con.execute(
                f"select count(id) from vector where id in ({placeholders})", ids
            ).fetchone()
            if count_result[0] != len(ids):
                warnings.warn(
                    "At least one of the ids you're trying to delete doesn't exist in the database"
                )

            con.executemany("DELETE FROM vector WHERE id = ?", [(i,) for i in ids])
        self.index = self.index[~np.isin(self.index["id"], ids)]

    def select_ids(self, ids: list[int]) -> list[SelectRecord]:
        placeholders = ",".join(["?" for _ in ids])
        with self.connect() as con:
            results = con.execute(
                f"select id, vec, doc from vector where id in ({placeholders}) order by id;",
                ids,
            ).fetchall()

        records = []
        for result in results:
            records.append(
                SelectRecord(
                    id=result["id"],
                    vec=self.blobs_to_ndarray([result["vec"]]),
                    doc=self.json_parse(result["doc"]),
                )
            )
        return records

    def search(self, query: np.ndarray, k: int) -> list[list[SearchRecord]]:
        if self.index is None:
            return list(list(dict()))

        # TODO: handle k > len(self.index)
        # FAISS handles this by padding the results list with -1
        if k > len(self.index):
            raise ValueError(
                f"Asked for {k} results but there are only {len(self.index)} vectors in the index"
            )
        q_vecs = self.float32_row_vecs(query)

        # TODO: vectorize this loop
        search_ids = []
        search_distances = []
        for q_vec in q_vecs:
            distances = np.linalg.norm(self.index["vec"] - q_vec, ord=2, axis=1)
            search_distances.append(np.sort(distances)[:k])
            # search_distances.append(distances[])
            # these ids have nothing to do with our real ids
            # they're just a 0-based enumeration of our current self.index items
            # so we have to go get the real ids from self.index
            result_ids = np.argsort(distances)[:k]
            search_ids.append(self.index[result_ids]["id"])

        search_ids = np.array(search_ids)
        search_distances = np.array(search_distances)

        # it's possible that the same result could show up multiple times
        # if there are multiple query vectors
        # but we only want to get each result from the db once
        unique_ids = np.unique(search_ids).tolist()
        placeholders = ",".join(["?" for id_ in unique_ids if id_ != -1])

        with self.connect() as con:
            rows = con.execute(
                f"select id, vec, doc from vector where id in ({placeholders})",
                unique_ids,
            ).fetchall()
        unique_results = {}
        for row in rows:
            unique_results[row["id"]] = {
                "id": int(row["id"]),
                "vec": self.blobs_to_ndarray([row["vec"]])[0],
                "doc": self.json_parse(row["doc"]),
            }

        # fill in a 2D list of dicts for the results
        result = []
        for i, r in enumerate(search_ids):
            result_row = []
            for j, id_ in enumerate(r):
                result_row.append(
                    SearchRecord(**unique_results[id_], distance=search_distances[i][j])
                )
            result.append(result_row)

        return result

    def query_by_doc(self, path: list[str], values: list[str | int]) -> list[dict]:
        json_path = "$." + ".".join(path)

        placeholders = ",".join(["?" for _ in range(len(values))])

        with self.connect() as con:
            rows = con.execute(
                f"""\
                SELECT id, vec, doc
                FROM vector
                WHERE json_extract(doc, ?) in ({placeholders});""",
                (json_path, *values),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "vec": self.blobs_to_ndarray([r["vec"]]),
                "doc": self.json_parse(r["doc"]),
            }
            for r in rows
        ]

    def dump_vecs(self):
        return self.index["vec"]
