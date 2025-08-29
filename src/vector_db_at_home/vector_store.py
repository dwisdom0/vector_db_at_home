import json
import os
import sqlite3
import warnings
import numpy as np
from pathlib import Path
from contextlib import contextmanager


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

        # TODO: maybe remove FAISS and just do the brute-force knn search myself
        # https://gist.github.com/mdouze/a8c914eb8c5c8306194ea1da48a577d2

        # I need to keep track of ids

        # so I might need a lookup dict
        # or I might need to make this like
        # [[id, vec], [id, vec]]
        # but I'm not sure I can do that in Numpy
        # in a way that makes it easy to get all the vecs at once
        # to do a search

        # suggestion from Qwen3
        #
        # # Create structured array
        # dtype = [('id', 'i4'), ('vec', 'f4', (128,))]  # 128-dimensional vectors
        # vectors = np.array([
        #     (1, np.random.rand(128)),
        #     (2, np.random.rand(128)),
        #     (3, np.random.rand(128)),
        # ], dtype=dtype)

        # # Access all vectors easily
        # all_vectors = vectors['vec']  # Shape: (3, 128)

        # # Access specific id
        # specific_vector = vectors[vectors['id'] == 2]['vec'][0]

        # # KNN search on all vectors
        # # Example: find nearest neighbors
        # distances = np.linalg.norm(all_vectors - target_vector, axis=1)
        # nearest_idx = np.argsort(distances)[:k]

        # another suggestion from Qwen3
        #
        # ids = np.array([1, 2, 3])
        # vectors = np.array([
        #     np.random.rand(128),
        #     np.random.rand(128),
        #     np.random.rand(128),
        # ])

        # # Easy access to all vectors for KNN
        # all_vectors = vectors  # Shape: (3, 128)

        # # Get vector by id
        # def get_vector_by_id(id_val):
        #     idx = np.where(ids == id_val)[0]
        #     return vectors[idx[0]] if len(idx) > 0 else None

        # # KNN search
        # distances = np.linalg.norm(vectors - target_vector, axis=1)
        # nearest_indices = np.argsort(distances)[:k]

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

    def blobs_to_ndarray(self, blobs: list[bytes]) -> np.ndarray | None:
        if len(blobs) == 0:
            return None

        return np.concat(
            [np.frombuffer(blob, dtype=self.vec_dtype) for blob in blobs]
        ).reshape(-1, self.dim)

    def ndarray_to_blobs(self, arr: np.ndarray) -> list[bytes]:
        return [self.float32_row_vecs(a).tobytes() for a in arr]

    @staticmethod
    def parse_json(s: str | None) -> dict:
        try:
            return json.loads(s)  # type: ignore
        except TypeError:
            return dict()

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
                    "doc": self.parse_json(row["doc"]),
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
        # the other option is to chunk into batches of 999
        # (or look up max_variable_number for our current sqlite)
        # and build a single insert for each batch with multiple VALUES like
        # insert into vector (vec) values (?), (?), (?) ... returning id;
        #
        # or I manually insert ids like range(max_id, max_id + num_vecs)
        # which will leave holes in the id column if things get deleted but that's fine

        # TODO: clean this up a bit
        # 1. reduce repeated code
        # 2. don't convert to bytes and back to numpy several times
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

        self.index = np.concat(
            [
                self.index,
                np.array(
                    [
                        (row["id"], np.frombuffer(row["vec"], dtype=self.vec_dtype))
                        for row in to_insert
                    ],
                    dtype=self.structured_dtype,
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

    def search(self, query: np.ndarray, k: int) -> list[list[dict]]:
        q_vecs = self.float32_row_vecs(query)
        # Since we're using FlatIndexIP (inner product),
        # largest "distances" will be the best matches
        distances: np.ndarray
        ids: np.ndarray
        distances, ids = self.faiss_index.search(q_vecs, k)  # type: ignore

        # If there are only 3 vectors in the index and you ask for 5,
        # FAISS will fill in the final 2 ids wth -1 in the ids array
        # like
        # [[3, 1, 2, -1, -1]]
        unique_ids = np.unique(ids).tolist()
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
                "doc": self.parse_json(row["doc"]),
            }

        # fill in a 2D list of dicts for the results
        result = []
        for i, r in enumerate(ids):
            result_row = []
            for j, id_ in enumerate(r):
                result_row.append({**unique_results[id_], "distance": distances[i][j]})
            result.append(result_row)

        return result

    def query_by_doc(self, path: list[str], value: str | int) -> list[dict]:
        json_path = "$." + ".".join(path)

        with self.connect() as con:
            rows = con.execute(
                """\
                SELECT id, vec, doc
                FROM vector
                WHERE json_extract(doc, :json_path) = :value;""",
                {"json_path": json_path, "value": value},
            ).fetchall()
        return [
            {
                "id": r["id"],
                "vec": self.blobs_to_ndarray([r["vec"]]),
                "doc": self.parse_json(r["doc"]),
            }
            for r in rows
        ]
