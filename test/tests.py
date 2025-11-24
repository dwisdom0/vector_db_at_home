import os
import sqlite3
from unittest import TestCase

import numpy as np

from vector_db_at_home.vector_store import VectorStore


class TestVectorStore(TestCase):
    def setUp(self):
        # unit tests don't persist the schema if I use an in-memory sqlite db
        # it seems like SQLAlchemy can do this with SQLite using a Session
        # so theres's probably some way to do it

        self.vs_path = "tmp_vector_test.sqlite3"
        self.vs_dim = 10
        self.vs = VectorStore(self.vs_path, self.vs_dim)
        self.assertEqual(self.vs.count(), 0)

    def tearDown(self):
        os.remove(self.vs_path)
        super().tearDown()

    @staticmethod
    def gen_docs(ns: list) -> list[dict]:
        return [{f"k{n}": f"v{n}"} for n in ns]

    def assertNumpyEqual(self, a, b):
        return self.assertTrue(np.array_equal(a, b))

    def test_insert_1(self):
        arr = np.ones((self.vs_dim,), dtype=np.float32)
        self.vs.insert(arr)
        self.assertEqual(self.vs.count(), 1)

    def test_insert_5(self):
        arr = np.ones((5, self.vs_dim), dtype=np.float32)
        self.vs.insert(arr)
        self.assertEqual(self.vs.count(), 5)

    def test_multiple_inserts(self):
        loops = 3
        size = 3
        for _ in range(loops):
            a = np.ones((size, self.vs_dim), dtype=np.float32)
            self.vs.insert(a)
        self.assertEqual(self.vs.count(), loops * size)

    def test_insert_bad_shape(self):
        a = np.ones((self.vs_dim + 1,), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.vs.insert(a)
        self.assertEqual(self.vs.count(), 0)

    def test_insert_many_bad_shape(self):
        a = np.ones((5, self.vs_dim + 1), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.vs.insert(a)
        self.assertEqual(self.vs.count(), 0)

    def test_insert_most_dtypes(self):
        total = 0

        working_dtypes = set(
            (
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
        ) - set((self.vs.vec_dtype,))

        for dtype in working_dtypes:
            self.assertEqual(self.vs.count(), total)
            a = np.ones((self.vs_dim), dtype=dtype)
            with self.assertWarns(UserWarning):
                self.vs.insert(a)
            total += 1
            self.assertEqual(self.vs.count(), total)

        non_working_dtypes = set(
            (
                np.void,
                np.str_,
                np.complex64,
                np.complex128,
                np.bytes_,
                np.object_,
            )
        )

        for dtype in non_working_dtypes:
            a = np.ones((self.vs_dim), dtype=dtype)
            with self.assertRaises(ValueError):
                self.vs.insert(a)

        self.assertEqual(self.vs.count(), total)

    def test_insert_many_most_dtypes(self):
        total = 0

        working_dtypes = set(
            (
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
        ) - set((self.vs.vec_dtype,))

        num_vecs = 3

        for dtype in working_dtypes:
            self.assertEqual(self.vs.count(), total)
            a = np.ones((num_vecs, self.vs_dim), dtype=dtype)
            with self.assertWarns(UserWarning):
                self.vs.insert(a)
            total += num_vecs
            self.assertEqual(self.vs.count(), total)

        non_working_dtypes = set(
            (
                np.void,
                np.str_,
                np.complex64,
                np.complex128,
                np.bytes_,
                np.object_,
            )
        )

        for dtype in non_working_dtypes:
            a = np.ones((num_vecs, self.vs_dim), dtype=dtype)
            with self.assertRaises(ValueError):
                self.vs.insert(a)

        self.assertEqual(self.vs.count(), total)

    def test_head_0(self):
        self.assertEqual(self.vs.head(0), [])

    def test_head_1(self):
        a = np.ones((self.vs_dim,), dtype=np.float32)
        docs = [{"k1": "v1"}]
        self.vs.insert(a, docs)
        self.assertEqual(self.vs.count(), 1)
        head_result = self.vs.head(1)
        self.assertIsNotNone(head_result)
        self.assertEqual(len(head_result), 1)

        # can't just make the dict and do assertEqual
        # b/c it will have a numpy array
        self.assertEqual(head_result[0]["id"], 0)
        self.assertNumpyEqual(head_result[0]["vec"], a.reshape(-1, self.vs_dim))
        self.assertEqual(head_result[0]["doc"], docs[0])

    def test_head_5(self):
        size = 5
        a = np.ones((size, self.vs_dim), dtype=np.float32)
        docs = self.gen_docs(list(range(size)))
        self.vs.insert(a, docs)
        self.assertEqual(self.vs.count(), size)
        head_result = self.vs.head(size)
        self.assertIsNotNone(head_result)
        self.assertEqual(len(head_result), size)

        for i, res in enumerate(head_result):
            self.assertEqual(res["id"], i)
            self.assertNumpyEqual(res["vec"], a[i].reshape(-1, self.vs_dim))
            self.assertEqual(res["doc"], docs[i])

    def test_search(self):
        a = np.eye(self.vs_dim, dtype=np.float32)
        docs = self.gen_docs(list(range(self.vs_dim)))
        self.vs.insert(a, docs)
        self.assertEqual(self.vs.count(), self.vs_dim)

        query = np.array([0, 0, 0, 0.5, 0, 0, 0, 0, 0, 1], dtype=np.float32).reshape(
            -1, self.vs_dim
        )

        search_results = self.vs.search(query, k=2)
        # we had 1 query
        self.assertEqual(len(search_results), 1)
        # we asked for 2 results
        self.assertEqual(len(search_results[0]), 2)

        # the 10th basis vector should be the best match
        # the next best match should be the 4th basis vector
        for i, bv in enumerate([9, 3]):
            self.assertEqual(search_results[0][i].id, bv)
            self.assertNumpyEqual(
                search_results[0][i].vec,
                np.array(
                    [0] * bv + [1] + [0] * (self.vs_dim - bv - 1), dtype=np.float32
                ),
            )
            self.assertEqual(search_results[0][i].doc, {f"k{bv}": f"v{bv}"})

        self.assertEqual(search_results[0][0].distance, np.float32(0.5))
        self.assertEqual(search_results[0][1].distance, np.float32(1.118034))

    def test_search_many_queries(self):
        a = np.eye(self.vs_dim, dtype=np.float32)
        docs = self.gen_docs(list(range(self.vs_dim)))
        self.vs.insert(a, docs)
        self.assertEqual(self.vs.count(), self.vs_dim)

        qs = np.array(
            [
                [1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0.5, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )

        search_results = self.vs.search(qs, k=2)
        # we had 2 queries
        self.assertEqual(len(search_results), 2)
        # we asked for 2 results
        self.assertEqual(len(search_results[0]), 2)
        self.assertEqual(len(search_results[1]), 2)

        # first query
        # the best match should be the first basis vector
        # the next best should be the second basis vector
        for i, bv in enumerate([0, 1]):
            self.assertEqual(search_results[0][i].id, bv)
            self.assertNumpyEqual(
                search_results[0][i].vec,
                np.array(
                    [0] * bv + [1] + [0] * (self.vs_dim - bv - 1), dtype=np.float32
                ),
            )
            self.assertEqual(search_results[0][i].doc, {f"k{bv}": f"v{bv}"})
        self.assertEqual(search_results[0][0].distance, np.float32(0.5))
        self.assertEqual(search_results[0][1].distance, np.float32(1.118034))

        # second query
        # the best match should be the second basis vector
        # the next best should be the third basis vector
        for i, bv in enumerate([1, 2]):
            self.assertEqual(search_results[1][i].id, bv)
            self.assertNumpyEqual(
                search_results[1][i].vec,
                np.array(
                    [0] * bv + [1] + [0] * (self.vs_dim - bv - 1), dtype=np.float32
                ),
            )
            self.assertEqual(search_results[1][i].doc, {f"k{bv}": f"v{bv}"})
        self.assertEqual(search_results[1][0].distance, np.float32(0.5))
        self.assertEqual(search_results[1][1].distance, np.float32(1.118034))

    def test_load_from_existing(self):
        size = 5
        a = np.ones((size, self.vs_dim), dtype=np.float32)
        docs = self.gen_docs(list(range(size)))
        self.vs.insert(a, docs)

        # make a new store using the sqlite db used by self.vs
        new = VectorStore(self.vs_path, self.vs_dim)
        self.assertEqual(new.count(), size)

        head_result = new.head(size)
        self.assertIsNotNone(head_result)
        self.assertEqual(len(head_result), size)

        for i, res in enumerate(head_result):
            self.assertEqual(res["id"], i)
            self.assertNumpyEqual(res["vec"], a[i].reshape(-1, self.vs_dim))
            self.assertEqual(res["doc"], docs[i])

    def test_insert_doc(self):
        a = np.ones((self.vs_dim,), dtype=np.float32)
        docs = self.gen_docs([0])
        self.vs.insert(a, docs)
        self.assertEqual(self.vs.count(), 1)

        search_results = self.vs.search(a, k=1)
        self.assertEqual(len(search_results), 1)
        self.assertEqual(len(search_results[0]), 1)

        self.assertEqual(search_results[0][0].id, 0)
        self.assertNumpyEqual(search_results[0][0].vec, a)
        self.assertEqual(search_results[0][0].doc, {"k0": "v0"})
        self.assertEqual(search_results[0][0].distance, np.float32(0))

    def test_insert_many_docs(self):
        size = 5
        docs = self.gen_docs(list(range(size)))
        a = np.ones((size, self.vs_dim), dtype=np.float32)
        self.vs.insert(a, docs)
        self.assertEqual(self.vs.count(), size)

        search_results = self.vs.search(a[0], k=size)
        # only had 1 query vector
        self.assertEqual(len(search_results), 1)
        # asked for size results
        self.assertEqual(len(search_results[0]), size)

        found = set()
        for i in range(size):
            for result in search_results[0]:
                if result.id != i:
                    continue
                self.assertTrue(i not in found)
                found.add(i)
                self.assertEqual(result.id, i)
                self.assertNumpyEqual(
                    result.vec, np.ones((self.vs_dim,), dtype=np.float32)
                )
                self.assertEqual(result.doc, {f"k{i}": f"v{i}"})
                self.assertEqual(result.distance, np.float32(0))

        self.assertEqual(found, set(list(range(size))))

    def test_delete(self):
        a = np.ones((self.vs_dim), dtype=np.float32)
        docs = self.gen_docs([0])
        self.vs.insert(a, docs)
        self.assertEqual(self.vs.count(), 1)

        self.vs.delete(ids=[0])

        self.assertEqual(self.vs.count(), 0)
        self.assertEqual(self.vs.head(), [])

    def test_delete_many(self):
        size = 5
        a = np.ones((size, self.vs_dim), dtype=np.float32)
        docs = self.gen_docs(list(range(size)))
        self.vs.insert(a, docs)
        self.assertEqual(self.vs.count(), size)

        self.vs.delete(ids=list(range(size)))

        self.assertEqual(self.vs.count(), 0)
        self.assertEqual(self.vs.head(), [])

    def test_delete_subset(self):
        size = 5
        del_size = 3
        a = np.ones((size, self.vs_dim), dtype=np.float32)
        docs = self.gen_docs(list(range(size)))
        self.vs.insert(a, docs)
        self.assertEqual(self.vs.count(), size)

        self.vs.delete(list(range(size - 1, size - del_size - 1, -1)))

        self.assertEqual(self.vs.count(), size - del_size)

        head_result = self.vs.head()
        for i, record in enumerate(head_result):
            self.assertEqual(record["id"], i)
            self.assertNumpyEqual(
                record["vec"], np.ones((1, self.vs_dim), dtype=np.float32)
            )
            self.assertEqual(record["doc"], {f"k{i}": f"v{i}"})

    def test_delete_mess_up_ids(self):
        # try to mess up the ids by deleting things from the middle of the id range

        # insert 5
        self.vs.insert(
            np.ones((5, self.vs_dim), dtype=np.float32), self.gen_docs(list(range(5)))
        )
        self.assertEqual(5, self.vs.count())
        # delete id 2, which will be in the middle
        self.vs.delete([2])
        self.assertEqual(self.vs.count(), 4)
        # insert again
        self.vs.insert(
            np.ones((3, self.vs_dim), dtype=np.float32),
            self.gen_docs(list(range(5, 8))),
        )
        self.assertEqual(self.vs.count(), 7)

        # check that the ids in SQLite match up with the ids in the NumPy index
        # should be 0, 1, 3, 4, 5, 6, 7
        # missing 2 in the middle there
        db_ids = sorted([r["id"] for r in self.vs.head(10)])
        self.assertEqual([0, 1, 3, 4, 5, 6, 7], db_ids)
        index_ids = sorted(self.vs.index["id"].flatten().tolist())
        self.assertEqual(db_ids, index_ids)

    def test_delete_too_many(self):
        # delete more than SQLITE_MAX_VARIABLE_NUMBER
        with self.vs.connect() as con:
            rows = con.execute("pragma compile_options;").fetchall()
            opts = [r["compile_options"] for r in rows]
        opt_name = "MAX_VARIABLE_NUMBER="
        max_vars = None
        for opt in opts:
            if opt.startswith(opt_name):
                max_vars = int(opt.replace(opt_name, ""))
        self.assertIsNotNone(max_vars)
        assert max_vars is not None

        ids = list(range(max_vars + 1))
        with self.assertRaises(sqlite3.OperationalError):
            self.vs.delete(ids)

    def test_insert_dicts_not_seralizable(self):
        data = [{"vec": np.ones((1, self.vs_dim)), "doc": list}]
        with self.assertRaises(TypeError):
            self.vs.insert_dicts(data)

    def test_insert_dict(self):
        data = [{"vec": np.ones((1, self.vs_dim), dtype=np.float32)}]
        self.vs.insert_dicts(data)
        self.assertEqual(self.vs.count(), 1)

    def test_insert_dicts(self):
        size = 5
        data = [
            {
                "vec": np.ones((1, self.vs_dim), dtype=np.float32),
            }
            for _ in range(size)
        ]
        self.vs.insert_dicts(data)
        self.assertEqual(self.vs.count(), 5)

    def test_insert_dict_doc(self):
        data = [
            {"vec": np.ones((1, self.vs_dim), dtype=np.float32), "doc": {"k0": "v0"}}
        ]
        self.vs.insert_dicts(data)
        self.assertEqual(self.vs.count(), 1)

    def test_insert_dicts_doc(self):
        size = 5
        data = [
            {
                "vec": np.ones((1, self.vs_dim), dtype=np.float32),
                "doc": {f"k{i}": f"v{i}"},
            }
            for i in range(size)
        ]
        self.vs.insert_dicts(data)
        self.assertEqual(self.vs.count(), 5)

    def test_query_doc(self):
        self.vs.insert(
            np.ones((3, self.vs_dim), dtype=np.float32), self.gen_docs(list(range(3)))
        )
        results = self.vs.query_by_doc(["k1"], ["v1"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, 1)
        self.assertNumpyEqual(
            results[0].vec, np.ones((1, self.vs_dim), dtype=np.float32)
        )
        self.assertEqual(results[0].doc, {"k1": "v1"})

    def test_query_many_doc(self):
        docs = [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]
        vecs = np.ones((len(docs), self.vs_dim), dtype=np.float32)
        self.vs.insert(vecs, docs)
        results = self.vs.query_by_doc(["id"], [2, 4])
        self.assertEqual(len(results), 2)
        docs = [r.doc for r in results]
        self.assertIn({"id": 2}, docs)
        self.assertIn({"id": 4}, docs)

    def test_repr(self):
        self.assertEqual(
            str(self.vs), f"VectorStore(db_path={self.vs_path}, dim={self.vs_dim})"
        )

    def test_select_ids(self):
        self.vs.insert(
            np.ones((5, self.vs_dim), dtype=np.float32), self.gen_docs(list(range(5)))
        )
        ids = [1, 3]
        result = self.vs.select_ids(ids)
        self.assertEqual(len(result), 2)

        for i, id_ in enumerate(ids):
            self.assertEqual(result[i].id, id_)
            self.assertNumpyEqual(
                result[i].vec, np.ones((1, self.vs_dim), dtype=np.float32)
            )
            self.assertEqual(result[i].doc, {f"k{id_}": f"v{id_}"})

    def test_select_bad_id(self):
        self.vs.insert(
            np.ones((3, self.vs_dim), dtype=np.float32), self.gen_docs(list(range(3)))
        )
        result = self.vs.select_ids([6])
        self.assertEqual(len(result), 0)
        self.assertEqual(result, [])

    def test_dump_vecs(self):
        self.vs.insert(
            np.ones((3, self.vs_dim), dtype=np.float32), self.gen_docs(list(range(3)))
        )
        result = self.vs.dump_vecs()
        self.assertNumpyEqual(result, np.ones((3, self.vs_dim)))

    def test_dump_vecs_order(self):
        self.vs.insert(
            np.array(
                [[0] * self.vs_dim, [1] * self.vs_dim, [2] * self.vs_dim],
                dtype=np.float32,
            )
        )
        self.vs.delete([1])
        r1 = self.vs.dump_vecs()
        self.assertNumpyEqual(
            r1, np.array([[0] * self.vs_dim, [2] * self.vs_dim], dtype=np.float32)
        )
        self.vs.insert(np.array([3] * self.vs_dim, dtype=np.float32))
        r2 = self.vs.dump_vecs()
        self.assertNumpyEqual(
            r2,
            np.array(
                [[0] * self.vs_dim, [2] * self.vs_dim, [3] * self.vs_dim],
                dtype=np.float32,
            ),
        )

    def test_search_by_doc(self):
        size = 5
        a = np.ones((size, self.vs_dim), dtype=np.float32)
        docs = self.gen_docs(list(range(size)))
        self.vs.insert(a, docs)

        res = self.vs.search_by_doc([{"1": "1"}], k=size)
        self.assertEqual(len(res), 1)
        self.assertEqual(len(res[0]), size)

        # the best search result should be {"k1": "v1"}
        self.assertEqual(res[0][0].id, 1)
        self.assertNumpyEqual(
            res[0][0].vec, np.ones((1, self.vs_dim), dtype=np.float32).flatten()
        )
        self.assertEqual(res[0][0].doc, {"k1": "v1"})
        self.assertEqual(res[0][0].distance, 9.090909090909093)

        for i in range(1, size):
            self.assertNotEqual(res[0][i], {"k1": "v1"})
            self.assertEqual(res[0][i].distance, 27.272727272727266)

    def test_search_by_doc_multiple_queries(self):
        size = 5
        a = np.ones((size, self.vs_dim), dtype=np.float32)
        docs = self.gen_docs(list(range(size)))
        self.vs.insert(a, docs)

        res = self.vs.search_by_doc([{"k1": ""}, "v4"], k=3)
        self.assertEqual(len(res), 2)
        self.assertEqual(len(res[0]), 3)
        self.assertEqual(len(res[1]), 3)

        # check the results for the first query
        self.assertEqual(res[0][0].id, 1)
        self.assertNumpyEqual(
            res[0][0].vec, np.ones((1, self.vs_dim), dtype=np.float32).flatten()
        )
        self.assertEqual(res[0][0].doc, {"k1": "v1"})
        self.assertEqual(res[0][0].distance, 9.090909090909093)

        self.assertNotEqual(res[0][1].doc, {"k1": "v1"})
        self.assertEqual(res[0][1].distance, 18.181818181818187)

        # check the results for the second query
        self.assertEqual(res[1][0].id, 4)
        self.assertNumpyEqual(
            res[1][0].vec, np.ones((1, self.vs_dim), dtype=np.float32).flatten()
        )
        self.assertEqual(res[1][0].doc, {"k4": "v4"})
        self.assertEqual(res[1][0].distance, 71.42857142857143)

        self.assertNotEqual(res[1][1].doc, {"k4": "v4"})
        self.assertEqual(res[1][1].distance, 85.71428571428571)

    def test_search_by_doc_zero_results(self):
        size = 5
        a = np.ones((size, self.vs_dim), dtype=np.float32)
        docs = self.gen_docs(list(range(size)))
        self.vs.insert(a, docs)

        with self.assertRaises(ValueError):
            _ = self.vs.search_by_doc([{"1": "1"}], k=0)

    def test_search_by_doc_too_many_results(self):
        size = 5
        a = np.ones((size, self.vs_dim), dtype=np.float32)
        docs = self.gen_docs(list(range(size)))
        self.vs.insert(a, docs)

        res = self.vs.search_by_doc([{"1": "1"}], k=size + 1)
        self.assertEqual(len(res), 1)
        self.assertEqual(len(res[0]), 5)
