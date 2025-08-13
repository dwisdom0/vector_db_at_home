import os
import numpy as np
from vector_store import VectorStore
from unittest import TestCase


class TestVectorStore(TestCase):
    def setUp(self):
        # unit tests don't persist the schema if I use an in-memory sqlite db
        # it seems like SQLAlchemy can do this with SQLite using a Session
        # so theres's probably some way to do it

        self.vs_path = "tmp_vector_test.sqlite3"
        self.vs_dim = 10
        self.vs = VectorStore(self.vs_path, self.vs_dim)
        self.assertEqual(self.vs.count(), 0)
        self.assertIsNone(self.vs.head(0))

    def tearDown(self):
        super().tearDown()
        os.remove(self.vs_path)

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
        ) - set((self.vs.numpy_dtype,))

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
        ) - set((self.vs.numpy_dtype,))

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
        self.assertIsNone(self.vs.head(0))

    def test_head_1(self):
        a = np.ones((self.vs_dim,), dtype=np.float32)
        self.vs.insert(a)
        self.assertEqual(self.vs.count(), 1)
        head_result = self.vs.head(1)
        self.assertIsNotNone(head_result)
        self.assertTrue(np.array_equal(head_result, a.reshape(-1, self.vs_dim)))  # type: ignore

    def test_head_5(self):
        size = 5
        a = np.ones((size, self.vs_dim), dtype=np.float32)
        self.vs.insert(a)
        self.assertEqual(self.vs.count(), size)
        head_result = self.vs.head(size)
        self.assertIsNotNone(head_result)
        self.assertTrue(np.array_equal(head_result, a))  # type: ignore

    def test_search(self):
        a = np.eye(self.vs_dim, dtype=np.float32)
        self.vs.insert(a)
        self.assertEqual(self.vs.count(), self.vs_dim)

        query = np.array([0, 0, 0, 0.5, 0, 0, 0, 0, 0, 1], dtype=np.float32).reshape(
            -1, self.vs_dim
        )

        similarities, ids = self.vs.search(query, k=2)
        # the best matching vector should be the 10th basis vector and then the 4th basis vector
        # but 0-based indexing
        self.assertTrue(np.array_equal(ids, np.array([[9, 3]], dtype=np.int64)))
        # these are the cosine similarities
        # this is kind of misleading since I'm not normalizing the vectors at all
        self.assertTrue(
            np.array_equal(similarities, np.array([[1.0, 0.5]], dtype=np.float32))
        )

    def test_load_from_existing(self):
        size = 5
        a = np.ones((size, self.vs_dim), dtype=np.float32)
        self.vs.insert(a)

        # make a new store using the sqlite db used by self.vs
        new = VectorStore(self.vs_path, self.vs_dim)
        self.assertEqual(new.count(), size)
        self.assertTrue(np.array_equal(new.head(size), np.ones((size, self.vs_dim))))  # type: ignore

    # TODO:
    # VectorStore.remove()
    # does remove() mess up the id column in sqlite?
    # Add basis vectors, search, then remove some and search again
