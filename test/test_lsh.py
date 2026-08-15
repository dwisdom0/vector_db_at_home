import os
from unittest import TestCase

import numpy as np

from test.common import assertNumpyEqual
from vector_db_at_home import VectorStore


class TestLSH(TestCase):
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

    def test_ann_search(self):
        v1 = [0] * self.vs_dim
        v1[0] = 1
        v1 = np.array(v1, dtype=np.float32)
        self.vs.insert(v1, [{"name": "v1"}])

        v_close = [0] * self.vs_dim
        v_close[0] = 1
        v_close[1] = 0.1
        v_close = np.array(v_close, dtype=np.float32)
        self.vs.insert(v_close, [{"name": "v_close"}])

        v_far = [1] * self.vs_dim
        v_far = np.array(v_far, dtype=np.float32)
        self.vs.insert(v_far, [{"name": "v_far"}])

        results = self.vs.search_lsh(v1, k=3)

        # [[SearchRecord(id=0,
        #        vec=array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32),
        #        doc={'name': 'v1'},
        #        distance=np.float32(0.0)),
        # SearchRecord(id=1,
        #        vec=array([1. , 0.1, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ], dtype=float32),
        #        doc={'name': 'v_close'},
        #        distance=np.float32(0.1))]]

        # only had 1 query vector
        self.assertEqual(1, len(results))
        # LSH should've pruned out the vector that's really far away
        # (v_far)
        # leaving only 2 vectors left to get returned as results
        self.assertEqual(2, len(results[0]))

        # our search query was a vector that was already in the db
        # so it should come back as the best match
        self.assertEqual("v1", results[0][0].doc["name"])
        assertNumpyEqual(v1, results[0][0].vec)

        # the next closest vector should be v_close
        self.assertEqual("v_close", results[0][1].doc["name"])
        assertNumpyEqual(v_close, results[0][1].vec)

    def test_lsh_search_after_delete_with_id_hole(self):
        vecs = np.eye(self.vs_dim, dtype=np.float32)[:5]
        self.vs.insert(vecs)

        self.vs.delete([1])

        query = vecs[3]
        results = self.vs.search_lsh(query, k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0].id, 3)
        self.assertEqual(results[0][0].distance, np.float32(0))

    def test_persistence_preserves_lsh_index(self):
        rng = np.random.default_rng(123)

        vecs = rng.normal(size=(50, self.vs_dim)).astype(np.float32)
        self.vs.insert(vecs)

        new = VectorStore(self.vs_path, self.vs_dim)

        np.testing.assert_array_equal(
            self.vs.lsh_idx["vec_id"],
            new.lsh_idx["vec_id"],
        )

        np.testing.assert_array_equal(
            self.vs.lsh_idx["hash"],
            new.lsh_idx["hash"],
        )
