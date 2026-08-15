# from here
# this chat also lists some suggestions for bug fixes
# https://chatgpt.com/share/6a7e8a3b-3954-83ea-bf18-bd38dbf4370a
import os
from unittest import TestCase

import numpy as np

from vector_db_at_home import VectorStore


class TestVectorStoreAdditional(TestCase):
    def setUp(self):
        # always wipe the database, even if the integrity check fails in tearDown()
        self.addCleanup(self.cleanup_database)

        self.vs_path = "tmp_vector_test.sqlite3"
        self.vs_dim = 10
        self.vs = VectorStore(self.vs_path, self.vs_dim)
        self.assertEqual(self.vs.count(), 0)
        self.assert_indexes_consistent()

    def tearDown(self):
        super().tearDown()
        self.assert_indexes_consistent()

    def cleanup_database(self):
        if os.path.exists(self.vs_path):
            os.remove(self.vs_path)

    def assertNumpyEqual(self, a, b):
        self.assertTrue(np.array_equal(a, b))

    def assert_indexes_consistent(self):
        """
        Check that the persisted and in-memory representations contain
        the same vector IDs and LSH IDs.
        """
        with self.vs.connect() as con:
            vector_rows = con.execute("SELECT id FROM vector ORDER BY id").fetchall()

            lsh_rows = con.execute(
                "SELECT vec_id FROM lsh_idx ORDER BY vec_id"
            ).fetchall()

        db_vector_ids = [r["id"] for r in vector_rows]
        memory_vector_ids = self.vs.index["id"].tolist()

        self.assertEqual(db_vector_ids, memory_vector_ids)

        db_lsh_ids = [r["vec_id"] for r in lsh_rows]
        memory_lsh_ids = self.vs.lsh_idx["vec_id"].tolist()

        self.assertEqual(db_lsh_ids, memory_lsh_ids)

        self.assertEqual(memory_vector_ids, memory_lsh_ids)

    def set_deterministic_hyperplanes(self):
        """
        Make the LSH behavior deterministic and easy to reason about.

        Each of the first five coordinates controls one hash bit.
        """
        self.vs.lsh_hyperplanes = np.zeros(
            (self.vs_dim, self.vs.lsh_dim),
            dtype=np.float32,
        )

        for i in range(self.vs.lsh_dim):
            self.vs.lsh_hyperplanes[i, i] = 1

    # ------------------------------------------------------------------
    # Search validation
    # ------------------------------------------------------------------

    def test_search_k_zero(self):
        self.vs.insert(np.ones((3, self.vs_dim), dtype=np.float32))
        query = np.ones(self.vs_dim, dtype=np.float32)
        self.assertEqual(self.vs.search(query, k=0), [[]])

    def test_search_k_negative(self):
        self.vs.insert(np.ones((3, self.vs_dim), dtype=np.float32))
        query = np.ones(self.vs_dim, dtype=np.float32)
        self.assertEqual(self.vs.search(query, k=-1), [[]])

    def test_search_k_too_large(self):
        self.vs.insert(np.ones((3, self.vs_dim), dtype=np.float32))

        # TODO: this isn't a value error, we just return 3 things
        with self.assertRaises(ValueError):
            self.vs.search(np.ones(self.vs_dim, dtype=np.float32), k=4)

    def test_search_lsh_k_zero(self):
        self.vs.insert(np.ones((3, self.vs_dim), dtype=np.float32))
        query = np.ones(self.vs_dim, dtype=np.float32)
        self.assertEqual(self.vs.search_lsh(query, k=0), [[]])

    def test_search_lsh_k_negative(self):
        self.vs.insert(np.ones((3, self.vs_dim), dtype=np.float32))
        query = np.ones(self.vs_dim, dtype=np.float32)
        self.assertEqual(self.vs.search_lsh(query, k=-1), [[]])

    def test_search_lsh_k_too_large(self):
        self.vs.insert(np.ones((3, self.vs_dim), dtype=np.float32))

        # TODO: this isn't a value error, we just return 3 things
        with self.assertRaises(ValueError):
            self.vs.search_lsh(np.ones(self.vs_dim, dtype=np.float32), k=4)

    def test_search_empty_store(self):
        query = np.ones(self.vs_dim, dtype=np.float32)
        self.assertEqual(self.vs.search(query, k=1), [[]])

    def test_search_lsh_empty_store(self):
        query = np.ones(self.vs_dim, dtype=np.float32)
        self.assertEqual(self.vs.search_lsh(query, k=1), [[]])

    # ------------------------------------------------------------------
    # ID behavior
    # ------------------------------------------------------------------

    def test_ids_start_at_zero(self):
        self.vs.insert(np.ones((3, self.vs_dim), dtype=np.float32))

        self.assertEqual(
            self.vs.index["id"].tolist(),
            [0, 1, 2],
        )

    def test_ids_continue_across_inserts(self):
        self.vs.insert(np.ones((3, self.vs_dim), dtype=np.float32))
        self.vs.insert(np.ones((2, self.vs_dim), dtype=np.float32))

        self.assertEqual(
            self.vs.index["id"].tolist(),
            [0, 1, 2, 3, 4],
        )

    def test_deleted_ids_are_not_reused(self):
        self.vs.insert(np.ones((3, self.vs_dim), dtype=np.float32))

        self.vs.delete([1])

        self.vs.insert(np.ones((2, self.vs_dim), dtype=np.float32))

        self.assertEqual(
            self.vs.index["id"].tolist(),
            [0, 2, 3, 4],
        )

    def test_delete_nonexistent_id(self):
        self.vs.insert(np.ones((2, self.vs_dim), dtype=np.float32))

        with self.assertWarns(UserWarning):
            self.vs.delete([99])

        self.assertEqual(self.vs.count(), 2)
        self.assertEqual(
            self.vs.index["id"].tolist(),
            [0, 1],
        )

    def test_delete_existing_and_nonexistent_ids(self):
        self.vs.insert(np.ones((3, self.vs_dim), dtype=np.float32))

        with self.assertWarns(UserWarning):
            self.vs.delete([1, 99])

        self.assertEqual(self.vs.count(), 2)
        self.assertEqual(
            self.vs.index["id"].tolist(),
            [0, 2],
        )

    # ------------------------------------------------------------------
    # In-memory / SQLite consistency
    # ------------------------------------------------------------------

    def test_indexes_consistent_after_insert(self):
        self.vs.insert(np.ones((5, self.vs_dim), dtype=np.float32))

        self.assert_indexes_consistent()

    def test_indexes_consistent_after_multiple_inserts(self):
        self.vs.insert(np.ones((5, self.vs_dim), dtype=np.float32))
        self.vs.insert(np.ones((3, self.vs_dim), dtype=np.float32))

        self.assert_indexes_consistent()

    def test_indexes_consistent_after_delete(self):
        self.vs.insert(np.ones((5, self.vs_dim), dtype=np.float32))

        self.vs.delete([1, 3])

        self.assert_indexes_consistent()

    def test_indexes_consistent_after_insert_delete_insert(self):
        self.vs.insert(np.ones((5, self.vs_dim), dtype=np.float32))

        self.vs.delete([1, 3])

        self.vs.insert(np.ones((3, self.vs_dim), dtype=np.float32))

        self.assert_indexes_consistent()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def test_persistence_preserves_search_results(self):
        rng = np.random.default_rng(123)

        vecs = rng.normal(size=(50, self.vs_dim)).astype(np.float32)

        docs = [{"id": i, "text": f"document {i}"} for i in range(50)]

        self.vs.insert(vecs, docs)

        queries = rng.normal(size=(3, self.vs_dim)).astype(np.float32)

        before = self.vs.search(queries, k=5)

        new = VectorStore(self.vs_path, self.vs_dim)
        after = new.search(queries, k=5)

        self.assertEqual(
            [[r.id for r in row] for row in before],
            [[r.id for r in row] for row in after],
        )

        np.testing.assert_allclose(
            [[r.distance for r in row] for row in before],
            [[r.distance for r in row] for row in after],
            rtol=1e-6,
            atol=1e-6,
        )

    def test_persistence_preserves_vectors(self):
        rng = np.random.default_rng(123)

        vecs = rng.normal(size=(20, self.vs_dim)).astype(np.float32)

        self.vs.insert(vecs)

        new = VectorStore(self.vs_path, self.vs_dim)

        np.testing.assert_array_equal(
            self.vs.index["id"],
            new.index["id"],
        )

        np.testing.assert_array_equal(
            self.vs.index["vec"],
            new.index["vec"],
        )

    def test_persistence_preserves_lsh_index(self):
        rng = np.random.default_rng(123)

        vecs = rng.normal(size=(20, self.vs_dim)).astype(np.float32)

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

    def test_persistence_preserves_lsh_hyperplanes(self):
        """
        This is expected to fail with the current implementation because
        hyperplanes are generated again when VectorStore is constructed.

        It documents the desired persistence invariant if the LSH index
        is considered persisted state.
        """
        rng = np.random.default_rng(123)

        vecs = rng.normal(size=(20, self.vs_dim)).astype(np.float32)

        self.vs.insert(vecs)

        new = VectorStore(self.vs_path, self.vs_dim)

        np.testing.assert_array_equal(
            self.vs.lsh_hyperplanes,
            new.lsh_hyperplanes,
        )

    # ------------------------------------------------------------------
    # Exact search correctness
    # ------------------------------------------------------------------

    def test_random_search_matches_independent_brute_force(self):
        rng = np.random.default_rng(123)

        n = 100
        n_queries = 10
        k = 5

        vecs = rng.normal(size=(n, self.vs_dim)).astype(np.float32)

        queries = rng.normal(size=(n_queries, self.vs_dim)).astype(np.float32)

        self.vs.insert(vecs)

        results = self.vs.search(queries, k=k)

        self.assertEqual(len(results), n_queries)

        for query, result_row in zip(queries, results):
            distances = np.linalg.norm(
                vecs - query,
                ord=2,
                axis=1,
            )

            expected_ids = np.argsort(distances)[:k]
            expected_distances = distances[expected_ids]

            actual_ids = [r.id for r in result_row]
            actual_distances = np.array([r.distance for r in result_row])

            self.assertEqual(
                actual_ids,
                expected_ids.tolist(),
            )

            np.testing.assert_allclose(
                actual_distances,
                expected_distances,
                rtol=1e-6,
                atol=1e-6,
            )

    def test_random_search_matches_brute_force_after_deletes(self):
        rng = np.random.default_rng(42)

        n = 500

        vecs = rng.normal(size=(n, self.vs_dim)).astype(np.float32)

        self.vs.insert(vecs)

        deleted = [1, 17, 83, 200, 301]

        self.vs.delete(deleted)

        query = rng.normal(size=self.vs_dim).astype(np.float32)

        results = self.vs.search(query, k=10)[0]

        remaining_ids = np.array([i for i in range(n) if i not in deleted])

        remaining_vecs = vecs[remaining_ids]

        distances = np.linalg.norm(
            remaining_vecs - query,
            ord=2,
            axis=1,
        )

        expected_positions = np.argsort(distances)[:10]
        expected_ids = remaining_ids[expected_positions]

        self.assertEqual(
            [r.id for r in results],
            expected_ids.tolist(),
        )

    def test_search_does_not_return_deleted_vectors(self):
        rng = np.random.default_rng(123)

        vecs = rng.normal(size=(20, self.vs_dim)).astype(np.float32)

        self.vs.insert(vecs)

        deleted_id = 7
        self.vs.delete([deleted_id])

        query = vecs[deleted_id]

        results = self.vs.search(query, k=5)[0]

        self.assertNotIn(
            deleted_id,
            [r.id for r in results],
        )

    def test_search_after_delete_returns_correct_neighbor(self):
        """
        Construct an unambiguous nearest-neighbor case rather than relying
        on ties between basis vectors.
        """
        query = np.zeros(self.vs_dim, dtype=np.float32)

        close = np.zeros(self.vs_dim, dtype=np.float32)
        close[0] = 0.5

        far = np.zeros(self.vs_dim, dtype=np.float32)
        far[0] = 2.0

        self.vs.insert(np.stack([query, close, far]))

        self.vs.delete([0])

        results = self.vs.search(query, k=1)[0]

        self.assertEqual(results[0].id, 1)
        self.assertAlmostEqual(
            float(results[0].distance),
            0.5,
            places=6,
        )

    def test_search_returns_all_duplicate_vectors(self):
        vec = np.ones(
            self.vs_dim,
            dtype=np.float32,
        )

        self.vs.insert(np.stack([vec, vec, vec]))

        results = self.vs.search(vec, k=3)[0]

        self.assertEqual(
            [r.id for r in results],
            [0, 1, 2],
        )

        self.assertTrue(all(r.distance == np.float32(0) for r in results))

    def test_duplicate_vectors_keep_different_documents(self):
        vec = np.ones(
            self.vs_dim,
            dtype=np.float32,
        )

        docs = [
            {"name": "first"},
            {"name": "second"},
            {"name": "third"},
        ]

        self.vs.insert(
            np.stack([vec, vec, vec]),
            docs,
        )

        results = self.vs.search(vec, k=3)[0]

        self.assertEqual(
            [r.doc for r in results],
            docs,
        )

    # ------------------------------------------------------------------
    # Delete + ID-hole behavior
    # ------------------------------------------------------------------

    def test_search_with_id_holes(self):
        """
        Exact search should continue to work when database IDs are no
        longer contiguous.
        """
        vecs = np.eye(
            self.vs_dim,
            dtype=np.float32,
        )[:5]

        self.vs.insert(vecs)

        self.vs.delete([1])

        results = self.vs.search(vecs[3], k=1)[0]

        self.assertEqual(results[0].id, 3)
        self.assertEqual(
            results[0].distance,
            np.float32(0),
        )

    # ------------------------------------------------------------------
    # LSH hash behavior
    # ------------------------------------------------------------------

    def test_lsh_digest_is_deterministic(self):
        self.set_deterministic_hyperplanes()

        vec = np.zeros(
            self.vs_dim,
            dtype=np.float32,
        )
        vec[0] = 1
        vec[2] = 1
        vec[4] = -1

        digest1 = self.vs.lsh_digests(vec)
        digest2 = self.vs.lsh_digests(vec)

        np.testing.assert_array_equal(
            digest1,
            digest2,
        )

    def test_lsh_digest_changes_when_crossing_hyperplane(self):
        self.set_deterministic_hyperplanes()

        positive = np.zeros(
            self.vs_dim,
            dtype=np.float32,
        )
        positive[0] = 0.1

        negative = np.zeros(
            self.vs_dim,
            dtype=np.float32,
        )
        negative[0] = -0.1

        positive_digest = self.vs.lsh_digests(positive)
        negative_digest = self.vs.lsh_digests(negative)

        self.assertFalse(np.array_equal(positive_digest, negative_digest))

    def test_similar_vectors_same_lsh_digests(self):
        self.set_deterministic_hyperplanes()

        vec = np.zeros(
            self.vs_dim,
            dtype=np.float32,
        )
        vec[0] = 0.1

        same_bucket = vec.copy()
        same_bucket[0] = 0.2

        different_bucket = vec.copy()
        different_bucket[0] = -0.1

        vec_digest = self.vs.lsh_digests(vec.reshape(1, -1))[0]
        same_digest = self.vs.lsh_digests(same_bucket.reshape(1, -1))[0]
        different_digest = self.vs.lsh_digests(different_bucket.reshape(1, -1))[0]

        self.assertNumpyEqual(vec_digest, same_digest)
        self.assertFalse(np.array_equal(vec_digest, different_digest))

    def test_lsh_prunes_different_bucket(self):
        self.set_deterministic_hyperplanes()

        query = np.zeros(
            self.vs_dim,
            dtype=np.float32,
        )
        query[0] = 0.1

        same_bucket = query.copy()
        same_bucket[0] = 0.2

        different_bucket = query.copy()
        different_bucket[0] = -0.1

        self.vs.insert(
            np.stack(
                [
                    query,
                    same_bucket,
                    different_bucket,
                ]
            ),
            [
                {"name": "query"},
                {"name": "same"},
                {"name": "different"},
            ],
        )

        results = self.vs.search_lsh(
            query,
            k=3,
        )[0]

        self.assertEqual(
            {r.doc["name"] for r in results},
            {"query", "same"},
        )

        self.assertNotIn(
            "different",
            [r.doc["name"] for r in results],
        )

    def test_lsh_results_are_sorted_by_distance(self):
        self.set_deterministic_hyperplanes()

        query = np.zeros(
            self.vs_dim,
            dtype=np.float32,
        )
        query[0] = 0.5

        v1 = query.copy()
        v1[0] = 0.3

        v2 = query.copy()
        v2[0] = 0.1

        v3 = query.copy()
        v3[0] = 0.2

        self.vs.insert(np.stack([v1, v2, v3]))

        results = self.vs.search_lsh(
            query,
            k=3,
        )[0]

        self.assertEqual(
            [r.id for r in results],
            [0, 2, 1],
        )

        np.testing.assert_allclose(
            [r.distance for r in results],
            sorted([0.5 - 0.1, 0.5 - 0.2, 0.5 - 0.3]),
            rtol=1e-6,
            atol=1e-6,
        )

    # ------------------------------------------------------------------
    # LSH + deletion
    # ------------------------------------------------------------------

    def test_delete_removes_lsh_index(self):
        vecs = np.eye(
            self.vs_dim,
            dtype=np.float32,
        )[:3]

        self.vs.insert(vecs)

        self.assertEqual(
            len(self.vs.lsh_idx),
            3,
        )

        self.vs.delete([1])

        self.assertEqual(
            len(self.vs.lsh_idx),
            2,
        )

        self.assertEqual(
            self.vs.lsh_idx["vec_id"].tolist(),
            [0, 2],
        )

        self.assert_indexes_consistent()

    def test_lsh_search_after_delete_with_id_hole(self):
        """
        This specifically checks that an LSH vec_id is treated as a
        database ID rather than as a positional index into self.index.

        With IDs [0, 2, 3, 4], self.index[3] is ID 4, not ID 3.
        """
        self.set_deterministic_hyperplanes()

        vecs = np.eye(
            self.vs_dim,
            dtype=np.float32,
        )[:5]

        self.vs.insert(vecs)

        self.vs.delete([1])

        query = vecs[3]

        results = self.vs.search_lsh(
            query,
            k=1,
        )

        self.assertEqual(
            len(results),
            1,
        )

        self.assertEqual(
            results[0][0].id,
            3,
        )

        self.assertEqual(
            results[0][0].distance,
            np.float32(0),
        )

    # ------------------------------------------------------------------
    # LSH + persistence
    # ------------------------------------------------------------------

    def test_persisted_lsh_hashes_match_original(self):
        rng = np.random.default_rng(123)

        vecs = rng.normal(size=(20, self.vs_dim)).astype(np.float32)

        self.vs.insert(vecs)

        new = VectorStore(
            self.vs_path,
            self.vs_dim,
        )

        np.testing.assert_array_equal(
            self.vs.lsh_idx["vec_id"],
            new.lsh_idx["vec_id"],
        )

        np.testing.assert_array_equal(
            self.vs.lsh_idx["hash"],
            new.lsh_idx["hash"],
        )

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------

    def test_insert_without_doc_returns_empty_dict(self):
        self.vs.insert(
            np.ones(
                self.vs_dim,
                dtype=np.float32,
            )
        )

        result = self.vs.select_ids([0])[0]

        self.assertEqual(
            result.doc,
            {},
        )

    def test_nested_doc_round_trip(self):
        doc = {
            "user": {
                "name": "alice",
                "metadata": {
                    "age": 30,
                },
            },
            "tags": ["a", "b"],
        }

        self.vs.insert(
            np.ones(
                self.vs_dim,
                dtype=np.float32,
            ),
            [doc],
        )

        result = self.vs.select_ids([0])[0]

        self.assertEqual(
            result.doc,
            doc,
        )

    def test_query_by_doc_no_matches(self):
        docs = [
            {"type": "a"},
            {"type": "b"},
            {"type": "c"},
        ]

        self.vs.insert(
            np.ones((3, self.vs_dim), dtype=np.float32),
            docs,
        )

        result = self.vs.query_by_doc(
            ["type"],
            ["z"],
        )

        self.assertEqual(
            result,
            [],
        )

    def test_query_by_doc_multiple_matches(self):
        docs = [
            {"type": "a"},
            {"type": "b"},
            {"type": "a"},
        ]

        self.vs.insert(
            np.ones((3, self.vs_dim), dtype=np.float32),
            docs,
        )

        result = self.vs.query_by_doc(
            ["type"],
            ["a"],
        )

        self.assertEqual(
            [r.id for r in result],
            [0, 2],
        )

    def test_query_by_doc_nested_path(self):
        docs = [
            {"user": {"name": "alice"}},
            {"user": {"name": "bob"}},
        ]

        self.vs.insert(
            np.ones((2, self.vs_dim), dtype=np.float32),
            docs,
        )

        result = self.vs.query_by_doc(
            ["user", "name"],
            ["bob"],
        )

        self.assertEqual(
            [r.id for r in result],
            [1],
        )

    # ------------------------------------------------------------------
    # head()
    # ------------------------------------------------------------------

    def test_head_larger_than_count(self):
        self.vs.insert(np.ones((3, self.vs_dim), dtype=np.float32))

        result = self.vs.head(100)

        self.assertEqual(
            len(result),
            3,
        )

    # ------------------------------------------------------------------
    # Mutation sequence
    # ------------------------------------------------------------------

    def test_insert_delete_insert_persist(self):
        rng = np.random.default_rng(42)

        a = rng.normal(size=(10, self.vs_dim)).astype(np.float32)

        b = rng.normal(size=(5, self.vs_dim)).astype(np.float32)

        c = rng.normal(size=(7, self.vs_dim)).astype(np.float32)

        self.vs.insert(a)
        self.vs.delete([2, 5, 8])
        self.vs.insert(b)
        self.vs.delete([0, 12])
        self.vs.insert(c)

        new = VectorStore(
            self.vs_path,
            self.vs_dim,
        )

        np.testing.assert_array_equal(
            self.vs.index["id"],
            new.index["id"],
        )

        np.testing.assert_array_equal(
            self.vs.index["vec"],
            new.index["vec"],
        )

    def test_indexes_consistent_after_random_mutations(self):
        rng = np.random.default_rng(42)

        next_vectors = []

        for _ in range(10):
            n = int(rng.integers(1, 10))

            vecs = rng.normal(size=(n, self.vs_dim)).astype(np.float32)

            self.vs.insert(vecs)

            self.assert_indexes_consistent()

            next_vectors.extend(vecs)

        # Delete some currently existing IDs.
        existing_ids = self.vs.index["id"].tolist()
        ids_to_delete = existing_ids[::3]

        self.vs.delete(ids_to_delete)

        self.assert_indexes_consistent()

        # Insert again after deletion.
        vecs = rng.normal(size=(10, self.vs_dim)).astype(np.float32)

        self.vs.insert(vecs)

        self.assert_indexes_consistent()

    # ------------------------------------------------------------------
    # SQLite-specific sanity checks
    # ------------------------------------------------------------------

    def test_vector_and_lsh_row_counts_match(self):
        vecs = (
            np.random.default_rng(123).normal(size=(25, self.vs_dim)).astype(np.float32)
        )

        self.vs.insert(vecs)

        with self.vs.connect() as con:
            vector_count = con.execute("SELECT count(*) FROM vector").fetchone()[0]

            lsh_count = con.execute("SELECT count(*) FROM lsh_idx").fetchone()[0]

        self.assertEqual(vector_count, 25)
        self.assertEqual(lsh_count, 25)

    def test_vector_ids_and_lsh_ids_match_in_sqlite(self):
        vecs = (
            np.random.default_rng(123).normal(size=(25, self.vs_dim)).astype(np.float32)
        )

        self.vs.insert(vecs)

        with self.vs.connect() as con:
            vector_ids = [
                row["id"] for row in con.execute("SELECT id FROM vector ORDER BY id")
            ]

            lsh_ids = [
                row["vec_id"]
                for row in con.execute("SELECT vec_id FROM lsh_idx ORDER BY vec_id")
            ]

        self.assertEqual(
            vector_ids,
            lsh_ids,
        )

    # ------------------------------------------------------------------
    # A small randomized state-machine-ish test
    # ------------------------------------------------------------------

    def test_random_insert_delete_search_sequence(self):
        """
        This isn't a full property-based test, but it exercises the store
        through a sequence of mutations and checks exact search against an
        independent implementation after each mutation.
        """
        rng = np.random.default_rng(9876)

        expected = {}

        for _ in range(20):
            # Insert a small random batch.
            n = int(rng.integers(1, 6))

            vecs = rng.normal(size=(n, self.vs_dim)).astype(np.float32)

            start_id = 0 if not expected else max(expected) + 1

            self.vs.insert(vecs)

            for offset, vec in enumerate(vecs):
                expected[start_id + offset] = vec

            # Sometimes delete one existing ID.
            if expected and rng.random() < 0.5:
                ids = list(expected)
                delete_id = ids[int(rng.integers(0, len(ids)))]

                self.vs.delete([delete_id])
                del expected[delete_id]

            # Check the count.
            self.assertEqual(
                self.vs.count(),
                len(expected),
            )

            # Check the exact index IDs.
            self.assertEqual(
                sorted(self.vs.index["id"].tolist()),
                sorted(expected),
            )

            # If there are vectors, compare a random query against our
            # independent brute-force calculation.
            if expected:
                query = rng.normal(size=self.vs_dim).astype(np.float32)

                k = min(3, len(expected))

                results = self.vs.search(
                    query,
                    k=k,
                )[0]

                ids = np.array(
                    list(expected.keys()),
                    dtype=np.int64,
                )

                vectors = np.stack([expected[i] for i in ids])

                distances = np.linalg.norm(
                    vectors - query,
                    ord=2,
                    axis=1,
                )

                expected_positions = np.argsort(distances)[:k]
                expected_ids = ids[expected_positions]

                self.assertEqual(
                    [r.id for r in results],
                    expected_ids.tolist(),
                )
