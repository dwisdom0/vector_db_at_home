Mom, can we have `milvus-lite`?

No, we have an embedded vector database at home.


# Quickstart
```python
from pprint import pprint

import numpy as np

from vector_db_at_home import VectorStore

docs = [
    {"name": "Alice", "ice_cream_flavor": "Apricot"},
    {"name": "Bob", "ice_cream_flavor": "Banana Nut"},
    {"name": "Christy", "ice_cream_flavor": "Chocolate Chip Cookie Dough"},
    {"name": "Devin", "ice_cream_flavor": "Dulce de Leche"},
]

# usually you would use vectors from an embedding model
# or have some other way to create meaningful vectors
vector_dim = 10
vecs = np.random.rand(len(docs), vector_dim)

vs = VectorStore(db_path="my_vector_store.sqlite3", dim=vector_dim)

vs.insert(vecs, docs)

search_query = np.ones((1, vector_dim))

results = vs.search(search_query, k=len(docs))

pprint(results)
```

```
[[{'distance': np.float32(1.6019362),
   'doc': {'ice_cream_flavor': 'Dulce de Leche', 'name': 'Devin'},
   'id': 3,
   'vec': array([0.49663496, 0.38377824, 0.6321332 , 0.45838   , 0.14038095,
       0.5488281 , 0.8200291 , 0.6347742 , 0.82873976, 0.39435542],
      dtype=float32)},
  {'distance': np.float32(1.6356196),
   'doc': {'ice_cream_flavor': 'Apricot', 'name': 'Alice'},
   'id': 0,
   'vec': array([0.3221011 , 0.70819205, 0.7256356 , 0.93855625, 0.59427375,
       0.79462117, 0.74481624, 0.6318444 , 0.17722918, 0.01658864],
      dtype=float32)},
  {'distance': np.float32(1.754901),
   'doc': {'ice_cream_flavor': 'Chocolate Chip Cookie Dough',
           'name': 'Christy'},
   'id': 2,
   'vec': array([0.4852044 , 0.86260295, 0.20924412, 0.6022649 , 0.10757214,
       0.5913356 , 0.04581964, 0.77801263, 0.77060676, 0.8089424 ],
      dtype=float32)},
  {'distance': np.float32(1.7927729),
   'doc': {'ice_cream_flavor': 'Banana Nut', 'name': 'Bob'},
   'id': 1,
   'vec': array([0.02818148, 0.79818   , 0.52629197, 0.65800613, 0.8545857 ,
       0.6708415 , 0.12327503, 0.9758107 , 0.9879582 , 0.00569335],
      dtype=float32)}]]
```

And you can go look at it in SQLite as well
```
% sqlite3 my_vector_store.sqlite3
sqlite> .tables
vector
sqlite> .d vector
{...snip...}
CREATE TABLE vector (
    id INTEGER PRIMARY KEY,
    vec BLOB NOT NULL,
    doc TEXT
);
{...snip...}
sqlite> .mode lines
sqlite> select id, doc, hex(vec) from vector;
      id = 0
     doc = {"name": "Alice", "ice_cream_flavor": "Apricot"}
hex(vec) = 6FEAA43E134C353F41C3393F3945703F5322183F4B6C4B3F47AC3E3F8EC0213F917B353EE7E4873C

      id = 1
     doc = {"name": "Bob", "ice_cream_flavor": "Banana Nut"}
hex(vec) = DBDCE63C86554C3F12BB063F1773283F21C65A3F45BC2B3F9E77FC3DBBCE793FD4EA7C3F498FBA3B

      id = 2
     doc = {"name": "Christy", "ice_cream_flavor": "Chocolate Chip Cookie Dough"}
hex(vec) = B66CF83E8CD35C3F1744563E082E1A3FC84EDC3DC561173F61AD3B3DD62B473F7C46453FD9164F3F

      id = 3
     doc = {"name": "Devin", "ice_cream_flavor": "Dulce de Leche"}
hex(vec) = F046FE3E957EC43E7BD3213FC9B0EA3E06C00F3E00800C3F6DED513F9080223F4A28543FF4E8C93E
```
