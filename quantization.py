import nanopq
import numpy as np
import pdb
N, Nt, D = 10000, 2000, 128
X = np.random.random((N, D)).astype(np.float32)  # 10,000 128-dim vectors to be indexed
Xt = np.random.random((Nt, D)).astype(np.float32)  # 2,000 128-dim vectors for training
query = np.random.random((D,)).astype(np.float32)  # a 128-dim query vector

pq = nanopq.PQ(M=64)

# Train codewords
pq.fit(Xt)

# Encode to PQ-codes
X_code = pq.encode(X)  # (10000, 8) with dtype=np.uint8

# Results: create a distance table online, and compute Asymmetric Distance to each PQ-code 
dists = pq.dtable(query).adist(X_code)  # (10000, ) 
print(pq.codewords.shape)
