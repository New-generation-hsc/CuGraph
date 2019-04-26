# CuGraph

![](http://progressed.io/bar/86?title=done)

A CUDA-based graph processing system, include three primary graph primitives such as  `bfs`, `sssp`, `pagerank`.

CuGraph follow the method, a novel graph representation `G-shard` for coalesced memory access which was introduced in [CuSha paper](https://dl.acm.org/citation.cfm?id=2600227).

Also, CuGraph system implements two PRAM(parallel random access machine)-like algorithm, `PNBA` (parallel node-based algorithm) and `PEBA` (parallel edge-based algorithm).

---

**benchmarks**


