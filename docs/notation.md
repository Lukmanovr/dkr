The course-wide notation contract. Every lecture, lab, homework, and exam uses these
symbols with these meanings; deviations are bugs. Base convention follows Hamilton,
*Graph Representation Learning* (2020); where the book is ambiguous we fix one choice
(marked ⚑). CI lints lecture sources against the symbol table below.

## Graphs and matrices

| Symbol | Meaning |
|---|---|
| $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ | Graph with node set $\mathcal{V}$, edge set $\mathcal{E}$; $u, v \in \mathcal{V}$; $(u,v) \in \mathcal{E}$. Simple graph = no self-loops, at most one undirected edge per pair |
| $n = |\mathcal{V}|,\; m = |\mathcal{E}|$ | Node and edge counts |
| $\mathbf{A} \in \mathbb{R}^{n \times n}$ | Adjacency matrix, $\mathbf{A}[u,v] = 1 \iff (u,v) \in \mathcal{E}$ (real-valued if weighted; symmetric iff undirected) |
| $(u, \tau, v)$, $\tau \in \mathcal{R}$ | Multi-relational (KG) edge with relation $\tau$ from relation set $\mathcal{R}$; also written $(h, r, t)$ head/relation/tail in KG weeks; $\mathbf{A}_\tau$ = adjacency of relation $\tau$ |
| $\mathcal{A} \in \mathbb{R}^{n \times |\mathcal{R}| \times n}$ | Adjacency tensor of a multi-relational graph |
| $\mathbf{X} \in \mathbb{R}^{n \times m_0}$ | ⚑ Node feature matrix, one **row** $\mathbf{x}_u$ per node (the book once uses $d \times |\mathcal{V}|$; we standardize on rows) |
| $d_u$, $\mathbf{D}$ | Degree $d_u = \sum_v \mathbf{A}[u,v]$; diagonal degree matrix |
| $\mathcal{N}(u)$ | Neighborhood $\{v : (u,v) \in \mathcal{E}\}$; $\mathcal{N}_\tau(u)$ under relation $\tau$ |
| $\mathbf{L} = \mathbf{D} - \mathbf{A}$ | Unnormalized Laplacian; PSD; $\mathbf{x}^\top \mathbf{L} \mathbf{x} = \sum_{(u,v) \in \mathcal{E}} (x_u - x_v)^2$; multiplicity of eigenvalue 0 = #components |
| $\mathbf{L}_{\text{sym}} = \mathbf{D}^{-1/2} \mathbf{L} \mathbf{D}^{-1/2}$, $\mathbf{L}_{\text{rw}} = \mathbf{D}^{-1} \mathbf{L}$ | Normalized Laplacians; $\mathbf{L}_{\text{sym}}$ shares eigenvectors with $\mathbf{A}_{\text{sym}} = \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}$ |
| $\mathbf{P} = \mathbf{D}^{-1}\mathbf{A}$ | ⚑ Random-walk transition matrix, **row-stochastic** ($\mathbf{P}[u,v]$ = prob. of stepping $u \to v$). The book mixes row/column conventions; we fix row-stochastic everywhere. PageRank is written over $\mathbf{P}^\top$: $\mathbf{r} = \beta \mathbf{P}^\top \mathbf{r} + (1-\beta)\mathbf{1}/n$ with damping $\beta$ |
| $\mathbf{U}, \boldsymbol{\Lambda}$ | Laplacian eigendecomposition $\mathbf{L} = \mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^\top$, eigenvalues $\lambda_1 \le \dots \le \lambda_n$ |
| $\mathbf{P}_\pi$ (permutation) | ⚑ Permutation matrix is always subscripted $\mathbf{P}_\pi$ to avoid collision with the transition matrix. Invariance: $f(\mathbf{P}_\pi \mathbf{A} \mathbf{P}_\pi^\top) = f(\mathbf{A})$; equivariance: $f(\mathbf{P}_\pi \mathbf{A} \mathbf{P}_\pi^\top, \mathbf{P}_\pi \mathbf{X}) = \mathbf{P}_\pi f(\mathbf{A}, \mathbf{X})$ |

## Classical statistics and kernels

| Symbol | Meaning |
|---|---|
| $e_u$ | Eigenvector centrality: $\lambda \mathbf{e} = \mathbf{A} \mathbf{e}$, leading eigenvector (Perron–Frobenius) |
| $c_u$ | Local clustering coefficient $= \#\text{edges among } \mathcal{N}(u) / \binom{d_u}{2}$ |
| $\mathbf{S}$, $\mathbf{S}[u,v]$ | Node–node similarity matrix (Jaccard, Adamic–Adar, Katz, random-walk…); the reconstruction target of encoder–decoder methods |
| $l^{(i)}(v)$, $\text{HASH}$, $\{\!\{\cdot\}\!\}$ | WL label of $v$ at iteration $i$: $l^{(i)}(v) = \text{HASH}\big(l^{(i-1)}(v), \{\!\{ l^{(i-1)}(u) : u \in \mathcal{N}(v) \}\!\}\big)$; double braces = multiset |
| $Q$ | Modularity of a partition (Newman); $\Delta Q$ = modularity gain of a Louvain move |
| $\text{cut}$, $\text{vol}$ | Cut value of a partition; volume $\text{vol}(A) = \sum_{u \in A} d_u$ (RatioCut/NCut) |

## Embeddings and knowledge graphs

| Symbol | Meaning |
|---|---|
| $\text{ENC}$, $\mathbf{Z} \in \mathbb{R}^{n \times d}$, $\mathbf{z}_u$ | Encoder; shallow embedding = lookup $\text{ENC}(u) = \mathbf{Z}[u]$; $d$ = embedding dimension; $\mathbf{z}_{\mathcal{G}}$ = graph-level embedding |
| $\text{DEC}$ | Pairwise decoder $\mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}^+$ with $\text{DEC}(\mathbf{z}_u, \mathbf{z}_v) \approx \mathbf{S}[u,v]$; multi-relational form scores $(u, \tau, v)$ |
| $p_{\mathcal{G},T}(v \mid u)$ | Probability of visiting $v$ on a length-$T$ random walk from $u$ (DeepWalk/node2vec similarity) |
| $p, q$ | node2vec return and in-out parameters (BFS↔DFS interpolation) |
| $\sigma$, $P_n$, $k$ | Sigmoid; negative-sampling distribution; number of negative samples |
| $\mathbf{r}_\tau \in \mathbb{R}^d$ (or $\mathbb{C}^d$) | Relation embedding (TransE/DistMult/ComplEx/RotatE); $\mathbf{R}_\tau \in \mathbb{R}^{d \times d}$ relation matrix (RESCAL); $\bar{\mathbf{z}}$ complex conjugate; $\circ$ Hadamard/element-wise product |
| $s(h, r, t)$ | ⚑ KG triple score (higher = more plausible); distance-based models use $s = -\lVert \cdot \rVert$ so higher is always better |
| $\gamma$ | Margin in max-margin ranking losses |
| MRR, Hits@$K$ | Filtered ranking metrics (filtered = corrupted candidates that form true triples are removed before ranking) |

## GNNs

| Symbol | Meaning |
|---|---|
| $\mathbf{h}_u^{(k)}$ | Embedding of node $u$ after $k$ message-passing layers; $\mathbf{h}_u^{(0)} = \mathbf{x}_u$; final $\mathbf{z}_u = \mathbf{h}_u^{(K)}$; $\mathbf{H}^{(k)} \in \mathbb{R}^{n \times d_k}$ stacked |
| $\text{UPDATE}^{(k)}, \text{AGGREGATE}^{(k)}, \mathbf{m}_{\mathcal{N}(u)}^{(k)}$ | Message-passing framework: $\mathbf{h}_u^{(k+1)} = \text{UPDATE}\big(\mathbf{h}_u^{(k)}, \text{AGGREGATE}(\{\!\{\mathbf{h}_v^{(k)} : v \in \mathcal{N}(u)\}\!\})\big)$; AGGREGATE is permutation-invariant |
| $\mathbf{W}_{\text{self}}^{(k)}, \mathbf{W}_{\text{neigh}}^{(k)}, \mathbf{b}^{(k)}$ | Basic-GNN parameters |
| $\tilde{\mathbf{A}} = (\mathbf{D} + \mathbf{I})^{-1/2} (\mathbf{A} + \mathbf{I}) (\mathbf{D} + \mathbf{I})^{-1/2}$ | Self-loop symmetric-normalized adjacency; GCN layer $\mathbf{H}^{(k)} = \sigma(\tilde{\mathbf{A}} \mathbf{H}^{(k-1)} \mathbf{W}^{(k)})$ |
| $\alpha_{u,v}$ | Attention weight on neighbor $v$ when aggregating at $u$ (GAT); $\alpha_{u,v}^{(i)}$ for head $i$; $\Vert$ = concatenation |
| $\epsilon$ | GIN's learnable self-loop weight: $\mathbf{h}_u^{(k)} = \text{MLP}\big((1+\epsilon)\mathbf{h}_u^{(k-1)} + \sum_{v \in \mathcal{N}(u)} \mathbf{h}_v^{(k-1)}\big)$ |
| $I_K(u, v)$ | Influence of $u$ on $v$ after $K$ layers, $\mathbf{1}^\top \big(\partial \mathbf{h}_v^{(K)} / \partial \mathbf{h}_u^{(0)}\big) \mathbf{1}$ (oversmoothing analysis) |
| $p_w(\mathbf{L}) = \sum_i w_i \mathbf{L}^i$ | Polynomial graph filter; spectral filter $p_w(\boldsymbol{\Lambda})$ acts on eigenvalues; graph Fourier transform $\hat{\mathbf{f}} = \mathbf{U}^\top \mathbf{f}$ |
| $\mathbf{e}_{(u,\tau,v)}$ | Edge feature vector |
| $\mathcal{V}_{\text{train}}, \mathcal{V}_{\text{trans}}, \mathcal{V}_{\text{ind}}$; $\mathcal{E}_{\text{train}}$ | Training / transductive-test / inductive-test node splits; observed message-passing edges in link prediction |

## Generative models and temporal graphs

| Symbol | Meaning |
|---|---|
| $q_\phi(\mathbf{Z} \mid \mathcal{G})$, $p_\theta(\mathbf{A} \mid \mathbf{Z})$ | VGAE encoder/decoder; ELBO objective; $p(\mathbf{A}[u,v] = 1) = \sigma(\mathbf{z}_u^\top \mathbf{z}_v)$ |
| $\mathbf{s}_u(t)$ | ⚑ Temporal memory/state of node $u$ at time $t$ (TGN); events $(u, v, t, \mathbf{e}_{uv}(t))$ |
| $\text{MMD}$ | Maximum mean discrepancy between graph-statistic distributions (generation evaluation) |

## House rules

1. Every display formula gets a plain-English caption naming each newly introduced symbol.
2. Vectors bold lowercase ($\mathbf{x}$), matrices bold uppercase ($\mathbf{A}$), sets calligraphic ($\mathcal{V}$), scalars italic ($d$, $\lambda$).
3. Layer index is a parenthesized superscript ($\mathbf{h}_u^{(k)}$), never a subscript.
4. KG weeks may use $(h, r, t)$ prose shorthand but formulas keep $\mathbf{z}_h, \mathbf{r}, \mathbf{z}_t$ typing.
5. "Node", not "vertex", in prose (matches the GRL book and PyG docs).
6. Cite the GRL book by section number (stable across draft and published editions), never by page.
