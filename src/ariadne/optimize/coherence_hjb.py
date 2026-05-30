"""Sampled-graph Helmholtz value-function solver -- the Forge-Doctrine HJB substitute.

A mesh-free approach to Hamilton-Jacobi-Bellman in N-dim state spaces. Instead of gridding
state space (which costs O(M^N) and is intractable for N>=4), we:

1. SAMPLE the feasible region with quasi-random points (Halton/Sobol).
2. BUILD a graph whose nodes are samples and whose edges encode the system's local dynamics
   -- edge (i, j) means a control can drive the system from x_i to x_j in unit time at edge
   cost w(i, j) (the line-integral of L(x, u) over the segment).
3. SOLVE the graph Helmholtz PDE  (Gamma*I + D*L_graph) * V = source  -- ONE sparse SPD CG
   solve gives the field response at every sample. (Forge Doctrine Section 14: graph-Laplacian
   discretisation of an elliptic operator; Belkin & Niyogi 2003 / Coifman & Lafon 2006.)
4. APPLY the surprise transform  W(x) = -ln(V(x) / V_max)  -- maps the exponentially-decaying
   Helmholtz field to a linear-in-distance pseudo-eikonal so -grad W points TOWARD the goal
   (the optimal-control direction). Coherence length L_coh = sqrt(D / Gamma) sets the decay.

This is the Forge-Doctrine adjoint trick made concrete: one PDE solve at the goal gives the
value function at every potential start. No grid. No curse of dimensionality (cost scales with
sample count, not ambient dimension).

CALIBRATION (mandatory): validate on a 2D analytic eikonal problem with closed-form V = ||x||
before scaling to 6D CR3BP. If the surprise-transformed Helmholtz field doesn't match the
eikonal in 2D, this approach won't work in higher dimensions either.

References:
- The Forge Doctrine (Sections 14-20.13.14): tau_c field, log-cost, Green's adjoint
- Belkin & Niyogi 2003: Laplacian Eigenmaps
- Coifman & Lafon 2006: Diffusion Maps
- Sutton & Barto 2018, Ch 3-4: value functions and Bellman equations
"""
from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg
from scipy.spatial import cKDTree


def halton(n: int, dim: int, seed: int = 0):
    """N-dim Halton low-discrepancy sequence (deterministic, no random state). Returns (n, dim).

    Better than uniform for sample-grid coverage; same statistics as Sobol for our purposes.
    """
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    if dim > len(primes):
        raise ValueError(f"halton: only {len(primes)} primes available, dim={dim} too large")

    def _van_der_corput(k: int, base: int) -> float:
        q, denom = 0.0, 1.0
        while k > 0:
            denom *= base
            q += (k % base) / denom
            k //= base
        return q

    out = np.empty((n, dim))
    for d in range(dim):
        b = primes[d]
        for i in range(n):
            out[i, d] = _van_der_corput(i + 1 + seed, b)
    return out


def knn_graph(samples: np.ndarray, k: int = 12, sigma: float | None = None,
              metric: str = "euclidean"):
    """Build a k-nearest-neighbour weighted graph from a sample set.

    Edge weights: Gaussian-decayed (Belkin-Niyogi heat kernel)  w(i, j) = exp(-d^2 / sigma^2)
    if sigma > 0, else uniform 1/d. Returns a symmetric scipy.sparse CSR matrix.

    sigma defaults to the median k-NN distance -- the heuristic that gives the cleanest
    discrete approximation of the continuous Laplace-Beltrami operator on the sample's
    implicit manifold (Belkin & Niyogi 2003 Theorem 6).
    """
    n = len(samples)
    tree = cKDTree(samples)
    # k+1 because each point is its own nearest neighbour
    dists, idxs = tree.query(samples, k=k + 1)
    dists = dists[:, 1:]
    idxs = idxs[:, 1:]
    if sigma is None:
        sigma = float(np.median(dists))

    rows = np.repeat(np.arange(n), k)
    cols = idxs.flatten()
    if metric == "euclidean":
        w = np.exp(-(dists.flatten() ** 2) / (sigma * sigma + 1e-30))
    else:
        # 1/d weighting (more aggressive for far neighbours, faster decay)
        w = 1.0 / (dists.flatten() + 1e-12)

    W = sp.csr_matrix((w, (rows, cols)), shape=(n, n))
    # symmetrise (k-NN is asymmetric: i may be a neighbour of j but not vice versa)
    W = 0.5 * (W + W.T)
    return W.tocsr()


def graph_laplacian(W: sp.csr_matrix, normalized: bool = False):
    """Combinatorial L = D - W or symmetric-normalized L_sym = I - D^{-1/2} W D^{-1/2}.

    Combinatorial is the natural choice for the Forge-Doctrine elliptic operator (Gamma*I + D*L);
    normalised gives spectral properties closer to the continuous Laplace-Beltrami operator.
    """
    deg = np.asarray(W.sum(axis=1)).flatten()
    D = sp.diags(deg)
    if not normalized:
        return D - W
    d_inv_sqrt = sp.diags(1.0 / np.sqrt(np.where(deg > 1e-30, deg, 1.0)))
    return sp.eye(W.shape[0], format="csr") - d_inv_sqrt @ W @ d_inv_sqrt


def solve_helmholtz(L: sp.csr_matrix, gamma: float, D_coef: float, source_idx,
                    source_amp: float = 1.0, tol: float = 1e-10, maxiter: int = 5000):
    """Solve (gamma*I + D_coef*L)*V = source via sparse CG.

    `source_idx` may be a single int or a list of ints; `source_amp` distributes a unit total
    source amplitude over them. The operator is SPD when L is SPD and gamma > 0, so CG converges
    geometrically with rate sqrt(condition_number).

    Returns (V, cg_info, n_iter) where cg_info is 0 on success.
    """
    n = L.shape[0]
    A = gamma * sp.eye(n, format="csr") + D_coef * L
    s = np.zeros(n)
    if np.isscalar(source_idx):
        s[int(source_idx)] = source_amp
    else:
        for j in source_idx:
            s[int(j)] = source_amp / len(source_idx)
    iters = {"n": 0}

    def _cb(xk):
        iters["n"] += 1

    V, info = cg(A, s, rtol=tol, maxiter=maxiter, callback=_cb)
    return V, int(info), iters["n"]


def surprise_value(V: np.ndarray, V_max: float | None = None, eps: float = 1e-30):
    """W(x) = -ln(V(x) / V_max) -- the Equation-of-One §20.13.14 log-cost rewrite.

    Maps the exponentially-decaying Helmholtz response to a linear pseudo-eikonal: where the
    Helmholtz V(x) ~ exp(-r / L_coh) / (something), W ~ r / L_coh up to additive constant.
    `V_max` defaults to max(V); pass it explicitly to use a different reference (e.g., V at
    the goal node).
    """
    if V_max is None:
        V_max = float(np.max(V))
    V_clipped = np.where(V > eps, V, eps)
    return -np.log(V_clipped / V_max)


def hjb_solve(samples: np.ndarray, goal_idx, *, k: int = 12, gamma: float = 1.0,
              D_coef: float = 1.0, sigma: float | None = None, normalized: bool = False,
              tol: float = 1e-10):
    """End-to-end: samples -> k-NN graph -> Helmholtz solve -> surprise value function.

    Returns dict {V, W, L, W_graph, n_iter, cg_info, sigma}. W is the pseudo-eikonal usable as
    a value-function approximation; gradient of W (via finite-differences on k-NN edges) gives
    the local optimal-control direction.
    """
    W_graph = knn_graph(samples, k=k, sigma=sigma)
    L = graph_laplacian(W_graph, normalized=normalized)
    V, info, n_iter = solve_helmholtz(L, gamma, D_coef, goal_idx, tol=tol)
    # V_max at the goal sample (the strongest response, sanity-check)
    if np.isscalar(goal_idx):
        V_max = float(V[int(goal_idx)])
    else:
        V_max = float(np.mean([V[int(j)] for j in goal_idx]))
    W = surprise_value(V, V_max=V_max)
    return {"V": V, "W": W, "L": L, "W_graph": W_graph, "n_iter": n_iter,
            "cg_info": info, "V_max": V_max, "sigma_used": sigma}


def gradient_descent_policy(samples: np.ndarray, W: np.ndarray, W_graph: sp.csr_matrix,
                            start_idx: int, max_steps: int = 200,
                            goal_tol: float = 1e-6):
    """Follow -grad W from start_idx along graph edges -- the greedy / steepest-descent policy.

    At each step, move to the neighbour that most decreases W. Returns the trajectory of sample
    indices and final W value. Stops when no neighbour has lower W (reached the basin's bottom)
    or after max_steps.
    """
    path = [int(start_idx)]
    for _ in range(max_steps):
        i = path[-1]
        neighbours = W_graph[i].nonzero()[1]
        if len(neighbours) == 0:
            break
        # only neighbours with strictly lower W
        deltas = W[neighbours] - W[i]
        best = neighbours[int(np.argmin(deltas))]
        if W[best] >= W[i] - goal_tol:
            break
        path.append(int(best))
    return path
