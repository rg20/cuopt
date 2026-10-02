# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline diagnostic: exact block condensation of captured regularized KKT systems.

The partition is specific to the portfolio model; no timings are solver benchmarks.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def read_matrix(path):
    with path.open("rb") as f:
        n, m, nz, kind, view = np.fromfile(f, np.int64, 5)
        row = np.fromfile(f, np.int32, n + 1)
        col = np.fromfile(f, np.int32, nz)
        val = np.fromfile(f, np.float64, nz)
        assert not f.read(1)
    matrix = csr_matrix((val, col, row), shape=(n, m))
    assert (matrix - matrix.T).nnz == 0, (kind, view)
    return matrix


class Condensed:
    def __init__(self, matrix):
        n = matrix.shape[0]
        assert n == 15207
        self.keep = np.r_[np.arange(10102, 10153), np.arange(n - 2, n)]
        eliminated = np.setdiff1d(np.arange(n), self.keep)
        count, labels = connected_components(
            matrix[eliminated][:, eliminated], directed=False
        )
        sizes = np.bincount(labels)
        assert count == 5052 and np.all(np.isin(sizes, [2, 3]))
        order = np.argsort(labels, kind="stable")
        starts = np.r_[0, np.cumsum(sizes)]
        schur = matrix[self.keep][:, self.keep].toarray()
        self.groups = []
        for size in np.unique(sizes):
            ids = np.where(sizes == size)[0]
            indices = eliminated[order[starts[ids, None] + np.arange(size)]]
            rows = np.broadcast_to(
                indices[:, :, None], (len(ids), size, size)
            ).ravel()
            cols = np.broadcast_to(
                indices[:, None, :], (len(ids), size, size)
            ).ravel()
            blocks = np.asarray(matrix[rows, cols]).reshape(-1, size, size)
            coupling = (
                matrix[indices.ravel()][:, self.keep]
                .toarray()
                .reshape(-1, size, 53)
            )
            inv_coupling = np.linalg.solve(blocks, coupling)
            schur -= np.einsum("bik,bil->kl", coupling, inv_coupling)
            self.groups.append((indices, blocks, coupling, inv_coupling))
        self.condition = np.linalg.cond(schur)
        self.factor = lu_factor(schur)

    def solve(self, rhs):
        condensed_rhs = rhs[self.keep].copy()
        local = []
        for indices, blocks, coupling, inv_coupling in self.groups:
            values = np.linalg.solve(blocks, rhs[indices, None])[..., 0]
            condensed_rhs -= np.einsum("bik,bi->k", coupling, values)
            local.append(values)
        result = np.zeros_like(rhs)
        result[self.keep] = lu_solve(self.factor, condensed_rhs)
        for (indices, blocks, coupling, inv_coupling), values in zip(
            self.groups, local
        ):
            result[indices] = values - inv_coupling @ result[self.keep]
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    totals = []
    print(
        "factor solves cond(S) max_raw_relative_residual max_refined_relative_residual "
        "max_cudss_relative_residual max_solution_difference"
    )
    for path in sorted(args.directory.glob("*-matrix.bin")):
        matrix = read_matrix(path)
        solver = Condensed(matrix)
        values = []
        for rhs_path in sorted(
            args.directory.glob(path.name[:4] + "-*-rhs.bin")
        ):
            rhs = np.fromfile(rhs_path, np.float64)
            reference = np.fromfile(
                str(rhs_path).replace("-rhs.bin", "-solution.bin"), np.float64
            )
            scale = max(np.linalg.norm(rhs, np.inf), np.finfo(float).tiny)
            solution = solver.solve(rhs)
            raw = np.linalg.norm(rhs - matrix @ solution, np.inf) / scale
            for _ in range(3):
                residual = rhs - matrix @ solution
                if np.linalg.norm(residual, np.inf) / scale < 1e-12:
                    break
                solution += solver.solve(residual)
            refined = np.linalg.norm(rhs - matrix @ solution, np.inf) / scale
            ref_residual = (
                np.linalg.norm(rhs - matrix @ reference, np.inf) / scale
            )
            difference = np.linalg.norm(solution - reference, np.inf) / max(
                np.linalg.norm(reference, np.inf), np.finfo(float).tiny
            )
            assert np.all(np.isfinite(solution))
            values.append([raw, refined, ref_residual, difference])
        maxima = np.max(values, axis=0)
        totals.extend(values)
        print(
            path.name[:4],
            len(values),
            "%.3e" % solver.condition,
            " ".join("%.3e" % x for x in maxima),
            flush=True,
        )
    print("TOTAL", len(totals), np.max(totals, axis=0))


if __name__ == "__main__":
    main()
