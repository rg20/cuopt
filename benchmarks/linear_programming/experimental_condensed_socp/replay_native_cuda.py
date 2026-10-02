# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native CUDA replay of captured KKT systems; excludes solver integration and refinement timing."""

import ctypes
from pathlib import Path
import sys
import time

import cupy as cp
import numpy as np

from check_condensation import Condensed, read_matrix


class Native:
    def __init__(self, library):
        self.lib = ctypes.CDLL(str(library))
        self.lib.initialize.restype = ctypes.c_int
        self.work_size = self.lib.initialize()
        self.lib.factor.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * 11
        self.lib.solve.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * 13
        self.lib.set_stream.argtypes = [ctypes.c_void_p]


class Replay:
    def __init__(self, native, matrix, host, rhs):
        self.native = native
        blocks, coupling, indices = [], [], []
        for ind, block, couple, _ in host.groups:
            count, size = ind.shape
            padded = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
            padded[:, :size, :size] = block
            padded_c = np.zeros((count, 3, 53))
            padded_c[:, :size] = couple
            padded_i = np.full((count, 3), -1, dtype=np.int32)
            padded_i[:, :size] = ind
            blocks.append(padded)
            coupling.append(padded_c)
            indices.append(padded_i)
        self.blocks = cp.asarray(np.concatenate(blocks))
        self.count = len(self.blocks)
        self.indices = cp.asarray(np.concatenate(indices))
        self.coupling = cp.asarray(
            np.asfortranarray(np.concatenate(coupling).reshape(-1, 53))
        )
        self.keep = cp.asarray(host.keep.astype(np.int32))
        self.corner = cp.asarray(
            np.asfortranarray(matrix[host.keep][:, host.keep].toarray())
        )
        self.inverse = cp.empty_like(self.blocks)
        self.w = cp.empty_like(self.coupling)
        self.lu = cp.empty_like(self.corner)
        self.work = cp.empty(native.work_size, dtype=cp.float64)
        self.pivots = cp.empty(53, dtype=cp.int32)
        self.factor_info = cp.zeros(1, dtype=cp.int32)
        self.solve_info = cp.zeros(1, dtype=cp.int32)
        self.bad = cp.zeros(1, dtype=cp.int32)
        self.local = cp.empty(self.count * 3, dtype=cp.float64)
        self.small = cp.empty(53, dtype=cp.float64)
        self.rhs = [cp.asarray(b) for b in rhs]
        self.outputs = [cp.empty_like(b) for b in self.rhs]

    def factor(self, stream):
        arrays = (
            self.blocks,
            self.coupling,
            self.corner,
            self.inverse,
            self.w,
            self.lu,
            self.work,
            self.pivots,
            self.factor_info,
            self.bad,
        )
        self.native.lib.factor(
            self.count, *(a.data.ptr for a in arrays), stream.ptr
        )

    def solve(self, rhs, output, stream):
        arrays = (
            self.indices,
            self.keep,
            self.coupling,
            self.inverse,
            self.w,
            self.lu,
            self.pivots,
            rhs,
            self.local,
            self.small,
            output,
            self.solve_info,
        )
        self.native.lib.solve(
            self.count, *(a.data.ptr for a in arrays), stream.ptr
        )

    def run(self, stream):
        self.factor(stream)
        for rhs, output in zip(self.rhs, self.outputs):
            self.solve(rhs, output, stream)

    def check(self):
        assert self.factor_info.get()[0] == 0
        assert self.solve_info.get()[0] == 0
        assert self.bad.get()[0] == 0


def measure(replays, stream, graphs=None):
    times = []
    for repeat in range(8):
        begin = time.perf_counter()
        if graphs is None:
            for replay in replays:
                replay.run(stream)
        else:
            for graph in graphs:
                graph.launch(stream)
        stream.synchronize()
        times.append(1000 * (time.perf_counter() - begin))
    print(
        "graph" if graphs is not None else "native",
        "wall ms",
        times,
        "warm median",
        np.median(times[1:]),
        flush=True,
    )


def main():
    directory = Path(sys.argv[1])
    native = Native(Path(sys.argv[2]).resolve())
    stream = cp.cuda.Stream.null
    native.lib.set_stream(stream.ptr)
    replays, matrices, all_rhs = [], [], []
    worst_raw, worst_refined = 0.0, 0.0
    for path in sorted(directory.glob("*-matrix.bin")):
        matrix = read_matrix(path)
        rhs = [
            np.fromfile(p, np.float64)
            for p in sorted(directory.glob(path.name[:4] + "-*-rhs.bin"))
        ]
        replay = Replay(native, matrix, Condensed(matrix), rhs)
        replay.run(stream)
        stream.synchronize()
        replay.check()
        local_raw, local_refined = 0.0, 0.0
        for b, output in zip(rhs, replay.outputs):
            x = output.get()
            scale = np.linalg.norm(b, np.inf)
            local_raw = max(
                local_raw, np.linalg.norm(b - matrix @ x, np.inf) / scale
            )
            for _ in range(3):
                residual = b - matrix @ x
                if np.linalg.norm(residual, np.inf) / scale < 1e-12:
                    break
                correction = cp.empty_like(output)
                replay.solve(cp.asarray(residual), correction, stream)
                x += correction.get()
            local_refined = max(
                local_refined, np.linalg.norm(b - matrix @ x, np.inf) / scale
            )
        worst_raw = max(worst_raw, local_raw)
        worst_refined = max(worst_refined, local_refined)
        print(
            path.name[:4], "raw/refined", local_raw, local_refined, flush=True
        )
        replays.append(replay)
        matrices.append(matrix)
        all_rhs.append(rhs)
    print("WORST raw/refined", worst_raw, worst_refined, flush=True)
    assert worst_raw < 1e-7 and worst_refined < 1e-8
    measure(replays, stream)
    capture_stream = cp.cuda.Stream(non_blocking=True)
    native.lib.set_stream(capture_stream.ptr)
    graphs = []
    with capture_stream:
        for replay in replays:
            replay.run(capture_stream)
            capture_stream.synchronize()
            capture_stream.begin_capture()
            replay.run(capture_stream)
            graphs.append(capture_stream.end_capture())
        measure(replays, capture_stream, graphs)
    for replay, matrix, rhs in zip(replays, matrices, all_rhs):
        replay.check()
        for output, b in zip(replay.outputs, rhs):
            residual = np.linalg.norm(
                b - matrix @ output.get(), np.inf
            ) / np.linalg.norm(b, np.inf)
            assert residual < 1e-7
    print("All 97 graph outputs passed full-matrix residual check", flush=True)
    # Confirm replay uses mutable values, not only the inputs present at capture.
    changed_matrices = []
    with capture_stream:
        for replay, matrix in zip(replays, matrices):
            corner = replay.corner.get()
            corner[np.diag_indices(53)] += 1e-6
            replay.corner.set(corner, stream=capture_stream)
            for b in replay.rhs:
                b *= 1.125
            changed = matrix.copy().tolil()
            for index in replay.keep.get():
                changed[index, index] += 1e-6
            changed_matrices.append(changed.tocsr())
        for graph in graphs:
            graph.launch(capture_stream)
        capture_stream.synchronize()
    changed_worst = 0.0
    for replay, matrix, rhs in zip(replays, changed_matrices, all_rhs):
        replay.check()
        for output, b in zip(replay.outputs, rhs):
            changed_b = b * 1.125
            residual = np.linalg.norm(
                changed_b - matrix @ output.get(), np.inf
            ) / np.linalg.norm(changed_b, np.inf)
            changed_worst = max(changed_worst, residual)
            assert residual < 1e-7
    print(
        "Updated RHS and matrix graph check passed; worst residual",
        changed_worst,
        flush=True,
    )
    native.lib.finalize()


if __name__ == "__main__":
    main()
