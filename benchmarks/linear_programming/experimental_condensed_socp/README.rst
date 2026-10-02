Experimental condensed SOCP linear solver
========================================

This is a model-scoped diagnostic, not a production solver feature. It is not
built or enabled by the normal cuOpt build. Loading the live adapter substitutes
a condensed solve at the cuDSS execution boundary while retaining cuOpt's outer
iterative refinement, tolerances, and convergence checks.

Scope and limitations
---------------------

* Linux, double precision, 32-bit matrix indices; tested with CUDA 12.9 and
  cuDSS 0.7.1. The cuDSS matrix API changes in newer versions are not handled.
* Hard-coded retained indices target the tested diagonal-quadratic portfolio
  model: 5,050 original variables, 51 original equalities, one lifted SOC,
  and two cone scaling auxiliaries. The augmented system has dimension 15,207.
* Structural checks require 5,050 local triples and two local pairs after
  retaining 53 variables. These checks are not a general model detector.
* Process-global handles and state: one active solver, one GPU, and no concurrent
  solves. There is no persistent cache across solver instances.
* The fallback build runs cuDSS symbolic setup in advance and permanently
  switches to cuDSS after a numerical check fails. Other structures use cuDSS.
* The no-fallback build skips cuDSS execution phases and rejects unsupported
  structures or numerical failures. cuDSS remains a linked dependency for
  handles and matrix descriptors. Its displayed sparse-factor count is zero.
* CUDA/API failures abort the diagnostic process. Do not preload into a service.
* Full-system checks and status transfers add overhead. The live adapter does
  not use CUDA graphs; graphs are tested only by the offline replay.

Build and run
-------------

Activate the development environment and build cuOpt first. From the repository
root, choose the architecture of your GPU (for example, ``sm_90``)::

  bash benchmarks/linear_programming/experimental_condensed_socp/build_experiment.sh sm_90
  experiment_dir="$PWD/cpp/build/experimental_condensed_socp"
  solver_libraries="$PWD/cpp/build/libcuopt_mathopt.so:$PWD/cpp/build/libcuopt_client.so"
  model=/absolute/path/to/portfolio_socp_n5000_k50_mu_t12_hardfail.mps

Baseline::

  CUDA_MODULE_LOADING=EAGER LD_PRELOAD="$solver_libraries" cpp/build/cuopt_cli "$model"

Condensed solve with cuDSS fallback::

  CUDA_MODULE_LOADING=EAGER LD_PRELOAD="$experiment_dir/live_condensed.so:$solver_libraries" \
    cpp/build/cuopt_cli "$model"

Condensed solve without fallback, using lazy module loading::

  CUDA_MODULE_LOADING=LAZY LD_PRELOAD="$experiment_dir/live_condensed_nofallback.so:$solver_libraries" \
    cpp/build/cuopt_cli "$model"

Use eager loading with the no-fallback build as a separate comparison. The
adapter prints setup-stage timings and factor/solve/fallback counts. Existing
cuOpt ordering/symbolic timer labels include intercepted setup work and should
not be interpreted as cuDSS work in the no-fallback build.

To exercise failure handling, set ``CUOPT_CONDENSED_TEST_FAIL_FACTOR=3`` on a
single command. The fallback version must recover through cuDSS; the no-fallback
version must report a numerical error, not an optimal solution.

Measure iteration latency as the elapsed-time difference between the last and
first iteration rows. Report setup and total solver-clock time separately;
these exclude some command-line startup and file-reading work. Alternate modes,
avoid concurrent GPU work, and report several runs rather than the best one.

Formulation
-----------

Permuting the regularized KKT matrix gives::

  [ E    B ] [p] = [r_p]
  [ B^T  C ] [z]   [r_z]

Here z contains 51 original equality-multiplier directions and two cone scaling
auxiliaries. E is block diagonal with independent 3x3 and 2x2 blocks. Form::

  S = C - sum_j B_j^T inverse(E_j) B_j
  g = r_z - sum_j B_j^T inverse(E_j) r_j
  S z = g
  p_j = inverse(E_j) (r_j - B_j z)

Custom kernels invert the local blocks and reconstruct directions. cuBLAS forms
the dense 53x53 Schur matrix; cuSOLVER factors it using pivoted LU. Its factors
are reused across right-hand sides within the same barrier iteration. This is
exact elimination of the regularized system, not the scalar-diagonal ADAT path.

Capture and replay validation
-----------------------------

Capturing synchronizes the device and writes binary matrices/vectors. Do not
use capture runs for timing. The capture library alone does not change solves::

  capture_dir=$(mktemp -d)
  CUOPT_CAPTURE_KKT="$capture_dir" LD_PRELOAD="$experiment_dir/capture_kkt.so:$solver_libraries" \
    cpp/build/cuopt_cli "$model"
  OPENBLAS_NUM_THREADS=1 python benchmarks/linear_programming/experimental_condensed_socp/check_condensation.py "$capture_dir"
  OPENBLAS_NUM_THREADS=1 python benchmarks/linear_programming/experimental_condensed_socp/replay_native_cuda.py \
    "$capture_dir" "$experiment_dir/condensed_cuda.so"

Replay requires NumPy and SciPy; native GPU replay also requires CuPy. CPU replay
checks full-matrix residuals for captured right-hand sides. GPU replay checks
residuals, refinement, graph execution, and updated graph inputs. Its timing
excludes input preparation, transfers, residual checks, and outer refinement;
it is not end-to-end solver timing. Binary captures contain model-derived data
and are not included in this repository.

Observed results
----------------

On the tested instance, general barrier-loop optimizations reduced iteration
time from about 86 ms to 71-72 ms. The live condensed path reduced it to about
48 ms, with 23 iterations, objective -0.0985055149, and relative objective gap
4.14e-7. Five-run median solver-clock totals were 233 ms for eager baseline,
207 ms for lazy baseline, and 168 ms for lazy no-fallback condensation.
These are single-instance measurements, not performance guarantees.
