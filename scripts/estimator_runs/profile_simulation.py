"""Profile the regular simulation pipeline with ``cProfile``.

The script accepts exactly the same command-line arguments as
``scripts.estimator_runs.run_simulation`` and executes that entry point unchanged.
This makes it useful for simple CPU profiling with real experiment configurations.

Example
-------
python -m scripts.estimator_runs.profile_simulation \
    --config configs/my_experiment.yaml \
    --snr 0.01 \
    --n-runs 1
"""

from cProfile import Profile
from pstats import SortKey, Stats
from time import perf_counter

from scripts.estimator_runs.run_simulation import main as run_simulation


N_PROFILE_ROWS = 30


def main() -> None:
    """Run the normal simulation entry point and print cumulative profiling stats."""
    profiler = Profile()

    start = perf_counter()
    profiler.runcall(run_simulation)
    elapsed = perf_counter() - start

    print(f"\nTotal elapsed time: {elapsed:.3f} s\n")
    Stats(profiler).strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(
        N_PROFILE_ROWS
    )


if __name__ == "__main__":
    main()
