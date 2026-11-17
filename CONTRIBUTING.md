# Contributing

1. Create a feature branch and install uv.
2. Run `uv sync` and `uv run pytest -q` before changes.
3. For each numerical method, add regression tests that verify known analytic answers or credible reference library results. Test odd/even sample sizes, boundary effects, NaNs, zero vectors, and multi-axis shapes where relevant.
4. Prefer clear PEP 8 `snake_case` public names. Document units, array shapes, sample-axis direction, and rotation conventions.
5. Keep public functions small, documented, and covered by numerical regression tests; describe API changes in updated usage and reference documentation.
6. Do not assert clinical validity based only on code passing tests.
7. Before releasing publicly, confirm rights and choose a suitable license; the original upload did not include one.

## Type annotations

Use type aliases from `biomechanics_signal_tools.types` for new functions.
`ArrayLike` belongs on inputs accepted via `np.asarray`; `FloatArray` belongs
only on outputs guaranteed to be `float64`. Keep dimension and sampling checks
at runtime and add tests for list/tuple/NumPy input and output dtypes.
See `docs/typing.md`.
