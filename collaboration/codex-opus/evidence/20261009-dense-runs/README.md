# Dense-run evidence — 2026-10-09

Download `dense-runs-small.tgz` and verify it with `SHA256SUMS`; extract and verify the inner SHA256SUMS.
```sh
sha256sum -c SHA256SUMS
tar -xzf dense-runs-small.tgz
cd 20261009-dense-runs
sha256sum -c SHA256SUMS
```

- `A/`–`D/`: configs, statuses, all train/probe logs, final evaluation JSON and generation manifests.
- `launch.md`: exact commands, effective flags, base commit, GPU identity, recorded/corrected times and deviations.
- `checkpoints.json`: persistent paths, sizes, hashes, actual optimizer metadata and restoration boundaries.
- `evaluation_inventory.json`: all four raw evaluations complete; only D EMA was run.
- `EXECUTION_DEVIATIONS.json`: CPU workaround, incomplete preflight and missing coarse probes, including correction of the inaccurate user-authorization label in raw logs.
- `execution/`: actual code and the local patch against 2391404.
- No weights or prediction arrays; these stay on the server.

See [results message](../../messages/20261009-003-codex-dense-results.md) for all requested metrics.

Server artifacts: `/guohaoran/tmp/nexus_vertex_dense_overfit50_20261009`. Best raw checkpoint: `A/final.pt`, SHA256 `1bd4ced48296b45c80d54386171652a6f9eb9c09e8fc976965caad62899b55f9`.
