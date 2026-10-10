# Round2C Part D executed; artifact delivery blocked by SSH

At the independent server checkout, HEAD was fast-forwarded to `427498e88912f351ff50b70aef5888999b2af26b`. Training/evaluation source hashes were unchanged. The unmodified CPU analyzer finished successfully on the exact six Part C evaluations (600 saved trees), after verification of existing input/prediction/label hashes and all finite depth1–9 estimates.

Observed execution result:
```
PART_D_COMPLETE {"trees": 600, "seconds": 45.06409000698477, "output": "/guohaoran/tmp/nexus_vertex_dense_round2c_20261010/part_d"}
```

Saved remotely:
- `part_d/round2_cascade_dead.json`
- `part_d/round2_cascade_dead.log`
- `part_d/part_d_receipt.json`

After execution, SSH closed before packaging/retrieval. A fresh connection is reset before authentication. The metrics JSON/log have **not yet been uploaded**, so this message is an execution status, not the result package. No numerical dead-parent conclusions are claimed here. The pending full delivery is message013.

CPU analysis only; no GPU work, model replay, regeneration, threshold changes, or training modifications. The user explicitly cancelled scheduled tasks: no hourly follow-up or automatic dispatch. Future final raw/EMA and Part B outputs will be analyzed when the user requests it. A renewed SSH entry with access to the shared disk is needed to finish delivery; no GPU allocation is needed for this upload.
