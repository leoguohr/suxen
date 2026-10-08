# E0/E1 implementation review — release for execution

Reviewed source on 2026-10-03:

| File | SHA256 |
|---|---|
| `diagnose_capability.py` | `892a1d54138b0afcd300deaebd4b202395a8ff60cba4a38f7ea15281facba8c0` |
| `aggregate_capability.py` | `87c7b688ae6cb143085dc5cd227b067a7883bd92a4bc21fc62d5e4637ece3b23` |
| `protocol_review.json` | `1bb6f5fa9788a1212ebaf66173a725ccf304d66edd206de7fd30f4a3b0423366` |

No remaining blocking source-review finding for E0/E1 at these hashes. This is permission to execute the already-authorized diagnostic, not a report that actual model replay has passed. The reviewer performed no GPU forward, training or SSH.

The second source inspection confirmed the requested fixes: full saved 20-step Euler recurrence is checked bitwise; both t0 paths freshly call the model and are directly compared; E1 compares first-correct against actual saved E0 predictions, restores the correct condition with another fresh call, and preserves all fixed inputs; global E0 pins the equation-review hash; diagnostic JSON and prediction arrays are linked by hashes; UID-level paired effects first average the two seeds and report direction/quantiles plus both-seed exact counts. Sample endpoint-reference velocity remains explicitly distinct from the interpolation training target.

Final integrity additions were also inspected before release: donor condition/cloud/context hashes are explicitly recorded beside each recipient, and E1 revalidates E0 shard provenance/manifest/context-manifest files against their saved gate hashes.

The implementation agent reports 12 passing CPU tests, including rejection of a fresh-t0 pair that disagrees despite both values matching the archived output separately, and rejection of an E1 baseline that matches archived output but fails direct comparison to E0. The agent also reports successful CPU formula/Euler verification on 8 archived level pairs (first UID, both seeds, depths 6–9). These are CPU/code checks, not the required 2400 E0 or 7200 E1 model forwards.

Freeze `protocol_review.json` before E0. Do not edit it after E0 begins; future E2 implementation findings belong in a separate review file. E2 still requires complete E0/E1 results, global E0 numerical pass, and verified full A24000 model/Adam/RNG restoration.
