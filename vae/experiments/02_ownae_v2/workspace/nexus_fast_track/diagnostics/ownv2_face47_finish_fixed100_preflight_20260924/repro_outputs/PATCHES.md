# Implementation scope

Branch: repro/2026-09-24-face47-preflight.
Implementation commit:2791097.

New standalone face_finish.py and preflight100.py implement the user-approved
bounded changes. native_models.py, data_objective.py, run_support.py and
evaluate_checkpoint.py are byte-identical copies of the completed prior run.
Original source, README, pools and all historical checkpoints remain untouched.

Validation: Python compilation; full parent model and all Adam slots restored
exactly; real parent prediction equality and full/cache Face gradient gate
before the first update; frozen-model/optimizer hash checks at every checkpoint;
complete actual mesh evaluation every50 updates. See runtime records for actual
pass/fail outcomes, rather than treating this list as proof of completion.

Highest relevant risk: inference array equality or gradient equivalence could
fail on the new device. Such failure stops this branch; it does not authorize
changing the backend, optimizer, threshold or scientific protocol.
