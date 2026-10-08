# math00 fresh-epsilon small-beta paired continuation

Starting checkpoint: `../math00_fresh_sampling_20260912/checkpoint-update0200.pt`, pinned by SHA-256 in each branch's `math00_fresh_eps_step200.json`. Both branches restore all five Adam groups and the saved training generator state (400 prior draws). No model/logvar reinitialization.

- Control: `../math00_kl_control_20260912`, beta=0.
- KL: this directory, beta=0 on update1, linear to 1e-4 on update50, constant through update200.
- Both execute 200 actual updates, sequentially on GPU0, with identical evolving epsilon tensors. The generator continues from the parent; evaluation does not advance it.
- Encoder/mu LR1e-8; decoder/body and heads LR1e-7; logvar LR1e-4; Adam defaults/state preserved, wd0, global clip1.
- math00 and deterministic Graph unchanged, FP32 MATH attention in encoder and decoder; no autocast or TF32. Fully-differentiable Edge+Face Soft4; KL is first averaged within each mesh, then equally over two meshes.
- Training Face uses the unchanged full training pool. Evaluation Face is rebuilt from each forward's predicted Edge graph. Success is both meshes' Edge+Face FP=FN=0 in one packed forward.
- Monitoring seeds are unchanged. Final unseen seeds10050000..10050099 are distinct from both the old monitoring and the previous final-only sets. They never train the model.

Before launch, `preflight.json` records the next training noise, unweighted KL and independent gradient norms, with no optimizer step. The RNG is restored after this probe. `beta_decision.json` pins the target before training.

Independent gradient probes occur on parameter states0,20,50,100,199 (before updates1,21,51,101,200). They separately report reconstruction, KL, K_mu and K_sigma norms. All actual training updates additionally report preclip group norms, clipping coefficient, parameter displacement, sigma/clamp distribution, epsilon/RNG hashes, and reconstruction/KL/total separately. The control's inherited manifest `kl_gradient_checks` says200 for its last entry; the per-record `parameter_state_for_training_forward=199` is the authoritative actual state. No extra update is run for this probe.

`verify_pair.py` performs read-only exact checks of checkpoint provenance, five Adam states, generator continuation, all 400 noise tensors, and equality of the two branches' noise streams. `analyze.py` generates comparison.json and comparison.png from completed logs.

Large raw checkpoints and noise tensors remain on the server under `/guohaoran/nexus_fast_track/diagnostics/`; metrics, scripts, provenance and plots are copied locally. Existing experiments and formal model sources are unchanged.
