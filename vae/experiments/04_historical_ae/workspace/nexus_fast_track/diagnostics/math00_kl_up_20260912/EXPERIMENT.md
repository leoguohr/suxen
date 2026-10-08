# Hold / Up paired continuation

Both branches start from `../math00_kl_warm_20260912/checkpoint-update0200.pt`. Its SHA is pinned in each branch's `math00_kl_warm_step200.json`. The previous small-beta success remains valid within its tested budget (`retained_success_baseline.json`). This is a bounded continuation test, not a requirement to reach beta=1 or sigma=1.

- Hold: beta=1e-4 for all200 updates.
- Up: update1 beta=1e-4; linear to3e-4 on update50; hold through update200.
- All five Adam groups restored exactly: parent E/decoder/heads step600, logvar step400. Final steps800/600.
- Restore parent training RNG after800 noise draws. Both branches use the same next400 tensors: two meshes, one new epsilon each per update. Forward/recompute shares each update's epsilon; evaluation never advances the training generator.
- E/mu LR1e-8; decoder/heads1e-7; logvar1e-4; unchanged clip1, wd0, clamp[-20,10], fully-differentiable Edge+Face Soft4, equal-per-mesh KL reduction, deterministic math00 across encoder and decoder.
- Mu and original fixed-noise diagnostics at0,1,10,20,50,100,200. Original monitoring50 at0,20,50,100,200. New final50 seed pairs10060000..10060099, identical for both branches and distinct from prior final-only sets.
- Strict success: each draw is one packed forward; both meshes' Edge and actual Face FP=FN=0, with Face rebuilt from that forward's predicted Edge graph. Main checkpoints50/100/200 require mu perfect and monitoring50/50; final new50/50 also required. Nonzero actual five-group updates and sampling perturbation retained.
- No candidate VJP, line search or extra gradient sweeps. Existing per-step group gradient norms, global clipping coefficients, actual parameter displacement, posterior/raw logvar/clamp, noise sizes, and four reconstruction parts/Kmu/Ksigma are logged.
- No comparisons of the branches' differently weighted training totals. `analyze.py` computes common-beta evaluations from the existing monitoring reconstruction and KL values, without another forward.
- Both200-update runs are sequential on GPU0. `verify_pair.py` only reads checkpoints and replays RNG tensors; it never updates model parameters.

Hold passes and Up passes: increased-duration evidence at1e-4 and a new successful work point at3e-4.
Hold passes and Up fails: stronger-beta schedule did not pass this budget; retain Hold. This does not prove3e-4 impossible.
Hold fails: longer continuation exposes a time-stability limitation even at1e-4; increased beta alone cannot explain it.

Large checkpoint/noise/NPZ artifacts stay on the server under `/guohaoran/nexus_fast_track/diagnostics/`. Metrics, manifests, scripts and comparisons are copied locally. Original experiments and formal model source are preserved.
