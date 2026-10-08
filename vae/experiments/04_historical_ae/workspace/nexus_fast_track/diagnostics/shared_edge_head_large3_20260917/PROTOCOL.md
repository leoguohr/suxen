# Fixed hidden, one shared Edge head

Starting network: round2 Face hard-negative epoch900/update22500, SHA256 specified in config.json. Fixed UIDs: 000446/001093/000898 (1519/1964/2547 vertices). Original Control and round2 model files remain unchanged.

Export exact inputs to the original nn.Linear Edge head after decoder_output_norm, original weight and bias, and per-mesh Edge gradients. Validate raw output, centered embedding, loss, gradients and all-pair counts against the prior network snapshot before any update. A mismatch stops execution.

Use one FP32 nn.Linear(1024,32,bias=True) for every mesh. No independent heads; no successful-embedding MSE target. All hidden tensors are detached constants. Preserve current 16+16 scoring, scale, centering, all unordered pairs, FP32 fully differentiable Soft4, threshold zero. No Face or KL.

Selected finite candidate settings: fresh Adam lr=1e-4, default betas/eps, wd=0, clip=1, 2000 updates. This LR is not inherited from the free-embedding probe or claimed optimal. Every update accumulates the three equally weighted mesh losses, clips once and updates the shared head once. Objective: mean_mesh(Edge Soft4)/4, retaining the previous probe's external coefficient. Soft4 also retains its own internal four-group average.

Record all-pair TP/FP/FN/TN, F1, Soft4 and minimum signed margins for all three after every update; record actual weight/bias and embedding changes, gradient norms, clip and head norm. Save steps 0/50/100/200/500/1000/2000 and the first joint-perfect state. Joint success requires all three FP=FN=0 using the same head at the same update. Continue to the fixed budget if success arrives early; stop at 2000.

If the server is reachable, upload this directory, run `prepare_remote.py`, then launch `run_job.py` with the server's /opt/conda/bin/python. The job exports, trains, reload-verifies, and packages. It refuses to overwrite an existing run. Check current GPU ownership before launch; do not stop another task to make room.

Status at local preparation: files prepared and syntax checked; actual export and CUDA execution require the server. No claim of baseline or training validation is made until corresponding output records exist.
