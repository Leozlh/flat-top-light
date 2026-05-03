# AI + Physics Architecture

## Goal

This stack is designed around three physical goals:
- pattern uniformity inside the active region
- phase flatness or target phase consistency
- light usage efficiency

For experimental use, these goals must be interpreted through hardware limits:
- non-ideal SLM phase response
- static aberration or correction maps
- camera noise and drift
- mismatch between Fourier model and real optics

## Training architecture

Files:
- `training.py`
- `models.py`
- `propagation.py`
- `losses.py`

Flow:
1. A synthetic Bowman-style target is sampled online.
2. `PhaseInitNet` receives:
   - target amplitude
   - target phase
   - active-region weighting
3. The network predicts an initial SLM phase.
4. The predicted phase is passed through a differentiable Fourier propagation layer.
5. The output field is compared to the target using a composite physics-informed loss.
6. The network weights are updated with AdamW.

This is supervised by physics, not by stored phase labels.
The propagation model acts as the supervision path.

## Inference architecture

Files:
- `pipeline.py`
- `hybrid.py`
- `runner.py`

Main production path:
1. Load trained `PhaseInitNet`.
2. Predict a neural initial phase.
3. Optionally refine with multiscale differentiable optimization.
4. Run short conjugate-gradient polish on the final phase.
5. Save phase, metrics, and visualizations.

For repeated nearby targets:
1. Compute target similarity.
2. Compare candidate initializations:
   - neural init
   - previous phase
   - blended init
3. Keep the best candidate.
4. Continue with refinement and short CG polish.

For experimental deployment, there are two extra hooks:
- `calibration.py`: adds a static SLM correction phase
- `camera_loop.py`: adjusts weighting using measured camera intensity

## Loss design

The current composite loss includes:
- overlap loss: field fidelity to the desired complex target
- intensity loss: amplitude matching
- phase loss: wrapped phase consistency
- uniformity loss: local intensity uniformity in the weighted region
- efficiency loss: fraction of light in the useful region
- total variation loss: smoother SLM phase

This combination is intended to balance:
- Bowman-style fidelity optimization
- phase regularity
- trap or pattern uniformity
- practical optical efficiency

## How this relates to the literature

The structure is intentionally hybrid:

- Bowman et al. 2017:
  direct optimization of a cost function for high-fidelity amplitude and phase control
- Harte et al. 2014:
  careful cost-function engineering for better optimization behavior and vortex suppression
- Peng et al. 2020 Neural Holography:
  learned initialization and camera-in-the-loop style thinking for model mismatch
- weighted / phase-induced GS work:
  practical emphasis on uniformity, continuity, and fast iterative updates

## What to improve next

- Train the network on target sequences, not independent targets.
- Predict phase residuals relative to a warm-start phase.
- Add a learned efficiency or camera surrogate model.
- Add camera-in-the-loop correction if experimental images are available.
- Use lower-resolution candidate scoring for faster warm-start routing.

## Experimental priorities

If the goal is a real setup rather than benchmark generality, optimize in this order:
1. stable output under repeat runs
2. useful-region efficiency
3. acceptable uniformity
4. target phase fidelity

In practice, a perfectly uniform simulation can be a bad experimental operating point if it throws away too much light.
For that reason, the `experiment` profile intentionally biases more toward efficiency and robustness than the `uniformity` profile.
