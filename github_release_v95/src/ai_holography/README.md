# AI-First SLM Holography

This folder is a clean rewrite path for the Bowman/SLM CG workflow.

Goals:
- keep the Fourier optics model explicit
- move optimization to a modern PyTorch pipeline
- use AI wherever it is genuinely useful
- preserve a hybrid path: neural initialization + differentiable refinement

Recommended workflow:
1. Build target amplitude, target phase, and weighting masks.
2. Use `PhaseInitNet` to predict an initial SLM phase map.
3. Refine the phase with differentiable propagation and composite losses.
4. Optionally add camera feedback, surrogate models, or reinforcement learning later.

Main files:
- `config.py`: shared configuration
- `targets.py`: target and mask generation
- `propagation.py`: differentiable SLM to output-plane propagation
- `models.py`: neural phase initializer
- `losses.py`: overlap, intensity, phase, and regularization losses
- `pipeline.py`: end-to-end AI-heavy optimization pipeline

Why this structure is better than editing the old script directly:
- no Theano/PyTensor compatibility burden
- easier to add CNNs, diffusion priors, surrogate models, and RL
- easier to train from synthetic or measured data
- easier to swap optimization strategies
