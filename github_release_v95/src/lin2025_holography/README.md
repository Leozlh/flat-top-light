# Lin 2025 Holography

This folder rewrites the hologram workflow along the main ideas of Lin et al. 2025:

1. Generate training labels with a classical weighted Gerchberg-Saxton solver.
2. Convert the hologram labels into amplitude and phase in the position domain.
3. Encode trap coordinates and target phases as position-domain input images.
4. Train a lightweight CNN to predict position-domain amplitude and phase.
5. Recover the final hologram by inverse FFT / FFT-domain reconstruction.

Main files:

- `config.py`: configuration for supervised hologram learning
- `wgs.py`: weighted GS label generator
- `encoding.py`: position-domain input encoding from trap coordinates and phases
- `model.py`: lightweight residual CNN
- `dataset.py`: on-the-fly WGS supervision dataset
- `training.py`: supervised training loop
- `inference.py`: demo inference and visualization
- `scripts/train_lin2025.py`: training entry
- `scripts/run_lin2025_demo.py`: demo entry

