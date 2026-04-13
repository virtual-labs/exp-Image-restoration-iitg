## Theory

1. Gaussian Noise
- Random noise with normal distribution, often due to sensors/electronics.

2. Salt & Pepper Noise
- Sparse white and black pixels due to bit errors or transmission noise.

## Denoising Filters

- Gaussian Filter: Smooths Gaussian noise but may blur edges.
- Median Filter: Best for salt-pepper noise, preserves edges.
- Bilateral Filter: Smooths noise while maintaining edges using spatial + intensity weights.
- Non-Local Means (NLM): Averages similar patches, preserves textures.
