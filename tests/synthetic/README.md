# Synthetic Test Data

This directory contains full pipeline tests using synthetic data with known ground truth.

## TODO: Specify Synthetic Data Generation

The following needs to be defined:

1. **Camera configurations to test**:
   - Number of cameras (2, 4, 8, 14)
   - Camera arrangement (planar array, varying heights)
   - Intrinsic parameter ranges
   - Distortion coefficient ranges

2. **Interface configurations**:
   - Water surface heights
   - Per-camera height variations
   - Optional interface tilt

3. **Board trajectory**:
   - Path through underwater volume
   - Visibility patterns (which cameras see board in each frame)
   - Ensuring pose graph connectivity

4. **Noise models**:
   - Detection noise (pixels)
   - Subpixel corner accuracy

5. **Ground truth format**:
   - Exact camera poses
   - Exact interface parameters
   - Exact board poses per frame

6. **Test scenarios**:
   - Ideal conditions (low noise)
   - Realistic conditions (typical noise)
   - Challenging conditions (high noise, sparse observations)
