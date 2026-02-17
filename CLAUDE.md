# CLAUDE.md

## Project Overview

**torch-fidelity** is a PyTorch library providing epsilon-exact implementations of generative model evaluation metrics: Inception Score (ISC), Frechet Inception Distance (FID), Kernel Inception Distance (KID), Precision/Recall/F-score (PRC), and Perceptual Path Length (PPL). The library prioritizes numerical fidelity with reference TensorFlow implementations.

Current version: **0.4.0** (`torch_fidelity/version.py`)

## Repository Structure

```
torch_fidelity/           # Main package (~4100 lines, 29 modules)
  metrics.py              # Orchestration: calculate_metrics() entry point
  metric_isc.py           # Inception Score
  metric_fid.py           # Frechet Inception Distance
  metric_kid.py           # Kernel Inception Distance (poly/rbf kernels)
  metric_prc.py           # Precision, Recall, F-score
  metric_ppl.py           # Perceptual Path Length
  feature_extractor_base.py         # Abstract base for feature extractors
  feature_extractor_inceptionv3.py  # InceptionV3 (TF-compatible weights)
  feature_extractor_clip.py         # CLIP feature extraction
  feature_extractor_vgg16.py        # VGG16 feature extraction
  feature_extractor_dinov2.py       # DinoV2 (4 variants)
  generative_model_base.py          # Abstract base for generative models
  generative_model_modulewrapper.py # PyTorch module wrapper
  generative_model_onnx.py          # ONNX/JIT model support
  sample_similarity_base.py         # Abstract base for similarity metrics
  sample_similarity_lpips.py        # LPIPS implementation
  registry.py             # Plugin registration system
  defaults.py             # All configuration defaults (~60 parameters)
  deprecations.py         # Deprecated parameter handling
  datasets.py             # Dataset wrappers (CIFAR-10/100, STL-10)
  noise.py                # Noise generators (normal, uniform, unit sphere)
  helpers.py              # vassert, vprint, get_kwarg utilities
  utils.py                # Feature extraction, caching, dataset handling
  utils_torch.py          # torch.compile support
  utils_torchvision.py    # TorchVision integration
  interpolate_compat_tensorflow.py  # TF-compatible bilinear interpolation
  fidelity.py             # CLI entry point
  version.py              # Version string
  __init__.py             # Public API exports

tests/                    # Test suite (42 test files)
  __init__.py             # TimeTrackingTestCase base class
  run_tests.sh            # Docker-based full test runner
  torch_pure/             # Pure PyTorch tests (batching, feature extractors, misc)
  tf1/                    # TF1 reference implementation comparison tests
  clip/                   # CLIP feature extractor tests
  prc_ppl_reference/      # PRC and PPL reference tests
  torch_versions_ge_1_13_0/  # PyTorch version compatibility tests
  sphinx_doc/             # Documentation build tests
  aws/                    # AWS test harness

examples/                 # Example training integrations
  sngan_cifar10.py        # SNGAN training with metric evaluation

doc/                      # Sphinx documentation source
.circleci/                # CI configuration
  config.yml              # CircleCI pipeline
  smoke_tests.py          # CI smoke tests
```

## Build and Development Commands

### Installation
```bash
pip install -e .
# or
pip install numpy pillow torch torchvision tqdm
```

### Running the CLI
```bash
# The package provides a `fidelity` console script
fidelity --input1 /path/to/images1 --input2 /path/to/images2 --fid --isc --kid

# Or run as module
python -m torch_fidelity.fidelity --input1 ... --input2 ... --fid
```

### Running Tests

**Smoke tests (CI-style, CPU-only):**
```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python .circleci/smoke_tests.py
```

**Full test suite (requires Docker + GPU):**
```bash
tests/run_tests.sh              # runs all suites except tf1
tests/run_tests.sh --with-tf1   # includes tf1 (requires pre-Ampere GPU)
```

The full suite uses Docker containers (NGC PyTorch base images) and runs six test flavors sequentially:

1. **`torch_versions_ge_1_13_0`** (CUDA, strict warnings) — Backward compatibility testing. Dynamically installs torch 1.13.1, 2.0.1, 2.1.1, and latest, running the full metrics pipeline (ISC/FID/KID/PRC with Inception-v3, CLIP, DINOv2) against each version. Verifies metric values stay within tight tolerances across versions. The Docker image strips all pre-installed torch packages and installs/uninstalls them per test. Minimum version is 1.13.0 due to CUDA sm_86 (Ampere) support requirement.

2. **`tf1`** (CUDA, no strict warnings) — **Skipped by default** (enable with `--with-tf1`). Legacy TensorFlow 1.14 numerical precision comparison. Uses an old NGC base image (`pytorch:19.02-py3`) with CUDA 10.0 to cross-validate against the original TF reference implementations. Requires pre-Ampere GPU (V100, T4, etc.) — fails on Ampere+ (sm_86+) due to missing cuBLAS kernels in CUDA 10.0. Warnings are not treated as errors because TF1/legacy code generates many deprecation warnings. These tests were validated up to and including v0.3.0; numerical correctness of v0.4.0 is ensured via the other test suites.

3. **`torch_pure`** (CUDA, strict warnings) — Core functionality tests (~135+ tests). Covers batch size independence (verifies metrics are identical across batch sizes in fp32/fp64), torch.compile() compatibility, all feature extractors (InceptionV3, VGG16, CLIP, DINOv2), all metrics, FID statistics edge cases, PRC convention correctness, generative model input (ISC/PPL with SNGAN), and LPIPS reference comparison (VGG16-based, validates against NVIDIA's pretrained module). Uses the latest torch from the NGC base image.

4. **`clip`** (CUDA, strict warnings) — CLIP feature extractor integration tests. Validates CLIP-based metric computation against the OpenAI CLIP model (pinned commit). Has a separate Docker image because it requires CLIP's dependencies (ftfy, regex, setuptools, clean-fid).

5. **`prc_ppl_reference`** (CUDA, strict warnings) — Reference implementation comparison for PRC and PPL metrics. Uses an older NGC base image (`pytorch:21.02-py3`) to compare against the original StyleGAN2-ADA reference implementation, ensuring epsilon-exact numerical agreement.

6. **`sphinx_doc`** (CPU) — Documentation build test. Uses the `sphinxdoc/sphinx` base image to verify the Sphinx documentation builds without errors. Runs shell-based tests (`test_*.sh`).

**Individual test discovery:**
```bash
python -W error -m unittest discover -s tests/<flavor> -t . -p 'test_*.py'
```

### Code Formatting
```bash
black --line-length 120 .
```

### Building Documentation
```bash
cd doc/sphinx && make html
```

## Code Style and Conventions

- **Formatter**: Black with line-length 120 (configured in `pyproject.toml`)
- **Python version**: >= 3.6
- **Assertions**: Use `vassert(condition, message)` from `helpers.py` instead of bare `assert`
- **Verbose output**: Use `vprint(verbose, message)` which prints to stderr
- **Configuration access**: Use `get_kwarg("name", kwargs)` to read parameters with defaults from `defaults.py`
- **Deprecations**: Handled via `process_deprecations()` and the `DEPRECATIONS` dict in `deprecations.py`
- **Test base class**: All tests extend `TimeTrackingTestCase` which tracks timing and clears CUDA cache

## Architecture

### Plugin Registry System

The registry (`registry.py`) supports five extension points:
- `register_dataset(name, fn_create)` - Custom datasets
- `register_feature_extractor(name, cls)` - Must subclass `FeatureExtractorBase`
- `register_sample_similarity(name, cls)` - Must subclass `SampleSimilarityBase`
- `register_noise_source(name, fn_generate)` - Noise generators
- `register_interpolation(name, fn_interpolate)` - Interpolation methods

Pre-registered components are in `registry.py` lines 143-199.

### Metric Computation Flow

```
Input → Feature Extraction (cached) → Metric Computation
```

- **Unary metrics** (ISC, PPL): require only `input1`
- **Binary metrics** (FID, KID, PRC): require `input1` and `input2`
- **PRC convention**: `input1` = generated (evaluated), `input2` = real (reference). Precision = fraction of generated samples in real manifold; recall = fraction of real samples in generated manifold
- Feature extraction results are cached to disk when `cache=True`
- FID has a shortcut path when statistics are cached but features are not
- When using default feature extractors, ISC/FID/KID use InceptionV3 and PRC uses VGG16; if both groups are requested, two separate feature extraction passes run automatically

### Feature Extractors

| Name | Class | Default For |
|------|-------|-------------|
| `inception-v3-compat` | `FeatureExtractorInceptionV3` | ISC, FID, KID |
| `vgg16` | `FeatureExtractorVGG16` | PRC |
| `clip-vit-b-32` (and other CLIP variants) | `FeatureExtractorCLIP` | - |
| `dinov2-vit-{s,b,l,g}-14` | `FeatureExtractorDinoV2` | - |

### Input Types

The `input1`/`input2` parameters accept:
- Registered dataset names (e.g., `cifar10-train`, `stl10-test`)
- Directory paths containing images
- ONNX/PTH model file paths
- `torch.utils.data.Dataset` instances
- `GenerativeModelBase` instances

## Key Design Decisions

- **Numerical fidelity**: The InceptionV3 implementation uses TF-compatible weights and a custom bilinear interpolation (`interpolate_compat_tensorflow.py`) to match TensorFlow output to machine precision
- **kwargs-based API**: All configuration flows through `**kwargs` checked against `defaults.py`; no dataclass or typed config objects
- **No scipy dependency**: Matrix square root for FID is implemented in pure PyTorch
- **Caching**: Multi-level caching (features and FID statistics) to avoid redundant computation
- **Verbose to stderr**: All progress output goes to stderr so stdout can be parsed as JSON

## CI/CD

- **CircleCI**: Python 3.11.7, large resource class, CPU-only smoke tests
- Runs on every push and weekly (Mondays) on master
- Smoke tests validate all metrics against known reference values with tight tolerances
- Tests use `psutil` for memory monitoring
- **Smoke test pattern**: all tests run via `_run_fidelity_command()` which wraps `subprocess.run`. Tests must not import `torch` or library internals directly; instead, run `python3 -m torch_fidelity.fidelity` or `python3 -c "..."` as a subprocess and assert on JSON output

## Common Pitfalls

- The `feature_extractor_compile` option is experimental and may affect numerical precision
- KID can produce negative values (this is mathematically expected)
- Lossy image formats (jpg/jpeg) trigger warnings since they affect metric precision
- When modifying feature extractors, ensure the output layer names remain consistent as they are used as cache keys
- The InceptionV3 implementation intentionally differs from torchvision's to maintain TF compatibility

## Code editing preferences

See [CONTRIBUTING.md](CONTRIBUTING.md#code-editing-guidelines).
