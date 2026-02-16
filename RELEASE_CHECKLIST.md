# Release Checklist

Steps to publish a new release of `torch-fidelity`.

## Pre-release

### Update files

- [ ] `CHANGELOG.md` is up to date
  - Move `[X.Y.Z] - Unreleased` to `[X.Y.Z] - YYYY-MM-DD`
  - Add a new `[Next] - Unreleased` section at the top (if desired)
- [ ] Version string updated in `torch_fidelity/version.py`
  - Remove `-beta` or other pre-release suffixes
  - Follows [Semantic Versioning](https://semver.org/)
- [ ] `setup.py` metadata is correct
  - If major features were added, update `long_description` to reflect the
    current scope (e.g., broaden from "GANs" to "generative image models"
    including diffusion, flow-matching, etc.)
  - Review `keywords` — add any newly relevant terms (e.g., `diffusion`,
    `flow-matching`, `dinov2`, `clip`) and remove outdated ones
- [ ] `README.md` reflects the changelog for major features
  - Verify that wording matches the current scope (e.g., "GAN evaluation"
    → "generative image model evaluation" if applicable)
  - Ensure new metrics, feature extractors, or capabilities are mentioned
- [ ] Review user-facing documentation for quality
  - `README.md` and ReadTheDocs (`doc/sphinx/`) are well structured, clear,
    and free of mistakes
  - No bloated or outdated sections; no major rewrites — just light cleanup
- [ ] Update citation blocks (`README.md` and `doc/sphinx/source/index.rst`)
  - Set `version={vX.Y.Z}` and update `note` to match
  - The concept DOI (`10.5281/zenodo.4957738`) stays as-is — it is stable
    and always resolves to the latest release
- [ ] Update test Dockerfiles to use the latest NVIDIA PyTorch release
  - Check https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch for
    the latest tag (currently `23.12-py3` in most Dockerfiles)
  - Update the `FROM nvcr.io/nvidia/pytorch:` tag in:
    - `tests/torch_pure/Dockerfile`
    - `tests/clip/Dockerfile`
    - `tests/torch_versions_ge_1_11_0/Dockerfile`
    - `tests/prc_ppl_reference/Dockerfile` (currently on older `21.02-py3`)
  - Note: `tests/tf1/Dockerfile` uses `19.02-py3` intentionally (TF1 compat)
- [ ] `MANIFEST.in` includes all necessary non-Python files

### Verify

- [ ] All CI checks pass on `master` (CircleCI smoke tests)
- [ ] Full test suite passes locally (`tests/run_tests.sh` — requires Docker + GPU)
  - This also covers documentation building (`sphinx_doc` test flavor)
- [ ] No untracked or uncommitted changes (`git status` is clean)

## Create the release commit

```bash
# 1. Update version
echo '__version__ = "X.Y.Z"' > torch_fidelity/version.py

# 2. Update CHANGELOG.md (replace "Unreleased" with today's date)

# 3. Commit
git add torch_fidelity/version.py CHANGELOG.md
git commit -m "Release vX.Y.Z"

# 4. Tag
git tag -a vX.Y.Z -m "Release vX.Y.Z"

# 5. Push commit and tag (the tag push triggers the PyPI publish workflow)
git push origin master --tags
```

## Publish to PyPI

Publishing is automated via GitHub Actions (`.github/workflows/publish.yml`)
using [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/).
Pushing a `v*` tag to any branch triggers the workflow, which builds and uploads
to PyPI — no API tokens needed.

### Manual publish (fallback)

If you ever need to publish without GitHub Actions:

```bash
pip install build twine

# Build
python -m build

# Check the package
twine check dist/*

# Upload to Test PyPI first (optional but recommended)
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*
```

## Post-release

- [ ] Verify the package on PyPI: https://pypi.org/project/torch-fidelity/X.Y.Z/
- [ ] Install from PyPI and run a quick sanity check:
  ```bash
  pip install torch-fidelity==X.Y.Z
  python -c "import torch_fidelity; print(torch_fidelity.__version__)"
  ```
- [ ] Create a GitHub Release from the tag at https://github.com/toshas/torch-fidelity/releases/new
  - Select the `vX.Y.Z` tag
  - Copy the relevant `CHANGELOG.md` section as the release body
  - This automatically triggers the Zenodo upload
    (`.github/workflows/zenodo.yml` runs on `release: [published]`)
- [ ] Verify the Zenodo record
  - Check that a new version appeared under the concept DOI:
    https://zenodo.org/records/4957738
  - If the workflow failed, upload manually: https://zenodo.org/deposit
    (create a new version of the existing record, upload source, publish)
- [ ] Bump version in `torch_fidelity/version.py` to the next development version (e.g., `X.Y+1.0-dev`) and commit to `master`
