# Release Checklist

Steps to publish a new release of `torch-fidelity`.

## Pre-release

- [ ] All CI checks pass on `master` (CircleCI smoke tests)
- [ ] Full test suite passes locally (`tests/run_tests.sh` — requires Docker + GPU)
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
  - Documentation builds cleanly (`cd doc/sphinx && make html`)
- [ ] Update citation blocks with the new version
  - In `README.md`: set `version={vX.Y.Z}`, erase `doi` and `note` fields
    (a new DOI will be minted after the Zenodo upload)
  - In `doc/sphinx/source/index.rst`: same changes
- [ ] `MANIFEST.in` includes all necessary non-Python files
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
  - Creating the release triggers the Zenodo upload (see below)
- [ ] Upload to Zenodo
  - If using the Zenodo–GitHub integration: creating the GitHub Release
    (above) automatically triggers a Zenodo deposit and mints a new DOI.
    Verify at https://zenodo.org/records/ that the new version appears.
  - If using the GitHub Actions workflow (`.github/workflows/zenodo.yml`):
    the upload happens automatically on `release: [published]`.
    Check the workflow run for success.
  - If uploading manually: go to
    https://zenodo.org/deposit, create a new version of the existing record,
    upload the source archive, and publish.
- [ ] Backfill the DOI into citation blocks
  - Once the Zenodo DOI is minted, update `doi` and `note` fields in
    `README.md` and `doc/sphinx/source/index.rst` with the new DOI
  - Commit to `master` (this is a metadata-only follow-up commit)
- [ ] Bump version in `torch_fidelity/version.py` to the next development version (e.g., `X.Y+1.0-dev`) and commit to `master`
