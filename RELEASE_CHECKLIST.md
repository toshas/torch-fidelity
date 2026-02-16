# Release Checklist

Steps to publish a new release of `torch-fidelity`.

## Pre-release

- [ ] All CI checks pass on `master` (CircleCI smoke tests + PRC convention test)
- [ ] Full test suite passes locally (`tests/run_tests.sh` — requires Docker + GPU)
- [ ] `CHANGELOG.md` is up to date
  - Move `[X.Y.Z] - Unreleased` to `[X.Y.Z] - YYYY-MM-DD`
  - Add a new `[Next] - Unreleased` section at the top (if desired)
- [ ] Version string updated in `torch_fidelity/version.py`
  - Remove `-beta` or other pre-release suffixes
  - Follows [Semantic Versioning](https://semver.org/)
- [ ] `setup.py` metadata is correct (description, URL, keywords, classifiers)
- [ ] `MANIFEST.in` includes all necessary non-Python files
- [ ] Documentation builds cleanly (`cd doc/sphinx && make html`)
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

# 5. Push
git push origin master --tags
```

## Publish to PyPI

Publishing is automated via GitHub Actions (`.github/workflows/publish.yml`).
Pushing a tag matching `v*` triggers the workflow, which builds and uploads to PyPI.

### First-time PyPI setup

1. **Create a PyPI account** at https://pypi.org/account/register/ (if you don't have one).
2. **Create a PyPI API token** scoped to the `torch_fidelity` project:
   - Go to https://pypi.org/manage/account/token/
   - Set scope to project `torch_fidelity`
   - Copy the token (starts with `pypi-`)
3. **Add the token as a GitHub Actions secret:**
   - Go to your repo → Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `PYPI_API_TOKEN`
   - Value: paste the token from step 2
4. **Alternative — Trusted Publishers (recommended):**
   Instead of an API token, you can use PyPI's Trusted Publisher mechanism
   which is more secure and doesn't require managing tokens:
   - Go to https://pypi.org/manage/project/torch_fidelity/settings/publishing/
   - Add a new publisher:
     - Owner: `toshas`
     - Repository: `torch-fidelity`
     - Workflow name: `publish.yml`
     - Environment: `pypi`
   - Then in the GitHub Actions workflow, replace the token-based upload with
     the `id-token: write` permission (already included in the workflow).

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
- [ ] Bump version in `torch_fidelity/version.py` to the next development version (e.g., `X.Y+1.0-dev`) and commit to `master`
