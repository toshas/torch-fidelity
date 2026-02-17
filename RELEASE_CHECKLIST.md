# Release Checklist

Steps to publish a new release of `torch-fidelity`.

## Pre-release

These edits can (and should) land as a normal PR before the release commit.

### Update files

- [ ] Version string updated in `torch_fidelity/version.py`
  - Remove `-beta` or other pre-release suffixes
  - Follows [Semantic Versioning](https://semver.org/)
- [ ] `CHANGELOG.md` is up to date
  - Move `[X.Y.Z] - Unreleased` to `[X.Y.Z] - YYYY-MM-DD`
  - Add a new `[X.Y+1.0] - Unreleased` section at the top for future changes
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
  - The DOI must be the **concept DOI** (`10.5281/zenodo.3786539`), which is
    stable and always resolves to the latest release. If a citation block
    contains a version-specific DOI (e.g., `10.5281/zenodo.NNNNNNN`),
    replace it with the concept DOI
- [ ] Update `tests/torch_pure/Dockerfile` to use the latest NVIDIA PyTorch release
  - Check https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch for
    the latest tag
  - Update the `FROM nvcr.io/nvidia/pytorch:` tag
- [ ] `MANIFEST.in` includes all necessary non-Python files
- [ ] Grep for lingering old version strings
  ```bash
  # Replace OLD_VERSION with the previous release (e.g., 0.3.0)
  grep -rn "OLD_VERSION" --include="*.py" --include="*.cfg" --include="*.toml" \
       --include="*.rst" --include="*.md" --include="*.yml" --include="*.yaml" .
  ```
  - Check results for any stale version references that should have been
    updated (version.py, setup.py, documentation, CI configs, etc.)
  - Ignore expected occurrences (e.g., CHANGELOG.md historical entries)

### Verify

Once the pre-release PR is merged to `master`:

- [ ] All CI checks pass on `master` (CircleCI smoke tests)
- [ ] Full test suite passes locally (`tests/run_tests.sh` — requires Docker + GPU)
  - This also covers documentation building (`sphinx_doc` test flavor);
    visually inspect the output too:
    ```bash
    cd doc/sphinx && make html && open build/html/index.html
    ```
  - All test flavors except `tf1` run with `-W error`, so deprecation and
    other warnings from updated dependencies will surface as failures
- [ ] No untracked or uncommitted changes (`git status` is clean)

## Tag the release

Merge the pre-release PR to `master`, then tag the merge commit:

```bash
git checkout master
git pull origin master
git tag -a vX.Y.Z -m "Release vX.Y.Z"
```

## Publish to PyPI

Publishing is automated via GitHub Actions (`.github/workflows/publish.yml`)
using [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/).
Pushing a `v*` tag triggers the workflow, which builds and uploads to PyPI — no
API tokens needed.

- [ ] Validate the package locally before pushing the tag:
  ```bash
  python -m build && twine check dist/*
  ```
- [ ] Push the tag (`git push origin master --tags`) — this triggers the workflow
- [ ] Verify the package on PyPI: https://pypi.org/project/torch-fidelity/X.Y.Z/
- [ ] Install from PyPI and run a quick sanity check:
  ```bash
  pip install torch-fidelity==X.Y.Z
  python -c "import torch_fidelity; print(torch_fidelity.__version__)"
  ```

**If it fails:** PyPI does not allow re-uploading the same version filename,
ever. If the workflow failed before anything landed, fix and re-push the tag.
If a broken package was uploaded, yank it in the PyPI UI and release a patch
version (e.g., `X.Y.1`).

## Publish to Zenodo

Automated via GitHub Actions (`.github/workflows/zenodo.yml`). Creating a
GitHub Release triggers the workflow, which uploads a new version under the
concept DOI.

- [ ] Create a GitHub Release at https://github.com/toshas/torch-fidelity/releases/new
  - Select the `vX.Y.Z` tag
  - Copy the relevant `CHANGELOG.md` section as the release body
- [ ] Verify the Zenodo record — check that a new version appeared under the
  concept DOI: https://zenodo.org/doi/10.5281/zenodo.3786539

**If it fails:** Zenodo is forgiving. Check https://zenodo.org/uploads for
stuck drafts and delete them. Published versions can be deleted within 30 days,
and metadata can be edited in-place at any time. To retry, delete the failed
draft/version and re-run the workflow (or upload manually via
https://zenodo.org/deposit — create a new version of the existing record).

## Post-release

- [ ] Bump version in `torch_fidelity/version.py` to the next development version (e.g., `X.Y+1.0-dev`) and commit to `master`
