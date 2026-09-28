# Releasing

Releases are cut on GitHub. Publishing a GitHub Release runs
`.github/workflows/release.yml`, which builds the sdist and wheel, smoke-tests
the wheel, uploads to PyPI via trusted publishing, and attaches the files to
the release. No PyPI token is stored anywhere.

## Cutting a release

1. Bump `version` in `pyproject.toml` and commit it to `main`:

   ```sh
   uv version 0.5.0            # or edit pyproject.toml by hand
   git commit -am "chore: release v0.5.0"
   git push
   ```

2. Wait for CI on that commit to go green.

3. Publish the release. The tag is created for you and must be `v` plus the
   `pyproject.toml` version, or the workflow fails before building:

   ```sh
   gh release create v0.5.0 --generate-notes
   ```

   Or use the "Draft a new release" button on GitHub, which lets you edit the
   generated notes before publishing.

4. Watch the run with `gh run watch` (or the Actions tab). When it finishes,
   the new version is on PyPI and the wheel and sdist hang off the release.

## One-time setup

Already done in the repo: the workflows and the `pypi` deployment environment.

On PyPI, the project must trust this workflow. Go to
<https://pypi.org/manage/project/dspy-monty-interpreter/settings/publishing/>
and add a GitHub publisher with:

| Field            | Value                   |
|------------------|-------------------------|
| Owner            | `dbreunig`              |
| Repository name  | `dspy-monty-interpreter` |
| Workflow name    | `release.yml`           |
| Environment name | `pypi`                  |

Until that publisher exists, the publish job fails with an "invalid-publisher"
error and nothing reaches PyPI. The build job still runs and the release stays
published, so delete the release and tag, add the publisher, and publish again.
