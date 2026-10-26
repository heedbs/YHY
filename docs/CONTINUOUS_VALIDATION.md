# Continuous Validation

The [CPU workflow](../.github/workflows/cpu-validation.yml) checks pushes and pull requests; manual dispatch is also available. It uses a GitHub-hosted Ubuntu CPU runner, read-only repository permissions, no secrets, no artifact upload and SHA-pinned official checkout/Python actions. It never connects to the research server or reads patient data, weights or cache.

Checks: complete candidate SHA coverage and heuristic hygiene scan; Markdown file links and source-index hashes; dependency consistency; synthetic and public aggregate metadata tests. The candidate is exported to a temporary directory without `.git`; the environment is outside the candidate. The package scanner intentionally remains strict rather than silently excluding research files.

Local equivalents, in the documented [CPU environment](../environment/CPU_TEST_ENVIRONMENT.md):

```bash
python -B scripts/check_public_package.py . --manifest
python -B scripts/check_evidence_links.py .
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -I -B -m unittest discover -s tests -v
```

External URL availability and Markdown heading anchors are not tested by the link checker. CPU tests do not execute research training, CUDA forward/backward, bootstrap or benchmark. A local pass is not a GitHub Actions pass. Check the [Actions page](https://github.com/heedbs/YHY/actions) for the actual run on the reviewed commit; Actions can be disabled or restricted by account policy or allowance. Remote dependency installation is not proven until that run succeeds.
