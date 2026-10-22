# Third-party source and redistribution notices

This file is an attribution/review inventory, not the license grant for original study code.

| Component | Source | License evidence | Treatment |
|---|---|---|---|
| timm1.0.19 | https://github.com/huggingface/pytorch-image-models/tree/v1.0.19 | Apache-2.0, official LICENSE checked 2026-10-21 | Dependency only; no timm source/weights bundled. Preserve license/notices if later vendored. |
| mamba_ssm2.2.4 | https://github.com/state-spaces/mamba/tree/v2.2.4 | Apache-2.0, official LICENSE; copyright Tri Dao/Albert Gu | Dependency only; CUDA binary/source not bundled; do not replace with Identity for Mamba. |
| PyTorch2.5.1 | https://github.com/pytorch/pytorch/tree/v2.5.1 | Upstream BSD-style license and third-party notices require dependency-level review | No binaries/source bundled. |
| NumPy/pandas/scikit-learn/OpenCV | Official projects; historical versions listed in environment/ | Project-specific notices apply; dependency artifacts not redistributed | Obtain from official distribution; review exact package notices. |
| torchvision InceptionV3 weights; timm EfficientNet-Lite2/MobileViT weights | URLs/registries and SHA in data/pretrained_sources.csv | Model weights may have terms distinct from framework code; full model-card redistribution review pending | No pretrained/fine-tuned weights bundled. |
| MixStyle/GeM and local composite implementation | Historical research module, exact extracted classes | Original author/contributor ownership and upstream attribution not fully established by local records | Author/institution review before public release. No license inferred from dependencies. |
| ORCHID images | Two official Zenodo DOIs in DATA_ACCESS.md | CC BY4.0 confirmed by official API | No images bundled. Required credits/change descriptions recorded; not a code license or IRB conclusion. |

Apache license sources: https://github.com/huggingface/pytorch-image-models/blob/v1.0.19/LICENSE and https://github.com/state-spaces/mamba/blob/v2.2.4/LICENSE .
