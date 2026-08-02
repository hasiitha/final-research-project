# Deployment bundle

These seven files **are committed to this repository** — together they are
80 KB, so `git clone` gives you a runnable system with no extra downloads.

| File | Role |
|---|---|
| `head_state_dict.pth` | The only trained weights (~38 KB). Both encoders are frozen public checkpoints and are not stored here. |
| `bundle_config.json` | Architecture, labels, tokenizer, image transform. |
| `thresholds.json` | Validation-tuned per-label decision thresholds (0.04 – 0.29, not 0.5). |
| `text_rule.json` | The pre-diagnostic section regex, identical to the one used at training time. |
| `model_card.json` | Achieved ε, δ, noise multiplier, placement, per-label metrics, cohort, limitations. |
| `selection_ranking.csv` | Why this run was selected over the other DP runs. |
| `MANIFEST.json` | SHA-256 of each of the six files above. |

Verify the bundle before starting the service:

```bash
python3 dp_cxr_service/preflight.py
```

It checks that every file is present, that each checksum matches `MANIFEST.json`,
that the head tensors match the `fusion_type` recorded in the config, and that
the thresholds were tuned rather than left at 0.5.

To serve a different exported model without touching the code, point
`DP_CXR_BUNDLE` at another directory.
