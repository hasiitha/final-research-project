#!/usr/bin/env python3
"""Validate a deployment bundle before starting the service.

Deliberately imports nothing beyond the standard library, so it runs in a bare
interpreter and tells you whether the bundle is sound *before* you spend time
installing the two-gigabyte ML stack.

    python dp_cxr_service/preflight.py
    python dp_cxr_service/preflight.py /path/to/bundle
"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

REQUIRED = ["bundle_config.json", "head_state_dict.pth"]
OPTIONAL = ["thresholds.json", "text_rule.json", "model_card.json",
            "selection_ranking.csv", "MANIFEST.json"]

# Which head tensors each fusion type must provide.
EXPECTED_KEYS = {
    "late": {"img_head.weight", "img_head.bias", "txt_head.weight", "txt_head.bias"},
    "early": {"mlp.0.weight", "mlp.0.bias", "mlp.3.weight", "mlp.3.bias"},
}

ok, warn, fail = [], [], []


def read_pth_keys(path: Path):
    """Pull tensor names out of a torch .pth without importing torch.

    A torch save is a zip whose data.pkl is a pickle of the state dict. The keys
    are stored as short strings, so scanning for the pickle's SHORT_BINUNICODE
    opcode recovers them without executing anything.
    """
    try:
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith("data.pkl"))
            blob = z.read(name)
    except Exception:
        return None
    # Strings are encoded differently by pickle protocol. torch.save defaults to
    # protocol 2, which uses BINUNICODE (4-byte length); protocol 4 uses
    # SHORT_BINUNICODE (1-byte). Handle every form or the scan silently finds
    # nothing and every key looks missing.
    OPCODES = {
        0x58: 4,   # BINUNICODE       — protocol 2, the torch.save default
        0x8C: 1,   # SHORT_BINUNICODE — protocol 4
        0x8D: 8,   # BINUNICODE8      — protocol 4, long strings
    }
    keys, i, n = set(), 0, len(blob)
    while i < n:
        width = OPCODES.get(blob[i])
        if width is None:
            i += 1
            continue
        if i + 1 + width > n:
            break
        ln = int.from_bytes(blob[i + 1:i + 1 + width], "little")
        start = i + 1 + width
        if ln <= 0 or ln > 512 or start + ln > n:
            i += 1
            continue
        try:
            text = blob[start:start + ln].decode("utf-8")
        except UnicodeDecodeError:
            i += 1
            continue
        if "." in text and text.endswith((".weight", ".bias")):
            keys.add(text)
        i = start + ln
    return keys or None       # None = could not parse, which is not the same
                              # as "parsed and found nothing"


def main(bundle: Path) -> int:
    print(f"Bundle: {bundle}\n")
    if not bundle.is_dir():
        print(f"FAIL  directory does not exist.\n"
              f"      Unpack dp_cxr_deployment_bundle.zip there first.")
        return 1

    for f in REQUIRED:
        (ok if (bundle / f).exists() else fail).append(f"{f} present")
    for f in OPTIONAL:
        if (bundle / f).exists():
            ok.append(f"{f} present")
        else:
            warn.append(f"{f} missing — "
                        + ("thresholds default to 0.5, which will report almost "
                           "no positives at this prevalence"
                           if f == "thresholds.json" else
                           "the training-time text rule will fall back to the "
                           "built-in copy" if f == "text_rule.json" else
                           "the download cannot be checksum-verified, only "
                           "presence-checked" if f == "MANIFEST.json" else
                           "provenance will be unavailable at /model-card"))

    cfg = None
    if (bundle / "bundle_config.json").exists():
        try:
            cfg = json.loads((bundle / "bundle_config.json").read_text())
            ok.append(f"config parses: {cfg.get('run_name')} "
                      f"({cfg.get('fusion_type')} fusion, "
                      f"{cfg.get('num_classes')} labels)")
        except Exception as e:
            fail.append(f"bundle_config.json is not valid JSON: {e}")

    if cfg and (bundle / "head_state_dict.pth").exists():
        keys = read_pth_keys(bundle / "head_state_dict.pth")
        ft = cfg.get("fusion_type")
        if keys is None:
            warn.append("could not read the tensor names inside head_state_dict.pth "
                        "— skipping the architecture cross-check. The service "
                        "itself validates this properly when it loads the model.")
        elif ft in EXPECTED_KEYS:
            missing = EXPECTED_KEYS[ft] - keys
            if missing:
                fail.append(f"head weights do not match fusion_type={ft!r}; "
                            f"missing {sorted(missing)}. The wrong model was "
                            f"exported — re-run notebook section 8.2.")
            else:
                ok.append(f"head weights match fusion_type={ft!r} "
                          f"({len(keys)} tensors)")

    # ── integrity: the notebook records a sha256 per file, so a truncated or
    #    corrupted Drive download can be caught here rather than at inference ──
    if (bundle / "MANIFEST.json").exists():
        try:
            man = json.loads((bundle / "MANIFEST.json").read_text())
            bad, absent = [], []
            for name, meta in man.items():
                f = bundle / name
                if not f.exists():
                    absent.append(name)
                    continue
                raw = f.read_bytes()
                if len(raw) != meta.get("bytes") or \
                        hashlib.sha256(raw).hexdigest() != meta.get("sha256"):
                    bad.append(name)
            if bad:
                fail.append(f"checksum mismatch on {', '.join(bad)} — the "
                            f"download is corrupted or was edited. Re-download "
                            f"dp_cxr_deployment_bundle.zip.")
            elif absent:
                warn.append(f"listed in MANIFEST.json but not unpacked: "
                            f"{', '.join(absent)}")
            else:
                ok.append(f"integrity verified: {len(man)} files match their "
                          f"recorded sha256")
        except Exception as e:
            warn.append(f"MANIFEST.json unreadable: {e}")

    if (bundle / "thresholds.json").exists():
        try:
            thr = json.loads((bundle / "thresholds.json").read_text())
            if all(abs(v - 0.5) < 1e-9 for v in thr.values()):
                warn.append("every threshold is 0.5 — these were not tuned on "
                            "validation and will under-report positives")
            else:
                ok.append("thresholds tuned: "
                          + ", ".join(f"{k} {v:.2f}" for k, v in thr.items()))
        except Exception as e:
            fail.append(f"thresholds.json is not valid JSON: {e}")

    if (bundle / "model_card.json").exists():
        try:
            card = json.loads((bundle / "model_card.json").read_text())
            priv = card.get("privacy", {})
            eps = priv.get("achieved_epsilon")
            if eps is None:
                warn.append("model card records no achieved epsilon")
            else:
                ok.append(f"privacy: {priv.get('mechanism')} at "
                          f"{priv.get('placement')}, achieved ε = {float(eps):.3f}, "
                          f"δ = {priv.get('delta')}")
        except Exception as e:
            warn.append(f"model_card.json unreadable: {e}")

    for tag, items in (("PASS", ok), ("WARN", warn), ("FAIL", fail)):
        for m in items:
            print(f"{tag}  {m}")

    print()
    if fail:
        print(f"{len(fail)} blocking problem(s). The service will not serve "
              f"correct predictions until these are fixed.")
        return 1
    print("Bundle is usable." + ("  Review the warnings above." if warn else ""))
    return 0


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "bundle"
    sys.exit(main(target))
