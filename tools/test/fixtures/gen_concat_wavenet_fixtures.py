"""Generate ConcatWaveNet parity fixtures for the C++ runtime tests.

Like gen_hyperwavenet_fixtures.py, this script is the *only* place these test models
are produced, and it produces them by driving the training repo (the single source of
truth) directly:

    https://github.com/phillipmself/neural-amp-modeler-parametric
    branch: feature/parametric-main

It does not reimplement any model, weight layout, or control encoding. It builds a real
``ConcatWaveNet`` via ``init_from_config``, perturbs its weights so the model is not
near-trivial, then uses the repo's own ``export_parametric`` to emit:

  * ``concat_wavenet.nam``         - the parametric model the C++ runtime loads.
  * ``concat_wavenet_golden.json`` - the input buffer plus, per setting, the raw param
                                     vector and the Python-rendered output
                                     (``model(x, params, pad_start=True)``), which is
                                     sample-aligned to a prewarmed C++ ``process``.

Unlike gen_hyperwavenet_fixtures.py, this generator is REPRODUCIBLE: the inner WaveNet's
Linear/Conv init draws from the global torch RNG, so the seed is pinned below and a
regenerated fixture is byte-identical to the committed one.

Usage (any of the repo's conda envs that import ``nam`` + torch works):

    /path/to/env/bin/python tools/test/fixtures/gen_concat_wavenet_fixtures.py \
        --repo ../repo_checkout_folder --outdir tools/test/fixtures

The generated files are committed; regenerate only when the contract changes.
"""

import argparse
import json
import sys
from pathlib import Path

# Pinned so the global-RNG draws in the inner WaveNet's parameter init are reproducible.
_SEED = 0


def _build_model(nam):
    ConcatWaveNet = nam["ConcatWaveNet"]
    InnerWaveNet = nam["InnerWaveNet"]
    torch = nam["torch"]

    torch.manual_seed(_SEED)

    # Two layer arrays, so the second array's input_size (the first array's channel width)
    # differs from its condition_size (1 audio + 4 encoded param channels). That is the
    # shape that distinguishes ConcatWaveNet from a stock WaveNet: every layer array is
    # conditioned on the concatenated tensor, but only the first one is fed by it.
    inner_config = {
        "layers_configs": [
            {
                "input_size": 5,
                "condition_size": 5,
                "head": {"out_channels": 2, "kernel_size": 1, "bias": True},
                "channels": 3,
                "kernel_size": 2,
                "dilations": [1, 2],
                "activation": "Tanh",
            },
            {
                "input_size": 3,
                "condition_size": 5,
                "head": {"out_channels": 1, "kernel_size": 1, "bias": True},
                "channels": 2,
                "kernel_size": 2,
                "dilations": [1],
                "activation": "Tanh",
            },
        ],
        # Deliberately not 1.0, so a head_scale the runtime mishandled would show up.
        "head_scale": 0.02,
    }
    config = InnerWaveNet.init_from_config(inner_config).export_config()
    config["sample_rate"] = 48_000.0
    config["params"] = [
        {"name": "gain", "min": 0.0, "max": 10.0, "default": 5.0, "type": "continuous"},
        {
            "name": "mode",
            "min": 0,
            "max": 2,
            "default": 1,
            "type": "switch",
            "enum_names": ["clean", "crunch", "lead"],
        },
    ]

    model = ConcatWaveNet.init_from_config(config)

    # Freshly-initialized convs are small and nearly linear; perturb so the conditioned
    # output actually varies with the control setting.
    generator = torch.Generator().manual_seed(_SEED)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(
                0.05
                * torch.randn(
                    parameter.shape,
                    generator=generator,
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            )
    model.eval()
    return model


def _make_input(torch, num_samples=160):
    # Deterministic, non-silent, non-degenerate input (sine + a little reproducible noise).
    t = torch.linspace(0.0, 1.0, num_samples)
    generator = torch.Generator().manual_seed(1)
    noise = 0.1 * torch.randn(num_samples, generator=generator)
    return (0.5 * torch.sin(2.0 * torch.pi * 6.0 * t) + noise).to(torch.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="path to neural-amp-modeler-parametric")
    parser.add_argument("--outdir", required=True, help="fixture output directory")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))

    import torch
    from nam.models.parametric import ConcatWaveNet, export_parametric
    from nam.models.wavenet._wavenet import WaveNet as InnerWaveNet

    nam = {
        "torch": torch,
        "ConcatWaveNet": ConcatWaveNet,
        "InnerWaveNet": InnerWaveNet,
    }

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    model = _build_model(nam)
    x = _make_input(torch)

    # name -> raw param vector [gain, mode_index] in the spec's positional order. Covers
    # both switch extremes and the low/high ends of the continuous param's range.
    settings = {
        "nominal": [5.0, 1.0],
        "lead": [9.0, 2.0],
        "clean": [1.0, 0.0],
    }

    export_parametric(model, outdir, basename="concat_wavenet")

    golden = {
        "sample_rate": 48_000.0,
        "input": x.tolist(),
        "settings": [],
    }
    with torch.no_grad():
        for name, params in settings.items():
            y = model(x, torch.tensor(params, dtype=torch.float32), pad_start=True)
            golden["settings"].append(
                {"name": name, "params": params, "output": y.tolist()}
            )

    golden_path = outdir / "concat_wavenet_golden.json"
    with open(golden_path, "w") as fp:
        json.dump(golden, fp)

    print(f"wrote {golden_path}")
    print(f"input samples: {len(golden['input'])}, settings: {len(golden['settings'])}")
    print(f"receptive_field: {model.receptive_field}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
