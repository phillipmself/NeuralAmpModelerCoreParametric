"""Generate HyperWaveNet parity fixtures for the C++ runtime tests.

This script is the *only* place test models are produced, and it produces them by
driving the training repo (the single source of truth) directly:

    https://github.com/phillipmself/neural-amp-modeler-parametric
    branch: feature/parametric-hypernetwork

It does not reimplement any model, weight layout, or hypernetwork math. It builds a
real ``HyperWaveNet`` via ``init_from_config``, perturbs the hypernetwork so the
generated deltas are non-trivial, then uses the repo's own ``export_parametric`` and
``bake`` to emit:

  * ``hyperwavenet.nam``            - the parametric model the C++ runtime loads.
  * ``baked_<setting>.nam``         - a stock WaveNet `.nam` baked at each setting,
                                      the independently-exported parity oracle.
  * ``hyperwavenet_golden.json``    - the input buffer plus, per setting, the raw
                                      param vector and the Python-rendered output
                                      (``model(x, params, pad_start=True)``), which is
                                      sample-aligned to a prewarmed C++ ``process``.

Usage (any of the repo's conda envs that import ``nam`` + torch works):

    /path/to/env/bin/python tools/test/fixtures/gen_hyperwavenet_fixtures.py \
        --repo ../repo_checkout_folder --outdir tools/test/fixtures

The generated files are committed; regenerate only when the contract changes.
"""

import argparse
import json
import sys
from pathlib import Path


def _build_model(nam):
    InnerWaveNet = nam["InnerWaveNet"]
    HyperWaveNet = nam["HyperWaveNet"]
    torch = nam["torch"]

    # Small inner WaveNet -> tiny receptive field -> compact golden buffers, while still
    # exercising a rechannel, a dilated stack, an input mixer, a 1x1, and a head1x1.
    inner_config = {
        "layers_configs": [
            {
                "input_size": 1,
                "condition_size": 1,
                "head": {"out_channels": 1, "kernel_size": 1, "bias": True},
                "channels": 3,
                "kernel_size": 2,
                "dilations": [1, 2],
                "activation": "Tanh",
            }
        ],
        "head_scale": 1.0,
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
    # A real hidden trunk + low-rank deltas: covers the C++ trunk activation path and the
    # low-rank decode (U+aU)(V+aV) path in one fixture.
    config["hypernet"] = {
        "hidden_sizes": [4],
        "activation": "ReLU",
        "mode": "low_rank",
        "rank": 2,
    }

    model = HyperWaveNet.init_from_config(config)

    # Zero-init hypernets emit zero deltas (== bare template); perturb params and the
    # low-rank anchor so each setting conditions to a distinct, non-trivial net.
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in model._hypernet.parameters():
            parameter.add_(
                0.05
                * torch.randn(
                    parameter.shape,
                    generator=generator,
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            )
        anchor = getattr(model._hypernet, "_low_rank_anchor", None)
        if anchor is not None:
            anchor.add_(
                0.05 * torch.randn(anchor.shape, generator=generator, dtype=anchor.dtype)
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
    from nam.models.parametric import HyperWaveNet, bake, export_parametric
    from nam.models.wavenet._wavenet import WaveNet as InnerWaveNet

    nam = {
        "torch": torch,
        "HyperWaveNet": HyperWaveNet,
        "InnerWaveNet": InnerWaveNet,
    }

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    model = _build_model(nam)
    x = _make_input(torch)

    # name -> raw param vector [gain, mode_index] in the spec's positional order.
    settings = {
        "nominal": [5.0, 1.0],
        "lead": [9.0, 2.0],
        "clean": [1.0, 0.0],
    }

    export_parametric(model, outdir, basename="hyperwavenet")

    golden = {
        "sample_rate": 48_000.0,
        "input": x.tolist(),
        "settings": [],
    }
    with torch.no_grad():
        for name, params in settings.items():
            params_tensor = torch.tensor(params, dtype=torch.float32)
            y = model(x, params_tensor, pad_start=True)
            bake(model, params_tensor).export(outdir, basename=f"baked_{name}")
            golden["settings"].append(
                {"name": name, "params": params, "output": y.tolist()}
            )

    golden_path = outdir / "hyperwavenet_golden.json"
    with open(golden_path, "w") as fp:
        json.dump(golden, fp)

    print(f"wrote {golden_path}")
    print(f"input samples: {len(golden['input'])}, settings: {len(golden['settings'])}")
    print(f"receptive_field: {model.receptive_field}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
