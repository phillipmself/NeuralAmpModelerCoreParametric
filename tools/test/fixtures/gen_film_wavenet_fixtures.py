"""Generate FiLMWaveNet parity fixtures for the C++ runtime tests.

Like gen_concat_wavenet_fixtures.py, this script is the *only* place these test models are
produced, and it produces them by driving the training repo (the single source of truth)
directly:

    https://github.com/phillipmself/neural-amp-modeler-parametric
    branch: feature/parametric-film-wavenet

It does not reimplement any model, weight layout, or control encoding. It builds a real
``FiLMWaveNet`` via ``init_from_config``, perturbs its weights so the model is not
near-trivial, then uses the repo's own ``export_parametric`` to emit, for each variant:

  * ``<name>.nam``         - the parametric model the C++ runtime loads.
  * ``<name>_golden.json`` - the input buffer plus, per setting, the raw param vector and
                             the Python-rendered output (``model(x, params,
                             pad_start=True)``), which is sample-aligned to a prewarmed
                             C++ ``process``.

Two variants are emitted, because they exercise different runtime paths:

  * ``film_wavenet``         - no param encoder: the FiLM condition is the encoded control
                               vector itself.
  * ``film_wavenet_encoder`` - a param encoder MLP sits between the encoded controls and
                               the FiLM condition.

This generator is REPRODUCIBLE: the inner WaveNet's Conv/Linear init draws from the global
torch RNG, so the seed is pinned below and a regenerated fixture is byte-identical to the
committed one.

Usage (any of the repo's conda envs that import ``nam`` + torch works):

    /path/to/env/bin/python tools/test/fixtures/gen_film_wavenet_fixtures.py \
        --repo ../repo_checkout_folder --outdir tools/test/fixtures

The generated files are committed; regenerate only when the contract changes.
"""

import argparse
import json
import sys
from pathlib import Path

# Pinned so the global-RNG draws in the inner WaveNet's parameter init are reproducible.
_SEED = 0

_PARAMS = [
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

# name -> raw param vector [gain, mode_index], in the spec's positional order. Covers both
# switch extremes and the low/high ends of the continuous param's range.
_SETTINGS = {
    "nominal": [5.0, 1.0],
    "lead": [9.0, 2.0],
    "clean": [1.0, 0.0],
}


def _model_config(with_encoder: bool) -> dict:
    # Every FiLM site is active somewhere, and both shift modes appear, so a runtime that
    # mis-orders the per-layer FiLM weight blocks or ignores `shift` cannot pass.
    #
    # Two layer arrays, so the second array's input_size (the first array's channel width)
    # differs from its condition_size. Under FiLMWaveNet condition_size is always 1 -- the
    # audio -- which is what distinguishes it from ConcatWaveNet: the controls reach the
    # layers only through FiLM.
    shift_on = {"active": True, "shift": True, "groups": 1}
    shift_off = {"active": True, "shift": False, "groups": 1}
    first = {
        # Must match the second array's per-layer head width (its head1x1 out_channels),
        # since head contributions are summed across arrays.
        "head": {"out_channels": 2, "kernel_size": 1, "bias": True},
        "channels": 3,
        "kernel_size": 2,
        "dilations": [1, 2],
        "activation": "Tanh",
        "layer1x1": {"active": True, "groups": 1},
        "conv_pre_film": dict(shift_on),
        "conv_post_film": dict(shift_off),
        "input_mixin_pre_film": dict(shift_on),
        "activation_post_film": dict(shift_on),
        "layer1x1_post_film": dict(shift_off),
    }
    second = {
        "input_size": 3,  # = first layer array's channels
        "head": {"out_channels": 1, "kernel_size": 1, "bias": True},
        "channels": 2,
        "kernel_size": 2,
        "dilations": [1],
        "activation": "Tanh",
        "head_1x1_config": {"active": True, "out_channels": 2, "groups": 1},
        "input_mixin_post_film": dict(shift_on),
        "activation_pre_film": dict(shift_off),
        "head1x1_post_film": dict(shift_on),
    }
    config = {
        "sample_rate": 48_000.0,
        "layers": [first, second],
        # Deliberately not 1.0, so a head_scale the runtime mishandled would show up.
        "head_scale": 0.02,
        "params": list(_PARAMS),
        # Identity init would make the model control-invariant; the perturbation below is
        # what makes the fixture discriminating, so leave it on and perturb after.
        "identity_init": True,
    }
    if with_encoder:
        config["param_encoder"] = {
            "hidden_sizes": [5],
            "out_features": 4,
            "activation": "ReLU",
        }
    return config


def _build_model(nam, with_encoder: bool):
    FiLMWaveNet = nam["FiLMWaveNet"]
    torch = nam["torch"]

    torch.manual_seed(_SEED)
    model = FiLMWaveNet.init_from_config(_model_config(with_encoder))

    # Freshly-initialized convs are small and nearly linear, and identity-init leaves every
    # FiLM control-invariant; perturb so the conditioned output actually varies with the
    # control setting.
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
    parser.add_argument(
        "--repo", required=True, help="path to neural-amp-modeler-parametric"
    )
    parser.add_argument("--outdir", required=True, help="fixture output directory")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))

    import torch
    from nam.models.parametric import FiLMWaveNet, export_parametric

    nam = {"torch": torch, "FiLMWaveNet": FiLMWaveNet}

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    for basename, with_encoder in (
        ("film_wavenet", False),
        ("film_wavenet_encoder", True),
    ):
        model = _build_model(nam, with_encoder)
        x = _make_input(torch)

        export_parametric(model, outdir, basename=basename)

        golden = {"sample_rate": 48_000.0, "input": x.tolist(), "settings": []}
        with torch.no_grad():
            for name, params in _SETTINGS.items():
                y = model(x, torch.tensor(params, dtype=torch.float32), pad_start=True)
                golden["settings"].append(
                    {"name": name, "params": params, "output": y.tolist()}
                )

        golden_path = outdir / f"{basename}_golden.json"
        with open(golden_path, "w") as fp:
            json.dump(golden, fp)

        print(f"wrote {outdir / (basename + '.nam')} and {golden_path}")
        print(f"  receptive_field: {model.receptive_field}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
