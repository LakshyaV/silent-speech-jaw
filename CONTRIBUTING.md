# Contributing

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
make check
```

`make check` runs what CI runs: `ruff check`, `ruff format --check` and `pytest`.
The suite needs no hardware and no downloads and finishes in well under a
minute. Two tests exercise a real GGUF model and are skipped unless
`SSJ_TEST_GGUF` points to one:

```bash
SSJ_TEST_GGUF=assets/llm/qwen2.5-0.5b-instruct-q4_k_m.gguf pytest tests/test_llm.py
```

## Workflow

1. Open an issue describing the change, or pick an existing one.
2. Branch from `main`: `feat/<topic>`, `fix/<topic>`, `docs/<topic>`.
3. Commit with [Conventional Commits](https://www.conventionalcommits.org):
   `feat(dsp): ...`, `fix(decode): ...`, `docs: ...`, `test: ...`, `chore: ...`.
   The scope is the package (`io`, `dsp`, `sim`, `collect`, `data`, `model`,
   `decode`, `llm`, `eval`, `runtime`, `firmware`, `app`).
4. Open a pull request that references the issue (`Closes #n`). CI must pass.

## Ground rules

These exist because breaking them has cost debugging time before.

- **One front-end.** Features for training and for live decoding come from the
  same streaming code (`ssj.dsp.frontend`). It must stay causal and
  chunk-invariant; `tests/test_frontend.py` checks that. Any change to its
  output changes the front-end fingerprint: say so in the pull request, because
  datasets must be rebuilt and models retrained.
- **Evaluate on held-out sessions.** Results from a split inside one session do
  not measure robustness to re-mounting the sensors and should not be reported
  as if they did.
- **The firmware measures; the host decides.** No vocabulary, labels or trial
  logic in `firmware/`. Changes to the wire format need the same change in
  `firmware/ssj_streamer/framing.h` and `src/ssj/io/protocol.py`, plus
  `docs/protocol.md`; `tests/test_firmware.py` compiles the former and decodes
  its output with the latter.
- **The runtime stays light.** `ssj.io`, `ssj.dsp`, `ssj.decode`, `ssj.llm` and
  `ssj.runtime` import only numpy, scipy and pyyaml at module level. PyTorch is
  confined to `ssj.models`, `ssj.train` and `ssj.data.loader`.
- **Configuration is typed.** Options are fields of frozen dataclasses; after
  adding or changing one, run `python tools/write_default_config.py`.
- **Test against ground truth.** The simulator returns the true mount
  rotations, hinge axis and jaw angle. New signal-processing code should be
  checked against those, not only for shape and finiteness.

## Firmware

```bash
make firmware
```

compiles the sketch for the Arduino UNO Q with pinned core and library versions
(`firmware/ssj_streamer/sketch.yaml`). CI additionally compiles the serial
transport for the UNO R4 WiFi.
