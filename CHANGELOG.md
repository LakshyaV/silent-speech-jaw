# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-20

First release: open-vocabulary silent speech from a dual jaw/reference IMU.

### Added

- **Firmware** (`firmware/ssj_streamer`): register-level LSM6DSOX driver with
  sensor-FIFO acquisition at 208 Hz, CRC-checked COBS framing, serial transport
  and Arduino UNO Q RPC-bridge transport.
- **Acquisition** (`ssj.io`): wire protocol, serial / bridge / TCP / replay
  sources, crash-safe session storage in raw sensor units.
- **Signal processing** (`ssj.dsp`): timestamp-driven resampling of two
  free-running sensors, three-phase calibration into an anatomical frame,
  chunk-invariant streaming front-end with 23 features, activity segmentation.
- **Phonology** (`ssj.phonology`): ARPAbet inventory with jaw-observable
  articulatory classes, CMUdict lexicon handling, 8,000-word core lexicon, text
  normalisation.
- **Simulator** (`ssj.sim`): text to jaw kinematics to dual-IMU recordings with
  random mounts, sensor clocks, head motion and ground truth.
- **Data collection** (`ssj.collect`): coverage-driven prompt selection, guided
  recorder with calibration routine and per-trial quality checks.
- **Dataset** (`ssj.data`): session-wise processing, held-out-session splits,
  physically motivated augmentation.
- **Recogniser** (`ssj.models`, `ssj.train`): multi-stream TCN + BiGRU with
  padding-invariant normalisation, multi-resolution CTC objective, session
  adapters, training loop with shuffled-label control, ONNX export with
  numerical verification.
- **Decoding** (`ssj.decode`): Kneser-Ney n-gram training and packed storage,
  lexicon-constrained CTC beam search with alignment merging, class-tied
  emission scores, exact CTC sequence scoring, confusion networks.
- **LLM stage** (`ssj.llm`): llama.cpp backend with prefix-cached scoring,
  lattice-to-grammar compilation, candidate pooling and evidence fusion.
- **Evaluation** (`ssj.eval`): per-stage error rates, confusion analysis,
  data-driven refinement of the phoneme classes, decoder weight tuning.
- **Runtime** (`ssj.runtime`): self-describing ONNX bundles with a front-end
  fingerprint check, the assembled pipeline, the live decoding loop.
- **Arduino UNO Q application** (`app/`): on-board decoding or TCP relay,
  assembled with `ssj app package`.
- **CLI**: `assets`, `lm`, `prompts`, `monitor`, `record`, `sim`, `data`,
  `train`, `export`, `adapt`, `tune`, `eval`, `inventory`, `decode`, `live`, `app`.
- Documentation: architecture, workflow, hardware, wire protocol, deployment.

[0.1.0]: https://github.com/LakshyaV/silent-speech-jaw/releases/tag/v0.1.0
