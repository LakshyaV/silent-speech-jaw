# silent-speech-jaw

Open-vocabulary silent speech from two small IMUs: one on the jaw, one behind
the ear as a head-motion reference. Jaw movement is recognised as a sequence of
phonemes, and a pronunciation lexicon, an n-gram model and a quantized local LLM
turn the phonemes into sentences.

The project builds on [6ix](https://github.com/LakshyaV/6ix), which recognised a
fixed set of jaw-gesture commands with the same sensor pair.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
make check
```

`ssj --help` lists the available commands. See [CONTRIBUTING.md](CONTRIBUTING.md)
for the workflow and conventions.

## License

MIT, see [LICENSE](LICENSE).
