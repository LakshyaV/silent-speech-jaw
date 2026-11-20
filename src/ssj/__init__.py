"""silent-speech-jaw: open-vocabulary silent speech from a dual jaw/reference IMU.

The pipeline is organised as independent layers, each importable on its own:

    ssj.io         wire protocol, sample sources, session storage
    ssj.dsp        calibration, streaming front-end, activity segmentation
    ssj.phonology  phoneme inventory, pronunciation lexicon, text normalisation
    ssj.sim        articulatory kinematic simulator
    ssj.collect    prompt design and guided recording
    ssj.data       dataset building and augmentation
    ssj.models     phoneme recogniser and its CTC objective
    ssj.train      training and session adaptation
    ssj.decode     n-gram LM, lexicon-constrained beam search, CTC scoring
    ssj.llm        local quantized LLM backends and evidence fusion
    ssj.eval       error rates, confusion analysis, weight tuning
    ssj.runtime    portable bundles and the live decoding loop

Only ``numpy``, ``scipy`` and ``pyyaml`` are needed to import the runtime path;
``torch`` is confined to ``ssj.models``, ``ssj.train`` and ``ssj.data.torch``.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
