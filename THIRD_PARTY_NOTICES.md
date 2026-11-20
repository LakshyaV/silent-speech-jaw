# Third-party notices

## CMU Pronouncing Dictionary

`src/ssj/phonology/data/core.dict` is derived from the CMU Pronouncing
Dictionary (<https://github.com/cmusphinx/cmudict>): stress markers removed and
restricted to frequent English words. `ssj assets fetch lexicon` downloads the
complete dictionary. It is distributed under the following licence.

```text
Copyright (C) 1993-2015 Carnegie Mellon University. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
   The contents of this file are deemed to be source code.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in
   the documentation and/or other materials provided with the
   distribution.

This work was supported in part by funding from the Defense Advanced
Research Projects Agency, the Office of Naval Research and the National
Science Foundation of the United States of America, and by member
companies of the Carnegie Mellon Sphinx Speech Consortium. We acknowledge
the contributions of many volunteers to the expansion and improvement of
this dictionary.

THIS SOFTWARE IS PROVIDED BY CARNEGIE MELLON UNIVERSITY ``AS IS'' AND
ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL CARNEGIE MELLON UNIVERSITY
NOR ITS EMPLOYEES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

The word selection of the core lexicon uses word frequencies counted on the
Tatoeba English sentence collection (<https://tatoeba.org>, CC BY 2.0 FR).

## Resources downloaded on request

`ssj assets fetch` downloads the following; they are not part of this
repository and carry their own licences.

| Resource | Source | Licence |
| --- | --- | --- |
| CMU Pronouncing Dictionary | github.com/cmusphinx/cmudict | BSD-style, above |
| English sentences | tatoeba.org | CC BY 2.0 FR |
| SmolLM2 360M Instruct (GGUF) | huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF | Apache-2.0 |
| Qwen2.5 0.5B / 1.5B Instruct (GGUF) | huggingface.co/Qwen | Apache-2.0 |
| Llama 3.2 3B Instruct (GGUF) | huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF | Llama 3.2 Community License |

## Arduino libraries used by the firmware

Arduino_RouterBridge and Arduino_RPClite (MPL-2.0), MsgPack, ArxContainer,
ArxTypeTraits and DebugLog (MIT). They are fetched by the Arduino toolchain at
build time.
