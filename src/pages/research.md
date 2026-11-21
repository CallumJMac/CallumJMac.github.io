---
layout: ../layouts/AboutLayout.astro
title: "Research"
---

I research how speech generation models work inside, and measure how they behave in production. Every project releases its code, experiment design and per-sample scores, so the results can be rebuilt without a GPU.

## Speaker identity in zero-shot voice cloning

**Speaker Identity Is Split Across Conditioning Sites: Prompt and Embedding Pathways in Zero-Shot Voice Cloning.** Callum Macpherson, 2026. Preprint forthcoming.

Zero-shot TTS systems condition on a speaker twice: through an in-context acoustic prompt and through a speaker embedding. I switched each route off across six open-weight systems (31,989 utterances). Identity survives without the prompt only when the embedding reaches the language model. Within one model, identity splits additively across the language model and the decoder, and the prompt can replace the embedding at the language model but not at the decoder.

[Write-up](/posts/voice-cloning-conditioning-sites/) · [Code, data and paper source](https://github.com/CallumJMac/conditioning-sites) · [Audio samples](/conditioning-sites/)

## Serving streaming TTS

A two-part measurement study of Qwen3-TTS on an NVIDIA A10G. It covers where first-audio latency goes, why one backend queues and another stutters, and what capacity costs per character. A queueing model reproduces the measured latency within 1%.

[How to build a low-latency streaming Qwen3-TTS server](/posts/qwen-tts-streaming/) · [Capacity planning Qwen3-TTS on an A10G](/posts/qwen-tts-at-scale/) · [Code and data](https://github.com/CallumJMac/qwen-tts-serve)

## Earlier work

**Multi-reference voice cloning for Qwen3-TTS.** How to combine several reference recordings of one speaker: 1,275 controlled synthesis runs, finding a trade-off between concatenating references (speaker identity) and averaging embeddings (naturalness and stability). Superseded by the conditioning-sites study above.

[Code and results](https://github.com/CallumJMac/tts) · [Audio samples](/tts/)
