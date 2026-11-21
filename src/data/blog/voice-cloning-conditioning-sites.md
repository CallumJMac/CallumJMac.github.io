---
pubDatetime: 2026-11-21T12:00:00Z
title: "Where Does a Cloned Voice Live? Prompts and Speaker Embeddings in Zero-Shot TTS"
slug: voice-cloning-conditioning-sites
featured: true
description: "Zero-shot TTS systems condition on a speaker twice: an in-context acoustic prompt and a speaker embedding. I switched each route off across six systems, 31,989 utterances, to measure what each one carries."
tags:
  - TTS
  - Voice Cloning
  - Research
  - Qwen
---

Zero-shot voice cloning looks like one operation: give the model a few seconds of someone speaking, get new speech in their voice. Inside most systems it is two.

The **acoustic prompt** places the reference recording in context, so the model continues it the way a language model continues a document. The **speaker embedding** compresses the same recording into one vector from a speaker-verification encoder. Systems also differ in where that vector enters. Some feed it to the autoregressive language model (LM) that generates speech tokens; others feed it only to the acoustic decoder that turns tokens into a waveform.

These choices get made, and changed between model versions, without anyone measuring what each route carries. CosyVoice 2 removed the embedding from its LM "to avoid information leaking", for example, but what that embedding had contributed was never quantified.

So I treated the routes as experimental variables and switched them off. This post summarises the paper, *Speaker Identity Is Split Across Conditioning Sites* (preprint forthcoming). The code, job lists and every per-utterance score are [on GitHub](https://github.com/CallumJMac/conditioning-sites), and the [audio samples](/conditioning-sites/) cover every system.

The short version:

- **Where the embedding enters decides whether identity survives without the prompt.** Systems whose LM receives the embedding lose 0.06 to 0.11 speaker similarity, or nothing. Decoder-only systems lose 0.18 to 0.24.
- **Within one model, identity is split across the two sites, and the losses add up.** The prompt can stand in for the embedding at the LM, but not at the decoder.
- **Prompts copy their recording, not just the speaker.** Removing them raises predicted naturalness on noisy prompts by up to 0.51, but the same metric also rises when *all* speaker information is removed, so that gain is not better cloning.
- **Three references beat one**, by 0.02 to 0.04 similarity, in either route.

## The setup

I compared two conditions in six open-weight systems:

- **Full:** the system's usual prompt and embedding, both from one reference recording.
- **x-vector only:** every piece of prompt-level conditioning removed, embedding kept.

The systems cover all three designs: the embedding entering the LM (Qwen3-TTS 0.6B and 1.7B, Chatterbox), both the LM and the decoder (CosyVoice-300M), or the decoder only (CosyVoice 3, XTTS-v2). F5-TTS, which has no embedding route at all, serves as a prompt-only control.

There were two test sets. LibriTTS-R gives 34 speakers, each with one fixed 6-second audiobook reference. Seed-TTS-eval gives 1,088 items whose Common Voice prompts are noisier and more varied. In total I synthesised **31,989 utterances**.

The main metric is **SIM-o**: cosine similarity between speaker-verification embeddings of the output and a *held-out* recording of the same speaker, never used as a prompt. Scoring against a held-out recording matters. Scoring against the prompt itself would reward copying the prompt's microphone and room along with the voice.

Targets and seeds are repeated measures, so speakers are the unit of analysis. Every effect below is a speaker-level mean with a bootstrap 95% confidence interval, tested with Wilcoxon signed-rank tests and Holm correction.

## Across systems: an LM site keeps the voice

<figure>
  <img
    src="/assets/conditioning-sites/pathway-contrasts.svg"
    alt="Dot plot of the change in speaker similarity and predicted naturalness when the prompt is removed, for six systems on two test sets. Systems whose language model receives the embedding lose little similarity; decoder-only systems lose much more."
    width="683"
    height="331"
    loading="lazy"
    decoding="async"
  />
  <figcaption>
    <strong>Figure 1.</strong> Removing prompt-level conditioning (x-vector only minus full). (a) Change in speaker similarity to a held-out recording. (b) Change in predicted naturalness. Speaker-level means with bootstrap 95% CIs; filled circles are clean LibriTTS-R prompts, open squares noisy Seed-TTS-eval prompts. Chatterbox's naturalness falls by 1.8, off the scale, because it stops producing intelligible speech.
  </figcaption>
</figure>

The systems split cleanly by design (Figure 1a).

When the LM receives the embedding, identity largely survives losing the prompt. Qwen3-TTS loses 0.062 to 0.073 similarity on LibriTTS-R and 0.099 to 0.108 on Seed-TTS-eval, at both sizes. CosyVoice-300M loses nothing on LibriTTS-R (+0.017, not significant) and 0.068 on Seed-TTS-eval.

When the embedding reaches only the decoder, losses are two to three times larger: 0.184 and 0.244 for CosyVoice 3, and 0.224 and 0.213 for XTTS-v2 (all p < 0.001 after correction).

The cleanest comparison is within one family. CosyVoice 3 shares its predecessor's lineage and front end but dropped the embedding from the LM. Without a prompt, the successor loses 0.18 where CosyVoice-300M loses nothing.

Chatterbox is the exception. Its LM does receive an embedding, but without its speech prompt it produces unintelligible speech (96 to 98% word error rate), so the embedding alone is outside its operating regime. Every other system keeps its word error rate when the prompt goes.

## Within one model: identity is split across sites

The systems above differ in data, scale, tokenisers and decoders, so they cannot isolate the site itself. CosyVoice-300M can: it passes the *same* speaker embedding to both its LM and its flow-matching decoder. That allows a 2 × 2 × 2 factorial, switching the prompt, the LM embedding and the decoder embedding on and off independently, with text, reference, seed and decoding identical across all eight cells (6,480 utterances).

<figure>
  <img
    src="/assets/conditioning-sites/site-factorial.svg"
    alt="Speaker similarity for each combination of prompt and embedding sites in CosyVoice-300M on two test sets. With the prompt removed, similarity falls step by step as sites are removed, to 0.14. With the prompt kept, removing the language-model embedding changes nothing."
    width="599"
    height="294"
    loading="lazy"
    decoding="async"
  />
  <figcaption>
    <strong>Figure 2.</strong> CosyVoice-300M site factorial: speaker similarity for every combination of prompt and embedding sites. Speaker-level means with bootstrap 95% CIs; 34 LibriTTS-R speakers and 268 Seed-TTS-eval prompt speakers.
  </figcaption>
</figure>

**Without the prompt, both sites carry identity, and the losses add.** On LibriTTS-R, removing the LM embedding costs 0.113 similarity and removing the decoder embedding costs 0.201. Removing both costs 0.310, close to their sum, which leaves similarity at the 0.14 floor of an unconditioned voice. Seed-TTS-eval shows the same pattern: 0.092, 0.181 and 0.272.

**With the prompt, the two sites behave differently.** Removing the LM embedding no longer matters: −0.010 and −0.008, both within noise. Removing the decoder embedding still costs 0.070 and 0.057.

So at the LM, the prompt and the embedding are *substitutes*. Both are context for generating speech tokens, and either is enough. At the decoder they are *complements*: the prompt's acoustic features do not replace the embedding.

This also explains the cross-system result. Switching off CosyVoice-300M's LM site moves its prompt-free change from +0.017 to −0.096, towards its decoder-only successor. Systems that condition both stages keep more identity without the prompt because losing the prompt costs them only one site's worth of speaker information.

## Prompts carry the recording, and the naturalness metric rewards losing the voice

Removing the prompt barely changes predicted naturalness on clean LibriTTS-R. On noisy Seed-TTS-eval prompts it raises it in every working system: +0.03 and +0.08 for the two Qwen3-TTS sizes, +0.13 for CosyVoice 3, +0.31 for CosyVoice-300M and +0.51 for XTTS-v2 (Figure 1b).

The prompt is importing its recording conditions. With the prompt, every system's output is closer to the specific prompt recording than to another recording of the same speaker. Removing the prompt shrinks that gap in all twelve system and dataset pairs. For XTTS-v2 on Seed-TTS-eval, the gap disappears entirely.

But the naturalness gain needs a caveat. In the factorial, removing *all* speaker information (prompt and both embeddings) raises predicted naturalness on Seed-TTS-eval from 4.04 to 4.35. UTMOS rewards a generic voice. A naturalness improvement that arrives together with an identity loss should not be read as better cloning, and that applies to any prompt-free conditioning result reported with an automatic naturalness predictor.

## More references help identity, not naturalness

With three references instead of one, Qwen3-TTS and CosyVoice 3 gain 0.022 to 0.044 similarity in both routes, whether the references are concatenated into the prompt or averaged as embeddings (p < 0.005 for every system and route). Naturalness moves by at most 0.02.

| System         | Route                    | Δ similarity | Δ naturalness |
| -------------- | ------------------------ | -----------: | ------------: |
| Qwen3-TTS 0.6B | prompt (concatenated)    |       +0.043 |        −0.008 |
|                | embedding (averaged)     |       +0.036 |        −0.003 |
| Qwen3-TTS 1.7B | prompt (concatenated)    |       +0.022 |        −0.018 |
|                | embedding (averaged)     |       +0.028 |        −0.001 |
| CosyVoice 3    | prompt (concatenated)    |       +0.044 |        +0.015 |
|                | embedding (averaged)     |       +0.031 |        +0.013 |
| F5-TTS         | prompt (concatenated)    |       −0.126 |        −0.722 |

The prompt-only F5-TTS gets *worse* with a three-reference prompt, consistent with prompt length being something a model has to be trained for.

## Listen

One LibriTTS-R speaker, drawn at random with a fixed seed rather than picked for effect. Every model was asked to say the same new sentence, "I thanked him, but I was not deceived by his compliment."

The models heard a **reference prompt** of the speaker reading a *different* sentence. Similarity is scored against a **held-out recording** that no model heard, so it rewards matching the voice, not copying the prompt.

**The speaker**

<div class="listen-row"><p>Reference prompt, heard by all</p><audio controls preload="none" src="/conditioning-sites/samples/3729_6852_000078_000003/prompt.mp3"></audio></div>
<div class="listen-row"><p>Held-out recording, heard by none</p><audio controls preload="none" src="/conditioning-sites/samples/3729_6852_000078_000003/heldout.mp3"></audio></div>

**Qwen3-TTS 0.6B**, embedding reaches the LM

<div class="listen-row"><p>Prompt + embedding: similarity 0.53</p><audio controls preload="none" src="/conditioning-sites/samples/3729_6852_000078_000003/qwen06_full.mp3"></audio></div>
<div class="listen-row"><p>Embedding only: similarity 0.60</p><audio controls preload="none" src="/conditioning-sites/samples/3729_6852_000078_000003/qwen06_xvector_only.mp3"></audio></div>

**CosyVoice 3**, embedding in the decoder only

<div class="listen-row"><p>Prompt + embedding: similarity 0.58</p><audio controls preload="none" src="/conditioning-sites/samples/3729_6852_000078_000003/cosyvoice3_full.mp3"></audio></div>
<div class="listen-row"><p>Embedding only: similarity 0.51</p><audio controls preload="none" src="/conditioning-sites/samples/3729_6852_000078_000003/cosyvoice3_xvector_only.mp3"></audio></div>

On this sentence, Qwen3-TTS actually scores *higher* without the prompt, while CosyVoice 3 loses 0.07. Single sentences vary: across all 34 speakers the averages are −0.07 and −0.18 (Figure 1). Similarity here is the per-clip SIM-o from the released scores.

The [demo page](/conditioning-sites/) has a second LibriTTS-R speaker, two noisy Seed-TTS-eval prompts, every system and all eight factorial cells.

## What this means if you build with these models

- **Removing the embedding from the LM is cheap only while a prompt is always present.** With a prompt, the LM embedding made no measurable difference. Without one, it was a third of the identity. Prompt-free, streaming and cross-lingual use, where the prompt is absent, short or mismatched, all depend on it.
- **For prompt-free cloning, condition both stages.**
- **Treat speaker embeddings as identity-bearing data.** In systems that condition the LM on it, an embedding alone reproduces much of a voice. A leaked embedding can act as a voice credential, so it deserves the same care as a recording.
- **Don't trust automatic naturalness gains that coincide with identity loss.** Check similarity against a held-out recording, not the prompt.

This connects to the [Qwen3-TTS serving series](/posts/qwen-tts-streaming/). Qwen3-TTS kept most of a voice in its x-vector-only mode, which also removes the reference recording from the model's context. I have not measured what that does to time to first audio. It is a natural next experiment for a voice service that clones from stored speaker profiles.

## Limitations

The causal factorial covers one model. The cross-system pattern rests on four working systems and system-specific interventions. Zeroing an embedding leaves the downstream projection bias, a standard but not unique ablation.

Every effect here is measured with automatic metrics. A paired listening study is in preparation. Evaluation is English only, Seed-TTS-eval uses a single seed, and the pathway study uses one reference length.

## Reproduce it

The per-utterance scores are in the [repository](https://github.com/CallumJMac/conditioning-sites), so every table and figure in the paper rebuilds in about two minutes without a GPU. `make check` recomputes the analysis and fails on any change, and runs on every push. The figures in this post come from `scripts/blog_figures.py`, which reads only the released analysis tables.
