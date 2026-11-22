---
pubDatetime: 2026-11-21T12:00:00Z
modDatetime: 2026-11-22T12:00:00Z
title: "Where Does a Cloned Voice Live? Prompts and Speaker Embeddings in Zero-Shot TTS"
slug: voice-cloning-conditioning-sites
featured: true
description: "Zero-shot TTS systems condition on a speaker twice: an in-context acoustic prompt and a speaker embedding. I switched these routes off in six systems (31,989 synthesised utterances) to measure what each one carries."
tags:
  - TTS
  - Voice Cloning
  - Research
  - Qwen
---

Zero-shot voice cloning looks like one operation: give the model a few seconds of someone speaking, get new speech in their voice. Inside, most systems use two.

The **acoustic prompt** places the reference recording in context for the model to continue, as a language model continues a document. The **speaker embedding** compresses the same recording into one vector from a speaker-verification encoder. Some systems feed that vector to the autoregressive language model (LM) that generates speech tokens; others feed it only to the acoustic decoder that turns tokens into a waveform.

These choices change between model versions without anyone measuring what each route carries. CosyVoice 2 removed the embedding from its LM "to avoid information leaking", but never quantified what it had contributed.

So I switched the routes off. This post summarises the paper, *Speaker Identity Is Split Across Conditioning Sites* (preprint forthcoming). Code, job lists and all per-utterance scores are [on GitHub](https://github.com/CallumJMac/conditioning-sites); the [audio samples](/conditioning-sites/) cover every system.

The short version:

- **Where the embedding enters predicts how much identity survives without the prompt.** Where the LM receives the embedding, systems lose 0.06 to 0.11 speaker similarity, or nothing; Chatterbox, the exception, breaks without its prompt. Decoder-only systems lose 0.18 to 0.24.
- **Within one model, identity is split across the two sites, and the losses roughly add.** The prompt can stand in for the embedding at the LM, but not at the decoder.
- **Prompts copy their recording, not just the speaker.** Removing them raises predicted naturalness on noisy prompts by up to 0.51, but the metric also rises when *all* speaker information is removed, so this is not better cloning.
- **Three references beat one**, by 0.02 to 0.04 similarity in either route for Qwen3-TTS and CosyVoice 3. The prompt-only F5-TTS gets worse.

## The setup

I compared two conditions in six open-weight systems:

- **Full:** the system's usual prompt and embedding, both from one reference recording.
- **x-vector only:** all prompt-level conditioning removed, embedding (the x-vector) kept.

The embedding enters the LM (Qwen3-TTS 0.6B and 1.7B, Chatterbox), both LM and decoder (CosyVoice-300M), or the decoder only (CosyVoice 3, XTTS-v2). F5-TTS, with no embedding route, is a prompt-only control in the reference-count comparison.

LibriTTS-R gives 34 speakers, each with one fixed 6-second audiobook reference; Seed-TTS-eval gives 1,088 items with noisier, more varied Common Voice prompts. I synthesised **31,989 utterances**: 25,509 for the cross-system and reference-count comparisons, 6,480 for the within-model factorial.

The main metric is **SIM-o**: cosine similarity between speaker-verification embeddings of the output and a *held-out* recording of the same speaker, never used as a prompt. Scoring against the prompt (**SIM-p**) would also reward copying its microphone and room. Naturalness is predicted by UTMOS, an automatic MOS predictor; intelligibility is Whisper-large-v3 word error rate.

Every effect is a speaker-level mean (targets and seeds are repeated measures) with a bootstrap 95% confidence interval, tested by Wilcoxon signed-rank with Holm correction.

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
    <strong>Figure 1.</strong> Removing prompt-level conditioning (x-vector only minus full): change in (a) speaker similarity to a held-out recording and (b) predicted naturalness. Speaker-level means with bootstrap 95% CIs; filled circles are clean LibriTTS-R prompts, open squares noisy Seed-TTS-eval prompts. Chatterbox's naturalness falls by 1.8, off the scale, as its speech becomes unintelligible.
  </figcaption>
</figure>

Apart from Chatterbox, the systems split cleanly by design (Figure 1a).

When the LM receives the embedding, identity largely survives. Both Qwen3-TTS sizes lose 0.062 to 0.073 similarity on LibriTTS-R and 0.099 to 0.108 on Seed-TTS-eval. CosyVoice-300M loses nothing on LibriTTS-R (+0.017, not significant) and 0.068 on Seed-TTS-eval.

When it reaches only the decoder, losses are two to three times larger: 0.184 and 0.244 for CosyVoice 3, and 0.224 and 0.213 for XTTS-v2 (all p < 0.001 after correction).

Within one family, CosyVoice 3 shares CosyVoice-300M's lineage and front end but, like CosyVoice 2, keeps the embedding out of the LM, and without a prompt on LibriTTS-R loses 0.18 where its predecessor loses nothing.

Chatterbox is the exception: its LM receives an embedding, but without its speech prompt it produces unintelligible speech (96 to 98% word error rate). No other system's word error rate suffers.

## Within one model: identity is split across sites

Differences in data, scale, tokenisers and decoders stop the systems above isolating the site. CosyVoice-300M can: it passes the *same* embedding to its LM and its flow-matching decoder. That allows a 2 × 2 × 2 factorial switching the prompt, LM embedding and decoder embedding independently, with text, reference, seed and decoding identical across the eight cells (6,480 utterances).

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
    <strong>Figure 2.</strong> CosyVoice-300M site factorial: speaker similarity for every combination of prompt and embedding sites. Means with bootstrap 95% CIs; 34 LibriTTS-R speakers and 268 Seed-TTS-eval prompt speakers.
  </figcaption>
</figure>

**Without the prompt, both sites carry identity, and the losses add.** On LibriTTS-R, removing the LM embedding costs 0.113 similarity, the decoder embedding 0.201. Removing both costs 0.310, close to the sum of 0.314, leaving similarity near the 0.14 floor of an unconditioned voice. Seed-TTS-eval matches: 0.092, 0.181 and 0.272 (sum 0.273).

**With the prompt, the sites differ.** Removing the LM embedding no longer matters (−0.010 and −0.008, within noise). Removing the decoder embedding still costs 0.070 and 0.057.

So at the LM, prompt and embedding are *substitutes*: either suffices as context for generating speech tokens. At the decoder they are *complements*: the prompt's acoustic features do not replace the embedding.

This may explain the cross-system result. Switching off CosyVoice-300M's LM site moves its prompt-free change on LibriTTS-R from +0.017 to −0.096, towards its decoder-only successor (−0.184). Systems conditioning both stages plausibly keep more identity without the prompt because they lose only one site.

## Prompts carry the recording, and the naturalness metric also rewards a generic voice

Removing the prompt barely changes predicted naturalness on clean LibriTTS-R (−0.04 to +0.07 in working systems). On noisy Seed-TTS-eval prompts it rises in all: +0.03 and +0.08 for the two Qwen3-TTS sizes, +0.13 for CosyVoice 3, +0.31 for CosyVoice-300M and +0.51 for XTTS-v2 (Figure 1b).

The prompt imports its recording conditions. With it, every system's output is closer to the prompt recording than to another recording of the same speaker (SIM-p above SIM-o). Removing it shrinks that gap by 0.010 to 0.053 in all twelve system and dataset pairs, the confidence interval excluding zero in eleven. For XTTS-v2 on Seed-TTS-eval the gap falls to 0.001: nothing recording-specific remains.

But UTMOS also rewards a generic voice. In the factorial on Seed-TTS-eval the prompt-free cell scores 4.04; removing both embeddings too, so no speaker information remains, raises it to 4.35. A naturalness gain that arrives with an identity loss is not better cloning, a caveat for any prompt-free result scored by automatic predictors.

## More references help identity, not naturalness

On LibriTTS-R, with three references instead of one (pooled over two selection strategies), Qwen3-TTS and CosyVoice 3 gain 0.022 to 0.044 similarity whether concatenated into the prompt or averaged as embeddings (p < 0.005 for every system and route). Naturalness moves by at most 0.02.

| System         | Route                    | Δ similarity | Δ naturalness |
| -------------- | ------------------------ | -----------: | ------------: |
| Qwen3-TTS 0.6B | prompt (concatenated)    |       +0.043 |        −0.008 |
|                | embedding (averaged)     |       +0.036 |        −0.003 |
| Qwen3-TTS 1.7B | prompt (concatenated)    |       +0.022 |        −0.018 |
|                | embedding (averaged)     |       +0.028 |        −0.001 |
| CosyVoice 3    | prompt (concatenated)    |       +0.044 |        +0.015 |
|                | embedding (averaged)     |       +0.031 |        +0.013 |
| F5-TTS         | prompt (concatenated)    |       −0.126 |        −0.722 |

The prompt-only F5-TTS gets *worse*, suggesting a model must be trained for prompt length.

## Listen

One LibriTTS-R item, drawn at random with a fixed seed. Every model says the same sentence: "I thanked him, but I was not deceived by his compliment."

The models heard a **reference prompt** of the speaker reading a *different* sentence. Similarity is SIM-o (seed 42), scored against a **held-out recording** no model heard.

**The speaker**

<div class="listen-row"><p>Reference prompt, heard by all</p><audio controls preload="none" src="/conditioning-sites/samples/3729_6852_000078_000003/prompt.mp3"></audio></div>
<div class="listen-row"><p>Held-out recording, heard by none</p><audio controls preload="none" src="/conditioning-sites/samples/3729_6852_000078_000003/heldout.mp3"></audio></div>

**Qwen3-TTS 0.6B**, embedding reaches the LM

<div class="listen-row"><p>Prompt + embedding: similarity 0.53</p><audio controls preload="none" src="/conditioning-sites/samples/3729_6852_000078_000003/qwen06_full.mp3"></audio></div>
<div class="listen-row"><p>Embedding only: similarity 0.60</p><audio controls preload="none" src="/conditioning-sites/samples/3729_6852_000078_000003/qwen06_xvector_only.mp3"></audio></div>

**CosyVoice 3**, embedding in the decoder only

<div class="listen-row"><p>Prompt + embedding: similarity 0.58</p><audio controls preload="none" src="/conditioning-sites/samples/3729_6852_000078_000003/cosyvoice3_full.mp3"></audio></div>
<div class="listen-row"><p>Embedding only: similarity 0.51</p><audio controls preload="none" src="/conditioning-sites/samples/3729_6852_000078_000003/cosyvoice3_xvector_only.mp3"></audio></div>

Here Qwen3-TTS scores *higher* without the prompt and CosyVoice 3 loses 0.07. Single clips vary: the 34-speaker averages are −0.07 and −0.18 (Figure 1).

The [demo page](/conditioning-sites/) adds a second LibriTTS-R speaker, two noisy Seed-TTS-eval prompts, every system and all eight factorial cells.

## What this means if you build with these models

- **In CosyVoice-300M, removing the embedding from the LM is cheap only while a prompt is always present.** With a prompt, the LM embedding made no measurable difference; without one, it accounted for about a third of the similarity lost when both embeddings went (0.113 of 0.310). Streaming or cross-lingual use, where the prompt is short or mismatched, plausibly depends on it too (untested).
- **For prompt-free cloning, condition both stages.**
- **Treat speaker embeddings as identity-bearing data.** Where the LM receives it, an embedding alone reproduces much of a voice (by automatic similarity), so a leaked one can act as a voice credential, as a recording can.
- **Don't trust automatic naturalness gains that coincide with identity loss.** Check similarity against a held-out recording, not the prompt.

This connects to the [Qwen3-TTS serving series](/posts/qwen-tts-streaming/): Qwen3-TTS kept most of its speaker similarity in x-vector-only mode, which also removes the reference recording from context. Its effect on time to first audio is unmeasured, and a natural next experiment for a voice service cloning from stored speaker profiles.

## Limitations

The causal factorial covers one model; the cross-system pattern rests on four working systems and system-specific interventions. Zeroing an embedding leaves the downstream projection bias, a standard but not unique ablation.

All metrics are automatic; a paired listening study is in preparation. Evaluation is English only, with one Seed-TTS-eval seed and one cross-system reference length.

## Reproduce it

With the per-utterance scores in the [repository](https://github.com/CallumJMac/conditioning-sites), every table and figure in the paper rebuilds in about two minutes without a GPU. `make check`, run on every push, recomputes the analysis and fails on any change. This post's figures come from `scripts/blog_figures.py`, which reads only the released analysis tables.
