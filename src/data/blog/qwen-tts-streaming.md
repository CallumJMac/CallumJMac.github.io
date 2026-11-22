---
pubDatetime: 2026-07-15T00:00:00Z
modDatetime: 2026-11-22T00:00:00Z
title: "How to Build a Low-Latency Streaming Qwen3-TTS Server"
slug: qwen-tts-streaming
featured: true
description: "Turning Qwen3-TTS into a deployable AWS streaming service with CUDA graphs and FastAPI WebSockets, then measuring where first-audio latency actually goes."
tags:
  - TTS
  - Qwen
  - CUDA
  - AWS
  - FastAPI
  - Voice Agents
---

The released Qwen3-TTS Python API returns a complete waveform: fine for offline synthesis, but it hides the model's causal audio path from latency-sensitive applications.

I built a streaming service around the 0.6B model, deployed it on AWS, and measured it:

- On an A10G behind an Application Load Balancer, the first audio byte arrived in **187 to 189ms at p50**.
- CUDA graphs cut generation time per codec frame by **4.7 to 4.9×** compared with the official path.
- First-audio latency grows by about **29ms per codec frame** in a packet. One-frame packets reached **108ms**.
- Every clip starts with about **420ms of silence**, so the listener hears speech **0.5 to 0.6s** after the request with packets of one to four frames.

The server, Terraform, per-request records and [analysis scripts](https://github.com/CallumJMac/qwen-tts-serve/tree/main/experiments/blog_analysis) are [open source](https://github.com/CallumJMac/qwen-tts-serve). A [companion capacity study](/posts/qwen-tts-at-scale/) covers behaviour under load.

## Why Qwen3-TTS?

Qwen3-TTS is small enough for one 24GB A10G, supports voice cloning, and represents audio at only 12.5 codec frames per second (80ms of audio per frame). Fewer frames mean fewer autoregressive steps.

Its codec decoder is causal, turning completed frames into audio without waiting for future ones. That enables real streaming.

## CUDA graphs, intuitively

Each codec frame holds 16 tokens, one per codebook, produced by two autoregressive transformers:

1. The **talker** (28 layers) predicts the frame's first token in one decoding step.
2. The **code predictor** (5 layers) fills in the other 15 in 15 sequential steps.

That is 16 dependent forward passes per frame at batch size one, each of many small kernels. Launched one by one from Python, they can cost more CPU time than GPU compute, leaving the GPU idle.

A CUDA graph records a fixed sequence of kernel launches once, then replays it with one CPU call. Replay reuses the captured shapes and memory addresses, so everything inside a graph must be static.

[`faster-qwen3-tts`](https://github.com/andimarafioti/faster-qwen3-tts) captures graphs for both stages. Prefill of the variable-length text prompt runs outside them. KV caches are preallocated at a fixed maximum length, so every replay reads and writes the same buffers.

<figure>
  <img
    src="/assets/qwen-tts/cuda-graph-capture-replay.svg"
    alt="Diagram contrasting repeated Python and CPU kernel launches with CUDA graph capture and single-launch replay. Variable-length prefill remains outside the graph."
    width="840"
    height="520"
    loading="lazy"
    decoding="async"
  />
  <figcaption>
    <strong>Figure 1.</strong> A CUDA graph records fixed-shape execution once and replays it with one launch per generation step; prefill stays outside. It is a recording of kernel launches, not the model's computational graph.
  </figcaption>
</figure>

With `chunk_size=4` (the client default), the server decodes and yields 320ms of audio every four completed frames:

```python
for audio, sample_rate, _timing in model.generate_custom_voice_streaming(
    text=text,
    speaker="ryan",
    language="English",
    chunk_size=4,
):
    yield audio.astype(np.float32), sample_rate
```

This is frame-level streaming: audio leaves the server while later frames are still generating. The official backend can only "generate everything, then slice".

## The service

The API is small:

```text
Client → WebSocket: {"text": "Hello world", "language": "English", "chunk_size": 4}
Server → Client:    [float32 PCM chunk]
Server → Client:    [float32 PCM chunk]
Server → Client:    {"done": true, "sample_rate": 24000}
```

FastAPI handles the WebSocket while the synchronous GPU generator runs in a thread executor. The client depends only on `websockets` and `numpy`, not PyTorch or CUDA.

The Base model also supports prompt-based voice cloning: a client registers a short reference clip once, then passes its `voice_id` in later requests. References live on local disk and disappear with the container.

## Deploying it on AWS

The Docker image bakes in the weights: **6.9GB compressed in ECR**, but startup does not depend on a Hugging Face download.

Terraform creates:

- ECR for the image;
- ECS on EC2, because Fargate does not expose GPUs;
- GPU-backed EC2 capacity (`g5.xlarge` in this test); and
- an Application Load Balancer for the WebSocket.

CUDA graph capture makes the first inference slow, so the server warms up before its health endpoint returns `200`, moving the cold start into deployment rather than a user's first request.

The one-instance deployment hit a rolling-update trap: ECS kept the old task running while placing the replacement, but the old task held the only GPU, stalling the deployment until I stopped it manually. Capacity and deployment policy must leave room for a replacement.

## Measured results

I ran the official and CUDA-graph engines one after the other on the same `g5.xlarge`: 0.6B CustomVoice model, fixed seed `20260714`, three inputs, three warm-ups, then 30 recorded runs per condition.

A client container on the same EC2 host measured endpoint latency through the ALB. Time to first audio (TTFA) runs from just before a request is sent on an established WebSocket to the first binary audio frame, so it excludes public-internet and connection-setup time.

| Input              | Official RTF p50 | CUDA graph RTF p50 | Speedup per frame | Deployed TTFA p50 [95% CI] |
| ------------------ | ---------------: | -----------------: | ----------------: | -------------------------: |
| Short (5 words)    |            1.71× |          **0.36×** |              4.7× |      **187.0ms** [186.8, 187.4] |
| Medium (~40 words) |            1.70× |          **0.35×** |              4.9× |      **188.3ms** [188.0, 188.5] |
| Long (~150 words)  |            1.71× |          **0.35×** |              4.9× |      **188.8ms** [188.2, 189.3] |

Real-time factor (RTF) is wall-clock generation time divided by audio duration; below 1.0 is faster than playback. Both RTF columns are full, non-streaming generation. As every frame is 80ms of audio, RTF converts directly to cost per frame: about **28 to 29ms** with CUDA graphs against **136ms** officially, and their ratio is the per-frame speedup.

The engines sampled different token sequences: CUDA-graph outputs were 6 to 20% shorter for the same text. Same-text wall-clock time improved by more, 5.2 to 5.9×, partly because fewer frames were generated, so per-frame cost is the fair measure of the engine.

Intervals are bootstrap 95% CIs over the 30 runs. With a fixed seed, each prompt's audio duration was constant, so the intervals capture timing noise on this host, not variation across prompts or machines.

The streaming endpoint's RTF is higher, 0.44 to 0.45, because decoding many small chunks costs more than one decode at the end. For a conversational agent, earlier audio is worth it.

Hardware and software: NVIDIA A10G 24GB (`g5.xlarge`), driver 550.163.01, PyTorch 2.6.0 + CUDA 12.4, `qwen-tts` 0.1.1, `faster-qwen3-tts` 0.2.6, `chunk_size=4`.

## Where first-audio latency goes

A week later I instrumented the server and swept the packet size, which is the same for every packet including the first: one, two, four and eight codec frames. Each condition sent 60 requests, one at a time, from a 20-prompt voice-agent corpus, on localhost and through the ALB.

<figure>
  <img
    src="/assets/qwen-tts/ttfa-decomposition.svg"
    alt="Line plot of median time to first audio against 1, 2, 4 and 8 codec frames per first packet, following a straight line of 79.5 plus 28.6 milliseconds per frame. Prefill and unattributed engine time stay near 40 milliseconds; first-packet decode grows linearly."
    width="422"
    height="280"
    loading="lazy"
    decoding="async"
  />
  <figcaption>
    <strong>Figure 2.</strong> Median time to first audio (large points; bootstrap 95% CIs are smaller than the markers) against first-packet size, with its linear fit (dashed) and server-side stages, on localhost. First-packet decode accounts for almost all of the growth. Data: <code>ttfa-stages.csv</code>.
  </figcaption>
</figure>

Prefill (38ms) and an unattributed block of engine work (41ms) are fixed costs. First-packet decode costs 27.6ms per frame, and sending the larger packet adds about 1ms per frame. Together they give **TTFA ≈ 79.5ms + 28.6ms × frames**, which matches all four measured medians to within 0.4ms. The 27.6ms agrees with the per-frame cost implied by the full-generation RTF.

The unattributed 41ms is engine time my timers did not split further; it can include request preparation, waveform decoding and device-to-host copies, and is the next thing to instrument.

Also from the sweep:

- **The ALB is not the bottleneck.** Paired localhost and ALB conditions differed by **0.9 to 1.4ms** at p50.
- **The 8ms gap between studies comes from the later setup, not the route or prompt.** The same prompt ("Hello, how are you today?") measured 187ms in the 15 July deployment and 195ms in both later studies, on localhost and through the ALB. Those used a rebuilt image on a different host setup; I have not isolated which change mattered.

The Qwen3-TTS README advertises "end-to-end synthesis latency as low as 97ms" without stating the hardware, model size, packet size or measurement method. My A10G medians were 187 to 195ms with four-frame packets and 108ms with one-frame packets; without a matching definition, the gap cannot be attributed to implementation or hardware.

Smaller packets improve onset but cost throughput: streaming RTF rose from 0.45 at four frames to 0.70 at one frame. The [capacity study](/posts/qwen-tts-at-scale/) tests them under real traffic.

## What the listener actually waits for

TTFA measures the first audio *byte*, not the first audible word. I measured leading silence in one retained clip per prompt and condition as the time until a 20ms frame's RMS level first exceeds -45 dBFS. Figure 3 also shows vLLM-Omni, the second backend in the [capacity study](/posts/qwen-tts-at-scale/).

<figure>
  <img
    src="/assets/qwen-tts/first-speech.svg"
    alt="Panel a: waveform envelope of one clip aligned to the request, with the first byte at 196 milliseconds and the first speech at 620 milliseconds. Panel b: leading silence for 20 prompts per backend, clustered between 420 and 540 milliseconds."
    width="548"
    height="276"
    loading="lazy"
    decoding="async"
  />
  <figcaption>
    <strong>Figure 3.</strong> (a) Amplitude envelope of one streamed clip ("assistant" prompt, four-frame packets) aligned to the request. (b) Leading silence for each of the 20 prompts per backend, four-frame packets on localhost; horizontal bars are medians.
  </figcaption>
</figure>

Every clip on both backends opens with **420 to 540ms** of near-silence. The full, non-streamed generations in the next section do too (420 to 480ms), so the silence comes from the model, not streaming. With no client buffer, the first word starts roughly **TTFA + 420ms** after the request: 528ms with one-frame packets and 614ms with the default four.

That changes the priority: shrinking packets from four frames to one saves 87ms, while the leading silence costs more than four times that on every request.

Dropping silent leading audio would start playback at the first voiced frame, but the five or six silent frames must still be generated, so the saving would be under 420ms. That is my next experiment.

For a voice agent, TTS is one part of the turn budget:

```text
turn latency ≈ end-of-speech detection + ASR finalisation
             + LLM time to first token + text chunking
             + TTS time to first byte + leading silence
             + network + client buffer
```

This post measures only the TTS terms: 108 to 195ms to first byte (one to four frames per packet), about 420ms of silence, and about 1ms of ALB overhead.

## Does streaming change the audio?

I generated ten sentences in full and streamed, then compared:

- **Words.** Whisper `small.en` transcripts matched for all ten pairs, after normalising punctuation.
- **Voice.** ECAPA speaker-embedding cosine similarity was **0.994 to 0.998** for nine pairs and 0.81 for one.
- **Boundaries.** No chunk-boundary sample jump exceeded the 99th percentile of ordinary sample-to-sample jumps within its clip.
- **Waveforms.** Not bit-identical: after alignment, the nine matching pairs had a signal-to-noise ratio of 18.6 to 28.6dB.

The outlier was "The train arrives at half past seven.", streamed 800ms shorter. It has the same words and leading and trailing silence within 20ms of the full version, so it is a faster rendition, not a truncation: streaming changed the sampled path, and the voice embedding moved with it.

A typical pair:

**Full generation**

<audio controls preload="none" src="/audio/qwen-tts/question-full.wav"></audio>

**Streaming**

<audio controls preload="none" src="/audio/qwen-tts/question-streamed.wav"></audio>

The divergent pair:

**Full generation**

<audio controls preload="none" src="/audio/qwen-tts/short-statement-full.wav"></audio>

**Streaming**

<audio controls preload="none" src="/audio/qwen-tts/short-statement-streamed.wav"></audio>

These are objective proxies, not a listening study, but they show streaming preserved the words and, in nine of ten cases, the voice.

## What this is and is not

This is a reproducible, deployable prototype, not a production service.

- Generation is serialised to protect the shared engine; the [capacity study](/posts/qwen-tts-at-scale/) found this meets the latency SLO at only one active request per replica.
- Voice references are ephemeral.
- The test ALB used plain HTTP/WebSocket with no authentication.
- Disconnecting stops delivery to the client, but I have not shown it cancels in-flight GPU work immediately.
- Leading silence is untrimmed.

## Try it

```bash
git clone https://github.com/CallumJMac/qwen-tts-serve.git
cd qwen-tts-serve
uv sync --extra cuda --extra dev

QWEN_TTS_ENGINE=faster uv run uvicorn qwen_tts_serve.server:app \
  --host 0.0.0.0 --port 8000
uv run python scripts/stream_demo.py "Hello world"
```

The streaming path needs an NVIDIA CUDA machine. Every number and figure here is generated from the retained per-request records by [`experiments/blog_analysis`](https://github.com/CallumJMac/qwen-tts-serve/tree/main/experiments/blog_analysis) in the same repository, which needs no GPU.
