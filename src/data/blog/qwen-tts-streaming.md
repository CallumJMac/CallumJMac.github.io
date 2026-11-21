---
pubDatetime: 2026-07-15T00:00:00Z
modDatetime: 2026-11-21T00:00:00Z
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

The released Qwen3-TTS Python API returns a complete waveform. That interface suits offline synthesis, but it hides the model's causal audio path from latency-sensitive applications.

I built a streaming service around the 0.6B model, deployed it on AWS, and measured it. The short version:

- On an A10G behind an Application Load Balancer, the first audio byte arrived in **187ms at p50** (95% CI 186.8 to 187.4ms).
- CUDA graphs cut generation time for the same text by **5.2 to 5.9×** compared with the official path.
- First-audio latency grows by about **29ms per codec frame** in the first packet. One-frame packets reached **108ms**.
- Every clip starts with about **420ms of silence**. The listener hears speech roughly half a second after the request, whichever packet size you pick.

This post covers the runtime, WebSocket API, AWS deployment, and measurements. The server and Terraform are [open source](https://github.com/CallumJMac/qwen-tts-serve); the per-request records and analysis scripts behind every figure will follow in the same repository. A [companion capacity study](/posts/qwen-tts-at-scale/) covers behaviour under load.

## Why Qwen3-TTS?

Qwen3-TTS is small enough for one 24GB A10G, supports voice cloning, and represents audio at only 12.5 codec frames per second (80ms of audio per frame). Fewer frames mean fewer autoregressive steps.

Its decoder is causal: it can turn completed frames into audio without waiting for future frames. That is what makes real streaming possible.

## CUDA graphs, intuitively

Generating one audio frame runs hundreds of small GPU operations. Normally, the CPU schedules each one separately. For a large batch, the useful computation hides that overhead. For one frame at a time, the repeated scheduling and launch costs dominate.

A CUDA graph records a fixed sequence of GPU operations once, then replays it with one CPU launch.

Qwen3-TTS has two repeated stages:

1. The **talker** predicts the first codec token of a frame.
2. The **code predictor** fills in the other 15 tokens for that frame.

[`faster-qwen3-tts`](https://github.com/andimarafioti/faster-qwen3-tts) captures one graph for each stage. Variable-length prompt processing stays outside the graphs, and generated tokens go into preallocated KV caches with fixed shapes.

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
    <strong>Figure 1.</strong> CUDA graph replay records fixed-shape execution once, then replays it with one launch per generation step. Prefill stays outside the captured graphs. A CUDA graph is a recording of runtime execution, not the model's computational graph.
  </figcaption>
</figure>

Every four completed frames, the server decodes and yields 320ms of audio:

```python
for audio, sample_rate, _timing in model.generate_custom_voice_streaming(
    text=text,
    speaker="ryan",
    language="English",
    chunk_size=4,
):
    yield audio.astype(np.float32), sample_rate
```

This is token-level streaming, not "generate everything, then slice".

## The service

The API is intentionally small:

```text
Client → WebSocket: {"text": "Hello world", "language": "English"}
Server → Client:    [float32 PCM chunk]
Server → Client:    [float32 PCM chunk]
Server → Client:    {"done": true, "sample_rate": 24000}
```

FastAPI handles the WebSocket while the synchronous GPU generator runs in a thread executor. The client package depends only on `websockets` and `numpy`, not PyTorch or CUDA.

The Base model also supports prompt-based voice cloning. A client registers a short reference clip once, then passes its `voice_id` in later requests. References currently live on local disk, so they disappear with the container.

## Deploying it on AWS

The Docker image bakes in the model weights. At **6.9GB compressed in ECR** it is large, but startup does not depend on a Hugging Face download.

Terraform creates:

- ECR for the image;
- ECS on EC2, because Fargate does not expose GPUs;
- GPU-backed EC2 capacity (`g5.xlarge` in this test); and
- an Application Load Balancer for the WebSocket.

CUDA graph capture makes the first inference slow, so the server warms the model before its health endpoint returns `200`. The cold start still exists, but it happens during deployment, not on a user's first request.

The one-instance deployment also hit a rolling-update trap. ECS kept the old task running while placing the replacement, but the old task still held the only GPU, so the deployment stalled until I stopped it manually. GPU capacity and deployment policy must leave room for replacement tasks.

## Measured results

I ran the official and CUDA-graph engines one after the other on the same `g5.xlarge`: 0.6B CustomVoice model, fixed seed `20260714`, three inputs, three warm-ups, then 30 recorded runs per condition.

Endpoint latency came from a client container on the same EC2 host, with every request still passing through the ALB. Time to first audio (TTFA) starts just before the request is sent on an established WebSocket and ends at the first binary audio frame. It therefore excludes public-internet and connection-setup time.

| Input              | Official RTF p50 | CUDA graph RTF p50 | Same-text speedup | Deployed TTFA p50 [95% CI] |
| ------------------ | ---------------: | -----------------: | ----------------: | -------------------------: |
| Short (5 words)    |            1.71× |          **0.36×** |              5.9× |      **187.0ms** [186.8, 187.4] |
| Medium (~40 words) |            1.70× |          **0.35×** |              5.2× |      **188.3ms** [188.0, 188.5] |
| Long (~150 words)  |            1.71× |          **0.35×** |              5.2× |      **188.8ms** [188.2, 189.3] |

Real-time factor (RTF) is generation time divided by audio duration, so below 1.0 is faster than playback. The CUDA-graph path spends about **28 to 29ms per 80ms codec frame**, against 136ms for the official path.

The two engines did not produce identical audio. CUDA-graph outputs were 6 to 20% shorter for the same text, which inflates an RTF ratio. The "same-text speedup" column avoids that problem by comparing wall-clock generation time for identical input: 5.2 to 5.9×.

The intervals are bootstrap 95% confidence intervals over the 30 runs. With a fixed seed, each prompt produces the same audio every time, so they capture run-to-run timing noise on this host. They say nothing about other prompts or other machines.

The streaming endpoint has a higher RTF of 0.44 to 0.45, because decoding many small chunks trades some throughput for earlier audio. That is the right trade for a conversational agent.

Hardware and software: NVIDIA A10G 24GB (`g5.xlarge`), driver 550.163.01, PyTorch 2.6.0 + CUDA 12.4, `qwen-tts` 0.1.1, `faster-qwen3-tts` 0.2.6, `chunk_size=4`.

## Where first-audio latency goes

A week later I instrumented the server and swept the size of the first packet: one, two, four and eight codec frames. Each condition used a 20-prompt voice-agent corpus with 60 requests, sent one at a time, both on localhost and through the ALB.

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
    <strong>Figure 2.</strong> Median time to first audio (large points; bootstrap 95% CIs are smaller than the markers) against first-packet size, with its linear fit (dashed) and server-side stages. Localhost, one request at a time, 60 requests per packet size. First-packet decode is the only stage that grows. Data: <code>ttfa-stages.csv</code>.
  </figcaption>
</figure>

Figure 2 has a simple structure. Prefill (38ms) and an unattributed block of engine work (41ms) are fixed costs. First-packet decode costs 27.6ms per frame. Together they give **TTFA ≈ 79.5ms + 28.6ms × frames**, which matches all four measured medians to within 0.4ms.

The 41ms "unattributed" block is time inside the engine that my timers did not split further. It can include request preparation, waveform decoding and device-to-host copies. Instrumenting it is the next profiling step.

Two other results from this sweep:

- **The ALB is not the bottleneck.** Paired localhost and ALB conditions differed by **0.9 to 1.4ms** at p50.
- **The 8ms gap between studies comes from the build, not the route or prompt.** The same prompt ("Hello, how are you today?") measured 187ms in the July 15 deployment and 195ms in both later studies, on localhost and through the ALB alike. The later studies used a rebuilt image on a different host setup. I have not isolated which change added the 8ms.

The Qwen3-TTS README advertises "end-to-end synthesis latency as low as 97ms". It does not state the hardware, model size, packet size or how the latency is measured. My best four-frame result on an A10G was 194ms, and one-frame packets reached 108ms. Without a matching definition, the gap cannot be attributed to the implementation or the hardware.

Smaller packets improve onset in isolation. Whether they survive real traffic is a separate question, covered in the [capacity study](/posts/qwen-tts-at-scale/).

## What the listener actually waits for

TTFA measures the first audio *byte*, not the first audible word. I measured leading silence in every retained clip: the time until the signal first rises above -45 dBFS, checked against a -30 dBFS peak threshold.

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
    <strong>Figure 3.</strong> (a) Amplitude envelope of one streamed clip ("assistant" prompt, four-frame packets) aligned to the request. (b) Leading silence before speech for each of the 20 prompts on each backend; horizontal bars are medians. First speech arrives about 420ms after the first audio byte.
  </figcaption>
</figure>

Every clip on both backends opens with **420 to 540ms** of near-silence before speech. With no client buffer, the first word therefore starts roughly **TTFA + 420ms** after the request. That is 528ms at the fastest packet size and 614ms at the standard four frames.

This changes the optimisation priority. Shrinking packets from four frames to one saves 87ms. The leading silence costs more than four times that, on every request.

A server that drops silent leading audio could start playback as soon as the first voiced frame is decoded. That removes playback delay but not generation time, so the saving would be smaller than 420ms. I have not measured it yet. It is the next experiment I would run.

For a voice agent, the TTS term sits inside a longer turn budget:

```text
turn latency ≈ end-of-speech detection + ASR finalisation
             + LLM time to first token + text chunking
             + TTS time to first byte + leading silence
             + network + client buffer
```

This post measures only the TTS terms: 108 to 195ms to first byte, about 420ms of silence, and about 1ms of ALB overhead.

## Does streaming change the audio?

For ten test sentences I generated each one twice, once in full and once streamed, then compared:

- **Words.** Whisper `small.en` produced identical transcripts for all ten pairs.
- **Voice.** ECAPA speaker-embedding cosine similarity was **0.994 to 0.998** for nine pairs. One pair scored 0.81.
- **Boundaries.** No chunk-boundary sample jump exceeded the 99th percentile of ordinary sample-to-sample jumps within its clip.
- **Waveforms.** They were not bit-identical. After alignment, the nine matching pairs had a signal-to-noise ratio of 18.6 to 28.6dB.

The outlier was "The train arrives at half past seven." The streamed version was 800ms shorter. It contains the same words with the same leading and trailing silence, so it is a faster rendition of the sentence, not a truncation. Streaming changed the sampled path for that input, and the voice embedding moved with it.

Here is a typical pair:

**Full generation**

<audio controls preload="none" src="/audio/qwen-tts/question-full.wav"></audio>

**Streaming**

<audio controls preload="none" src="/audio/qwen-tts/question-streamed.wav"></audio>

And the divergent pair:

**Full generation**

<audio controls preload="none" src="/audio/qwen-tts/short-statement-full.wav"></audio>

**Streaming**

<audio controls preload="none" src="/audio/qwen-tts/short-statement-streamed.wav"></audio>

ASR and speaker embeddings are objective proxies, not a listening study. They show that streaming preserved the words and, in nine of ten cases, the voice.

## What this is and is not

This is a reproducible, deployable prototype, not a finished production service.

- Generation is serialised to protect the shared engine. The [capacity study](/posts/qwen-tts-at-scale/) found that this meets the latency SLO at only one active request per replica.
- Voice references are ephemeral.
- The test ALB used plain HTTP/WebSocket with no authentication.
- Disconnecting stops delivery to the client, but I have not shown that it immediately cancels in-flight GPU work.
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

The streaming path needs an NVIDIA CUDA machine. Every number and figure in this post is generated from the retained per-request records by an analysis package that needs no GPU. I will publish it, with the records, in the same repository.
