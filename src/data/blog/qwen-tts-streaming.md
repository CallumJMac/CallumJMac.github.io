---
pubDatetime: 2026-07-15T00:00:00Z
title: "How to Build a Low-Latency Streaming Qwen3-TTS Server"
slug: qwen-tts-streaming
featured: true
description: "Turning Qwen3-TTS into a deployable AWS streaming service with CUDA graphs, FastAPI WebSockets, and measured sub-200ms first-audio latency."
tags:
  - TTS
  - Qwen
  - CUDA
  - AWS
  - FastAPI
  - Voice Agents
---

The released Qwen3-TTS Python API returns a complete waveform. That interface is suitable for offline synthesis, but it does not expose the model's causal audio path to latency-sensitive applications.

I built a streaming service around the 0.6B model and deployed it on AWS. On an A10G, the endpoint returned its first audio in **187–189ms at p50** through an Application Load Balancer. CUDA graphs reduced generation RTF from roughly **1.70 to 0.35**.

This post covers the inference runtime, WebSocket API, AWS deployment, and measurement methodology. The implementation and Terraform are [open source](https://github.com/CallumJMac/qwen-tts-serve).

## Why Qwen3-TTS?

Qwen3-TTS is small enough for one 24GB A10G, supports voice cloning, and represents audio at only 12.5 codec frames per second. Fewer frames mean fewer autoregressive steps.

Its decoder is causal: it can turn completed frames into audio without waiting for future frames. That makes real streaming possible.

The official Python API, however, returns only after the whole utterance has been generated. My first "streaming" implementation simply sliced that finished waveform:

```python
samples, sample_rate = model.generate(text)  # waits for everything
for chunk in split(samples):
    yield chunk, sample_rate
```

The client received chunks, but only after the full wait. It looked like streaming without improving latency.

## CUDA graphs, intuitively

Generating one audio frame runs hundreds of small GPU operations. Normally, Python asks the GPU to launch each operation separately. For a large batch, the useful computation hides that overhead. For one frame at a time, the repeated launches matter.

A CUDA graph records the fixed sequence once and replays it with one graph launch. Think of recording a kitchen's entire preparation routine rather than calling out every individual step for every order.

Qwen3-TTS has two repeated stages:

1. The **talker** predicts the first codec token.
2. The **code predictor** fills in the other 15 tokens for that frame.

[`faster-qwen3-tts`](https://github.com/andimarafioti/faster-qwen3-tts) captures one graph for each stage. Variable-length prompt processing stays outside the graphs; generated tokens go into preallocated KV caches with fixed shapes.

<figure>
  <img
    src="/assets/qwen-tts/cuda-graph-capture-replay.svg"
    alt="Diagram contrasting repeated Python and CPU kernel launches with CUDA graph capture and single-launch replay. Variable-length prefill remains outside the graph."
    width="840"
    height="560"
    loading="lazy"
    decoding="async"
  />
  <figcaption>
    CUDA graph replay records fixed-shape runtime execution once, then replays it with one launch per generation step. Variable-length prefill remains outside the captured graphs, which are runtime recordings rather than the model computational graph.
  </figcaption>
</figure>

Every four completed frames, I decode and yield roughly 320ms of audio:

```python
for audio, sample_rate, _timing in model.generate_custom_voice_streaming(
    text=text,
    speaker="ryan",
    language="English",
    chunk_size=4,
):
    yield audio.astype(np.float32), sample_rate
```

That is the difference between "generate, then slice" and actual token-level streaming.

## The service

The API is intentionally small:

```text
Client → WebSocket: {"text": "Hello world", "language": "English"}
Server → Client:    [float32 PCM chunk]
Server → Client:    [float32 PCM chunk]
Server → Client:    {"done": true, "sample_rate": 24000}
```

FastAPI handles the WebSocket while the synchronous GPU generator runs in a thread executor. The client package depends only on `websockets` and `numpy`, not PyTorch or CUDA.

The Base model also supports prompt-based voice cloning. A client registers a short reference clip once, then supplies its `voice_id` in later WebSocket requests. The current implementation keeps those references on local disk, so they disappear with the container.

## Deploying it on AWS

The Docker image contains the model weights. At about **6.9GB compressed in ECR**, it is large, but startup never races a Hugging Face download.

Terraform creates:

- ECR for the image
- ECS on EC2, because Fargate does not expose GPUs
- GPU-backed EC2 capacity (`g5.xlarge` in this test)
- an Application Load Balancer for the WebSocket

CUDA graph capture makes the first inference slow. The server therefore warms the model before its health endpoint returns `200`. The cold start still exists, but it happens during deployment rather than on a user's first request.

The one-instance benchmark deployment also exposed a rolling-update trap. ECS tried to keep the old task healthy while placing the replacement, but the old task still owned the GPU. The deployment could not progress until I stopped the old task. GPU capacity and deployment policy must explicitly allow room for replacement tasks.

## Measured results

I ran the official and CUDA-graph engines sequentially on the same `g5.xlarge`, using the 0.6B CustomVoice model, a fixed seed, and the same three inputs. Each condition had three warm-ups followed by 30 recorded runs.

Endpoint latency used one isolated client container on the same EC2 host, with every request still traversing the ALB. TTFA starts immediately before sending the request on an established WebSocket and ends at the first binary audio frame, so it excludes public-internet and connection-setup latency. A separate [capacity study](/posts/qwen-tts-at-scale/) remeasured TTFA on a 20-prompt corpus under load; at concurrency one it reported **195ms p50** and **197ms p95**, slightly higher than these three-prompt endpoint runs.

| Input              | Official RTF p50 | CUDA graph RTF p50 | Deployed TTFA p50 / p95 |
| ------------------ | ---------------: | -----------------: | ----------------------: |
| Short (5 words)    |            1.71x |          **0.36x** |       **187ms / 188ms** |
| Medium (~40 words) |            1.70x |          **0.35x** |       **188ms / 190ms** |
| Long (~150 words)  |            1.71x |          **0.35x** |       **189ms / 190ms** |

RTF is generation time divided by audio duration, so lower is better and anything below 1.0 is faster than playback. The CUDA-graph path was roughly **4.8× faster** than the official path and generated audio at about **2.8× real time**.

The streaming endpoint itself had a higher RTF of roughly 0.44 because repeatedly decoding small chunks trades some throughput for earlier audio. That is the trade I want for a conversational agent.

The engines did not produce identical durations: optimised audio was 6–20% shorter across these three prompts. RTF normalises by each output's duration, so the speedup is useful but not an identical-output comparison.

Hardware: NVIDIA A10G 24GB (`g5.xlarge`), driver 550.163.01, PyTorch 2.6.0 + CUDA 12.4, `qwen-tts` 0.1.1, `faster-qwen3-tts` 0.2.6, `chunk_size=4`.

## Does streaming change the audio?

Usually slightly, but one of ten pairs ended 800ms early. Full and streamed waveforms were not bit-identical.

Nine of ten single test pairs had identical durations, with aligned SNR between **18.6 and 28.6dB**; higher SNR means closer waveforms. One short sentence ended 800ms earlier in streaming mode and diverged substantially. Importantly, no chunk-boundary jump exceeded the 99th percentile of ordinary sample-to-sample jumps within its clip. The objective data therefore found no special spike at chunk boundaries, but it did find a real end-of-utterance edge case. This was not a systematic perceptual study, so the samples are included below for direct comparison.

Here is a representative high-SNR pair:

**Full generation**

<audio controls preload="none" src="/audio/qwen-tts/question-full.wav"></audio>

**Streaming**

<audio controls preload="none" src="/audio/qwen-tts/question-streamed.wav"></audio>

And the duration-mismatch case:

**Full generation**

<audio controls preload="none" src="/audio/qwen-tts/short-statement-full.wav"></audio>

**Streaming**

<audio controls preload="none" src="/audio/qwen-tts/short-statement-streamed.wav"></audio>

This is why I would not replace quality testing with a single latency number.

## What this is and is not

This is a reproducible, deployable prototype, not a finished production service.

- Generation is serialised to protect the shared engine. A separate [A10G capacity study](/posts/qwen-tts-at-scale/) found that this architecture meets the latency SLO at only one active request per replica under load.
- Voice references are ephemeral.
- The test ALB used plain HTTP/WebSocket with no authentication.
- Disconnecting stops delivery to the client, but I have not proved that it immediately cancels in-flight GPU work.
- Streaming is faster to first audio but slower in total throughput than full CUDA-graph generation.

Those are engineering tasks, not model problems, but they matter before serving real traffic.

## Try it

```bash
git clone https://github.com/CallumJMac/qwen-tts-serve.git
cd qwen-tts-serve
uv sync --extra cuda --extra dev

QWEN_TTS_ENGINE=faster uv run uvicorn qwen_tts_serve.server:app \
  --host 0.0.0.0 --port 8000
uv run python scripts/stream_demo.py "Hello world"
```

This true-streaming path requires an NVIDIA CUDA machine. The gap between a checkpoint and a responsive service is larger than `model.generate()`: CUDA graphs substantially reduced inference overhead; warm-up, health checks, packaging, WebSockets, and deployment behaviour handled the rest.
