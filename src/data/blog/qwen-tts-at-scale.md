---
pubDatetime: 2026-07-17T00:00:00Z
title: "Capacity Planning Qwen3-TTS on an A10G"
slug: qwen-tts-at-scale
featured: true
description: "A single-GPU capacity study for streaming Qwen3-TTS: SLOs, vLLM-Omni comparison, fleet estimates, and a proposed production architecture."
tags:
  - TTS
  - Qwen
  - vLLM
  - AWS
  - Load Testing
  - MLOps
---

A fast demo is not necessarily a scalable service.

My [streaming Qwen3-TTS server](/posts/qwen-tts-streaming/) reports **187–189ms p50** time to first audio (TTFA) on three fixed prompts through the ALB. This study used a different 20-prompt corpus and measured load under concurrency. At concurrency one, p50 TTFA was **195ms** and p95 was **197ms**. With two concurrent requests, median TTFA rises to **3.4 seconds**.

All **1,200 requests succeeded**, but most conditions still violated the latency or playback SLOs. For a voice agent, a successful HTTP response can still mean silence or stutter.

I tested the original server and vLLM-Omni on one A10G, then used the results to answer three questions:

1. How much traffic can one GPU safely handle?
2. What would hundreds of active requests require?
3. What should a production deployment scale on?

## Define “safe” first

Capacity depends on the user experience you are willing to accept. I defined a replica as safe only when every condition held:

- at least **99%** of requests succeeded;
- p95 time to first audio (TTFA) was at most **300ms**; and
- p95 playback starvation after the first audio byte was at most **100ms**.

The last metric needs some explanation. The simulated player started immediately when the first audio arrived, with no startup or jitter buffer. TTFA itself did not count as starvation. After playback began, any gap where the player ran out of audio did.

This is deliberately strict. A real client could buffer audio before playing it, trading a later start for smoother playback. The result is therefore not “vLLM-Omni cannot stream TTS.” It is “this configuration did not sustain uninterrupted zero-buffer playback under this SLO.”

## The experiment

Both backends ran sequentially on the same AWS `g5.xlarge` in `us-east-1`:

- NVIDIA A10G 24GB;
- Qwen3-TTS 0.6B CustomVoice;
- a corpus of 20 voice-agent prompts;
- 60 measured requests per condition; and
- streaming chunks of four codec frames.

The **current server** uses `faster-qwen3-tts` and serialises generation with an application lock. That protects its shared, non-thread-safe engine, but only one request can use the GPU at a time.

The comparison backend used **vLLM-Omni**, which schedules multiple sequences together to increase GPU utilisation.

I used two load patterns:

- **Closed loop:** keep 1, 2, 4, 8, 16, or 32 requests in flight.
- **Open loop:** introduce requests at 0.25, 0.5, 1, or 2 requests per second, whether earlier requests have finished or not.

Closed-loop tests answer “what happens with this many active users?” Open-loop tests expose queue growth when arrivals approach or exceed service capacity.

The benchmark client recorded every request, including failures and outliers. It also rejected empty, malformed, non-finite, or non-24kHz audio rather than counting a bad stream as a success. The server, Terraform, and harness code are in [qwen-tts-serve](https://github.com/CallumJMac/qwen-tts-serve). This run is study `20260716T121717Z-capacity` (1,200 requests, 20-prompt corpus, per-request JSONL with TTFA and starvation timings).

## The result: throughput is not enough

These four conditions capture the main result:

<figure>
  <img
    src="/assets/qwen-tts/closed-loop-p95-ttfa.svg"
    alt="Log-scale line chart of p95 time to first audio against closed-loop concurrency for the current server and vLLM-Omni, with the 300 millisecond TTFA SLO marked."
    width="840"
    height="560"
    loading="lazy"
    decoding="async"
  />
  <figcaption>
    Measured p95 TTFA by closed-loop concurrency, using a logarithmic y-axis in seconds so the 0.17–105.16s range and 300ms SLO remain visible. Each condition contains 60 measured requests.
  </figcaption>
</figure>

This TTFA-only view shows the current server crossing the 300ms threshold at concurrency two, while vLLM-Omni crosses it at concurrency four.

| Backend           |           Load | p95 TTFA | p95 starvation | Audio generated / wall second | SLO-safe? |
| ----------------- | -------------: | -------: | -------------: | ----------------------------: | --------- |
| Serialised server |  concurrency 1 |    197ms |            0ms |                         2.19s | Yes       |
| Serialised server |  concurrency 2 |    4.98s |            0ms |                         2.19s | No        |
| vLLM-Omni         |  concurrency 1 |    174ms |          214ms |                         3.73s | No        |
| vLLM-Omni         | concurrency 16 |    4.85s |          1.27s |                        12.66s | No        |

The serialised server processes one request at a time. Once a request starts, it receives audio consistently. Additional requests wait, so throughput stays around **0.29 requests per second** while TTFA grows with the queue. At concurrency 32, p95 TTFA reached **105 seconds**.

vLLM-Omni achieved substantially higher aggregate throughput. It peaked around **1.66 requests per second** and generated **12.66 seconds of audio per wall-clock second**, about **5.7×** the request throughput and **5.8×** the audio throughput of the current server.

But higher aggregate throughput did not satisfy the playback SLO. Even at concurrency one, vLLM-Omni's p95 TTFA was 174ms while post-first-byte starvation reached 214ms. **TTFA and throughput can both look good while playback still stutters.**

The outcome depends on `initial_codec_chunk_frames=4` and the zero-buffer player. A different chunk size or a short client buffer could improve continuity, but would change initial latency. That trade-off needs another measurement, not an assumption.

A later [packet-size study](/posts/qwen-tts-streaming/#where-first-audio-latency-goes) reduced isolated TTFA to **108ms**, but no single-replica sustained policy met every SLO. Packet tuning improves onset; capacity and admission remain separate problems.

## What would 100 concurrent requests cost?

Under the chosen SLO, the current backend's measured safe capacity is one active request per replica. I applied a **1.4× safety factor on the minimum fleet size**:

`estimated replicas = ceil(target active requests / safe capacity × 1.4)`

That adds 40% more GPUs than the bare minimum (140 for 100 active streams), not a target of 60% fleet utilisation. Sizing for 60% utilisation would use `ceil(target / safe capacity / 0.6)` instead, which requires **167** A10Gs for 100 streams.

| Simultaneously active requests | Estimated A10Gs | Approx. on-demand compute/hour |
| -----------------------------: | --------------: | -----------------------------: |
|                            100 |             140 |                           $141 |
|                            250 |             350 |                           $352 |
|                            500 |             700 |                           $704 |

The cost uses the recorded `g5.xlarge` price of **$1.006 per hour** in `us-east-1` on 16 July 2026.

These are deliberately conservative **estimates**, not tested fleet sizes. They assume replicas scale independently and exclude routing, autoscaling delays, and failure recovery. The study did not validate a multi-replica deployment.

They are still useful because they show that horizontally replicating the serialised server is inefficient at these target loads. Before provisioning hundreds of GPUs, I would improve per-replica scheduling and remeasure the safe boundary.

I also would not size this service from average RPS alone. The current backend failed the TTFA SLO even at the lowest open-loop rate, 0.25 RPS. Variable request duration and burst timing create queues despite an average completion rate near 0.29 RPS.

## The production architecture I would build

This is the next design, not something the experiment proved:

```text
Clients
   │
ALB / WebSocket upgrade
   │
Admission control ── reject or defer work that cannot meet the SLO
   │
Bounded request queues
   │
GPU replicas ── streaming scheduler / continuous batching
   │
Metrics: active streams, queue wait, TTFA, starvation, GPU utilisation
```

An upgraded WebSocket remains attached to one replica. The load balancer can distribute new connections, but it cannot move an active stream when a replica becomes busy. Scale-in therefore needs connection draining, and routing needs to consider active streams rather than just instance count.

I would autoscale primarily from:

- active streams per ready replica;
- queue depth and oldest-request wait;
- p95 TTFA;
- playback-starvation telemetry; and
- GPU utilisation and memory.

Requests per minute is a weak primary signal here. Two prompts can occupy the GPU for very different lengths of time, and a long-lived WebSocket represents ongoing work rather than one completed invocation.

Backpressure is equally important. Queues should be bounded. If the predicted wait already makes a 300ms TTFA impossible, returning an overload response is preferable to accepting work that cannot meet the SLO.

New replicas must also load the model, capture CUDA graphs, and warm up before becoming healthy. That makes reactive scale-out slow, so the fleet needs warm headroom. Scale-in should be slower and should wait for existing streams to drain.

## What is proved and what is not

On one A10G, all 1,200 requests succeeded. The serialised backend met the full SLO only at concurrency one. vLLM-Omni delivered much higher throughput but missed the strict playback-starvation SLO. Fleet counts of 140, 350, and 700 A10Gs are modelled estimates from that safe boundary with a 1.4× safety factor, not tested multi-replica deployments.

The proposed architecture (admission control, bounded queues, stream-aware routing, continuous batching) remains unvalidated, as do multi-replica balancing, autoscaling response time, replica failures, public-internet latency, and the best vLLM-Omni chunk/buffer configuration. The next experiment should sweep codec chunk size and client startup buffer until vLLM-Omni meets both TTFA and continuity targets, then rerun the load test across a real fleet.

## Generalising the method to LLMs

Audio starvation is specific to streaming media, but the capacity method is not:

1. Define user-facing latency and quality SLOs.
2. Test both fixed concurrency and independent arrivals.
3. Find the largest single-replica load that meets every SLO.
4. Add explicit headroom.
5. Validate routing, scaling, and failures across the real fleet.

For an LLM, replace audio starvation with inter-token latency or stalled-stream time. The same warning applies: successful requests and high token throughput do not guarantee a responsive application.

The objective is not to maximise a benchmark result. It is to find the highest load at which the system still meets the product's user-facing requirements.
