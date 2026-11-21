---
pubDatetime: 2026-07-17T00:00:00Z
modDatetime: 2026-11-21T00:00:00Z
title: "Capacity Planning Qwen3-TTS on an A10G"
slug: qwen-tts-at-scale
featured: true
description: "A single-GPU capacity study for streaming Qwen3-TTS: why one backend queues, why the other stutters, what a client buffer can and cannot fix, and what it costs per character."
tags:
  - TTS
  - Qwen
  - vLLM
  - AWS
  - Load Testing
  - MLOps
---

A fast demo is not necessarily a scalable service.

My [streaming Qwen3-TTS server](/posts/qwen-tts-streaming/) returns its first audio byte in about 195ms when it serves one request at a time. With two concurrent requests, median time to first audio (TTFA) rises to **3.4 seconds**. I load-tested it against vLLM-Omni on one A10G. All **1,200 requests succeeded**, yet most conditions still failed the latency or playback targets. For a voice agent, a successful response can still mean silence or stutter.

What the study found:

- **The serialised server is a single-server queue, and queueing theory predicts it exactly.** A first-in, first-out queue model built only from its one-at-a-time service times reproduces measured p95 latency within 1% at every open-loop rate.
- **vLLM-Omni stutters for a structural reason.** After its first 320ms packet, it sends audio only in 2-second blocks, and the first block takes about 530ms to arrive. The player runs dry in between.
- **A client buffer cannot buy capacity.** Each millisecond of buffer removes one millisecond of stutter and adds one millisecond of startup delay. It rescues vLLM-Omni only at one stream, inside a 10ms window.
- **Under these SLOs, both backends safely serve one stream per A10G.** That works out to about **$9.60 per million characters** at full utilisation. It beats the cheapest hosted APIs only above about 60% utilisation.

## Define "safe" first

Capacity depends on the experience you accept. A replica counts as safe at a given load only if:

- at least **99%** of requests succeed;
- p95 TTFA is at most **300ms**; and
- p95 **longest playback gap** is at most **100ms**.

The playback gap needs a precise definition. The benchmark replays each request's chunk arrival times through a simulated player. Playback starts when the first audio byte arrives, and each chunk plays as soon as it has arrived and the previous chunk has finished. Any interval with nothing to play is a gap. The metric is the *longest single gap* in each request, and the SLO applies to its 95th percentile across requests. TTFA itself does not count as a gap.

The baseline player has no startup buffer. That is strict, so I later replay every request with buffers from 0 to 500ms to see whether buffering changes the conclusion. It does not.

## The experiment

Both backends ran one after the other on the same AWS `g5.xlarge` in `us-east-1`:

- NVIDIA A10G 24GB;
- Qwen3-TTS 0.6B CustomVoice, speaker `ryan`, fixed seed;
- a 20-prompt voice-agent corpus;
- 60 measured requests per condition, plus excluded warm-ups; and
- first packets of four codec frames (320ms of audio).

The **serialised server** uses `faster-qwen3-tts` with an application lock, which protects its non-thread-safe engine but lets only one request use the GPU at a time. **vLLM-Omni** (vLLM 0.24.0, vLLM-Omni 0.24.1) schedules several sequences together to raise GPU utilisation.

There were two load patterns:

- **Closed loop:** keep 1, 2, 4, 8, 16 or 32 requests in flight.
- **Open loop:** start requests as a Poisson process at 0.25, 0.5, 1 or 2 requests per second (RPS), regardless of whether earlier requests have finished.

The open-loop client caps in-flight requests at 32. The serialised server reached that cap at 1 and 2 RPS, so some requests waited up to 69 seconds inside the client before being sent. Every open-loop latency below is measured from the scheduled arrival, including that client-side wait.

The benchmark rejected empty, malformed, non-finite and non-24kHz audio rather than counting them as successes. An earlier run used four prompts repeated fifteen times instead of the 20-prompt corpus. I kept it for audit but superseded it, and no number here comes from it.

## The result: neither backend is safe beyond one stream

<figure>
  <img
    src="/assets/qwen-tts/capacity-closed-loop.svg"
    alt="Two panels against closed-loop concurrency. Panel a: p95 time to first audio on a log scale with a 300 ms SLO line. Panel b: p95 longest playback gap with a 100 ms SLO line."
    width="595"
    height="276"
    loading="lazy"
    decoding="async"
  />
  <figcaption>
    <strong>Figure 1.</strong> (a) p95 time to first audio and (b) p95 longest playback gap against closed-loop concurrency, 60 requests per point. Error bars are bootstrap 95% confidence intervals. Dashed lines are the SLOs. Data: <code>capacity.csv</code>.
  </figcaption>
</figure>

The two backends fail in opposite ways:

| Backend           |           Load | p95 TTFA [95% CI]          | p95 longest gap | Audio generated / wall second | SLO-safe? |
| ----------------- | -------------: | -------------------------: | --------------: | ----------------------------: | --------- |
| Serialised server |  concurrency 1 | 197ms [196, 198]           |             0ms |                         2.19s | Yes       |
| Serialised server |  concurrency 2 | 4.98s [3.93, 8.49]         |             0ms |                         2.19s | No        |
| vLLM-Omni         |  concurrency 1 | 174ms [167, 222]           |           214ms |                         3.73s | No        |
| vLLM-Omni         | concurrency 16 | 4.85s [4.46, 5.22]         |           1.27s |                        12.66s | No        |

The serialised server never stutters, but it queues. Throughput stays at **0.29 requests per second** at every load, so extra requests just wait.

vLLM-Omni reached about **1.66 requests per second** and 12.66 seconds of audio per wall-clock second: 5.7× the request throughput. But it stutters at every load, including a single request.

High throughput and good TTFA can coexist with broken playback. The next two sections explain each failure.

## Why the serialised server queues

At concurrency one, the serialised server takes a mean **3.44s** per request (coefficient of variation 0.37). Its capacity is therefore 1 / 3.44s ≈ **0.29 RPS**, exactly the plateau in Figure 1.

At 0.25 RPS, the lowest open-loop rate, the server is **86% utilised**. Queueing theory predicts long waits at that utilisation even though the average arrival rate is below capacity. The Pollaczek-Khinchine formula for an M/G/1 queue gives a steady-state mean wait of **12.1s**.

I took a stronger test than the formula. I replayed the exact recorded arrival times through a single first-in, first-out server, using each prompt's mean service time from the concurrency-one run:

<figure>
  <img
    src="/assets/qwen-tts/open-loop-queue.svg"
    alt="p95 time to first audio from arrival against Poisson offered load. Serialised server measurements lie on a FIFO queue model line; vLLM-Omni is lower but above the SLO except at 0.25 RPS."
    width="419"
    height="280"
    loading="lazy"
    decoding="async"
  />
  <figcaption>
    <strong>Figure 2.</strong> p95 time to first audio, measured from scheduled arrival, under Poisson load with bootstrap 95% CIs. The dashed line is a single-server FIFO model built only from concurrency-one service times; it lies on the serialised server's measurements.
  </figcaption>
</figure>

| Offered RPS | Utilisation | Measured p50 / p95 | FIFO model p50 / p95 |
| ----------: | ----------: | -----------------: | -------------------: |
|        0.25 |         86% |      12.1s / 20.3s |        12.1s / 20.3s |
|         0.5 |        172% |      51.7s / 95.6s |        51.4s / 95.3s |
|           1 |        344% |     74.9s / 141.5s |       74.7s / 141.4s |
|           2 |        688% |     86.2s / 164.8s |       86.0s / 164.4s |

The model has no fitted parameters and matches within 1% at every rate. Nothing else is going on: the serialised server is one queue with one server. That also means no routing trick will save it. Each replica serves one stream at a time, and its latency is set by the queue in front of it.

## Why vLLM-Omni stutters

Figure 3 shows one request from each backend, for the same prompt, at concurrency one.

<figure>
  <img
    src="/assets/qwen-tts/playback-timeline.svg"
    alt="Cumulative audio received and played against time for one request per backend. The serialised server delivers small chunks steadily; vLLM-Omni delivers a small first chunk then two-second blocks, and the player stalls for about 0.2 s."
    width="567"
    height="276"
    loading="lazy"
    decoding="async"
  />
  <figcaption>
    <strong>Figure 3.</strong> Audio received (coloured) and audio played (neutral line) for the "assistant" prompt at concurrency one. The shaded interval is a playback stall.
  </figcaption>
</figure>

The serialised server sends a 320ms chunk every 144ms, more than twice real time, so the player never catches up. vLLM-Omni sends a 320ms first packet, then **2-second blocks** about every 505ms. The first block takes about **528ms** to arrive, so the player runs dry for roughly 528 − 320 ≈ 210ms.

The cause is in the configuration. The stage config sets `initial_codec_chunk_frames: 4` for the first packet and `codec_chunk_frames: 25` for everything after. Twenty-five frames is 2 seconds of audio.

If that explanation is right, the gap should equal the wait for the first block minus the audio already delivered. The packet-size study varied the first packet from one to eight frames, so I could test it (Figure 4a):

<figure>
  <img
    src="/assets/qwen-tts/omni-gap-mechanism.svg"
    alt="Panel a: vLLM-Omni p95 playback gap against frames in the first packet, following a simple subtraction model. Panel b: buffer sweep curves moving diagonally, with only concurrency 1 touching the SLO region."
    width="580"
    height="277"
    loading="lazy"
    decoding="async"
  />
  <figcaption>
    <strong>Figure 4.</strong> (a) vLLM-Omni p95 longest gap against first-packet size, one request at a time, with the model max(0, 528ms − 80ms × <em>k</em>) (dashed). (b) Every recorded request replayed with startup buffers from 0 to 400ms; markers are every 100ms of buffer. The shaded region meets both SLOs. Data: <code>packet-study.csv</code>, <code>buffer-sweep.csv</code>.
  </figcaption>
</figure>

The measured p95 gaps of 449, 368, 207 and 0ms follow the model closely. An eight-frame first packet (640ms of audio) is the first size that outlasts the 528ms wait, and it eliminates the gap for a single request.

Under sustained load, though, eight-frame packets still missed both targets: **376ms p95 TTFA** and **102ms p95 gap** at 0.25 RPS. Larger first packets shift the problem into TTFA.

### Can a client buffer fix it?

A startup buffer delays playback so more audio accumulates first. I replayed all 1,200 recorded requests through players with buffers from 0 to 500ms. The simulator reproduces the benchmark's recorded gaps to within 10⁻¹² seconds at zero buffer.

Figure 4b shows the result. Each condition moves diagonally: every millisecond of buffer removes one millisecond of p95 gap and adds one millisecond of p95 audible start. Only vLLM-Omni at concurrency one ever enters the SLO box, and only with a buffer between **115 and 125ms**. No other vLLM-Omni condition meets both SLOs at any buffer size.

The stall is a fixed amount of missing audio, and a buffer only moves it from the middle of playback to the start. The fix has to come from the server: a smaller `codec_chunk_frames`, so the first full block arrives sooner. That is the next experiment. The model above predicts the gap, but I have not measured the cost of smaller blocks to throughput.

## Do the backends sound different?

I transcribed one retained clip per prompt for every packet-study condition with Whisper `small.en`. Intelligibility was equivalent: nearly every transcript mismatch was number formatting ("1845" against "eighteen forty-five"). The rest were a misheard "CUDA" on both backends and one misheard word ("details") on the serialised server.

Both backends also open every clip with about **420ms of silence** before speech. That adds the same amount to each, so it does not change the comparison. It does change what a listener experiences, as discussed in the [streaming post](/posts/qwen-tts-streaming/#what-the-listener-actually-waits-for).

## What does capacity cost?

Under these SLOs, both backends serve one stream safely per A10G. For vLLM-Omni, that needs a precisely tuned client buffer.

Serving 100 simultaneous streams therefore needs at least **100 A10Gs** before any headroom: about **$101 per hour** at the recorded on-demand price of $1.006 per `g5.xlarge`-hour (`us-east-1`, 16 July 2026). Headroom has to be defined explicitly. Targeting 60% fleet utilisation means dividing by 0.6, which gives **167** A10Gs for 100 streams.

These are arithmetic estimates from one measured replica, not tested fleets. They exclude routing, autoscaling delay and failure recovery.

Per-character cost compares better with hosted APIs. At its safe load, one serialised replica processes about **105,000 characters per hour**, or **$9.60 per million characters** if it never sits idle. Real fleets idle, so cost scales with 1 / utilisation:

<figure>
  <img
    src="/assets/qwen-tts/cost-per-character.svg"
    alt="Self-hosted cost per million characters against utilisation, falling to 9.60 dollars at 100 percent, with hosted API list prices from 15 to 40 dollars as horizontal lines."
    width="414"
    height="280"
    loading="lazy"
    decoding="async"
  />
  <figcaption>
    <strong>Figure 5.</strong> Self-hosted cost per million characters at the SLO-safe load against GPU utilisation. Horizontal lines are hosted list prices, accessed 21 November 2026; open circles mark where the serialised server breaks even.
  </figcaption>
</figure>

| Hosted API (list price per 1M characters) | Price | Serialised replica breaks even above |
| ----------------------------------------- | ----: | -----------------------------------: |
| OpenAI `tts-1`                            |   $15 |                       64% utilisation |
| Amazon Polly Neural                       |   $16 |                       60% utilisation |
| Deepgram Aura-2, Google Chirp 3 HD        |   $30 |                       32% utilisation |
| ElevenLabs Flash                          |   $40 |                       24% utilisation |

Prices come from each provider's public pricing page and exclude free tiers, volume discounts and committed-use pricing. The comparison covers compute only: no engineering time, no redundancy, and no quality comparison. Hosted voices and Qwen3-TTS are not equivalent products.

The useful conclusion is directional. At this per-GPU capacity, self-hosting only pays off when you can keep GPUs busy. Improving per-replica scheduling would move the curve down faster than any pricing negotiation.

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
GPU replicas ── streaming scheduler with small, steady codec blocks
   │
Metrics: active streams, queue wait, TTFA, playback gaps, GPU utilisation
```

An upgraded WebSocket stays attached to one replica. The load balancer can spread new connections, but it cannot move an active stream off a busy replica. Scale-in therefore needs connection draining, and routing should consider active streams, not just instance count.

I would autoscale on active streams per ready replica, queue depth, oldest-request wait, p95 TTFA, playback-gap telemetry, and GPU utilisation and memory.

Requests per minute is a weak primary signal. Two prompts can occupy the GPU for very different times, and Figure 2 shows how sharply latency rises as utilisation approaches one.

Queues should be bounded. If the predicted wait already makes a 300ms TTFA impossible, an immediate overload response beats accepting work that will fail its SLO. For the serialised server, the queue model above gives that prediction directly.

New replicas must load the model, capture CUDA graphs and warm up before becoming healthy, so reactive scale-out is slow and the fleet needs warm headroom. Scale-in should be slower and should wait for streams to drain.

## What is proved and what is not

**Measured on one A10G:** all 1,200 requests succeeded. The serialised server met every SLO only at concurrency one, and a FIFO queue model explains its open-loop latency within 1%. vLLM-Omni delivered 5.7× the throughput but stalled after its first packet. That stall follows from its 25-frame block size, and client buffering shifts it rather than removing it.

**Modelled, not tested:** fleet sizes, per-character costs and the effect of a smaller `codec_chunk_frames`.

**Not yet measured:** multi-replica balancing, autoscaling response, replica failure, public-internet latency, and trimming leading silence.

The next experiment is a sweep of vLLM-Omni's `codec_chunk_frames` under sustained load, looking for the smallest block size that keeps the playback gap under 100ms without collapsing throughput. Then the load test should run across a real fleet.

## The method generalises

Audio playback gaps are specific to streaming media, but the method is not:

1. Define user-facing latency and continuity SLOs.
2. Test both fixed concurrency and independent arrivals, and time latency from arrival.
3. Find the largest single-replica load that meets every SLO.
4. Check the result against a queueing model. If the model fits, it tells you exactly where the latency comes from.
5. Validate routing, scaling and failures across a real fleet.

For an LLM, replace the playback gap with inter-token latency. The warning still applies: successful requests and high throughput do not guarantee a responsive application.

## Reproduce it

Every number and figure in this post comes from the retained per-request records, regenerated by an analysis package: `analysis.py` writes `summary.json` and one CSV per figure, and `figures.py` renders the SVGs. Neither needs a GPU. I will publish both, with the records, in [qwen-tts-serve](https://github.com/CallumJMac/qwen-tts-serve). The capacity run is study `20260716T121717Z`, and the packet-size run is `20260722T091317Z`.
