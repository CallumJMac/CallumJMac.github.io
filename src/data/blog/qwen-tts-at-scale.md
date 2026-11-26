---
pubDatetime: 2026-07-17T00:00:00Z
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

A fast demo is not necessarily scalable.

My [streaming Qwen3-TTS server](/posts/qwen-tts-streaming/) returns its first audio byte in about 195ms when serving one request at a time. With two concurrent requests, median time to first audio (TTFA) rises to **3.4 seconds**. I load-tested it against vLLM-Omni on one A10G: all **1,200 requests succeeded**, yet 19 of the 20 load conditions failed the latency or playback targets. For a voice agent, a successful response can still mean silence or stutter.

What the study found:

- **The serialised server is a single-server queue.** A first-in, first-out (FIFO) queue model built from one-at-a-time timings reproduces measured p50 and p95 latency within 1% at every open-loop rate.
- **vLLM-Omni stutters for a structural reason.** After its first 320ms packet it sends audio only in 2-second blocks, the first arriving about 530ms later, so the player runs dry in between.
- **A client buffer cannot buy capacity.** Each millisecond of buffer removes at most one millisecond of stutter and adds exactly one millisecond of startup delay. It rescues vLLM-Omni only at one stream, inside a 10ms window.
- **Under these SLOs, both backends safely serve one stream per A10G.** That is about **$9.60 per million characters** at full utilisation, beating the cheapest hosted APIs only above about 60% utilisation.

## Define "safe" first

A replica is safe at a given load only if:

- at least **99%** of requests succeed;
- p95 TTFA is at most **300ms**; and
- p95 **longest playback gap** is at most **100ms**.

The benchmark replays each request's chunk arrival times through a simulated player. Playback starts when the first chunk arrives, and each chunk plays once it has arrived and the previous one has finished. Any interval with nothing to play, excluding TTFA, is a gap. The SLO applies to the p95 across requests of each request's *longest single gap*.

The baseline player has no startup buffer. Later I replay every request with buffers from 0 to 500ms; buffering does not change the conclusion.

## The experiment

Both backends ran in turn on the same AWS `g5.xlarge` in `us-east-1`:

- NVIDIA A10G 24GB;
- Qwen3-TTS 0.6B CustomVoice, speaker `ryan`, fixed seed;
- a 20-prompt voice-agent corpus;
- 60 measured requests per condition, plus excluded warm-ups; and
- first packets of four codec frames (320ms of audio), varied in the packet-size study below.

The **serialised server** uses `faster-qwen3-tts` behind an application lock, which protects its non-thread-safe engine but lets only one request use the GPU at a time. **vLLM-Omni** (vLLM 0.24.0, vLLM-Omni 0.24.1) schedules several sequences together to raise GPU utilisation.

Two load patterns:

- **Closed loop:** keep 1, 2, 4, 8, 16 or 32 requests in flight.
- **Open loop:** Poisson arrivals at 0.25, 0.5, 1 or 2 requests per second (RPS), regardless of earlier requests.

The open-loop client caps in-flight requests at 32; the serialised server hit it at 1 and 2 RPS, so some requests waited up to 69 seconds in the client; at 0.25 and 0.5 RPS that wait never exceeded 34ms. Open-loop latency is measured from scheduled arrival, including that wait.

Empty, malformed, non-finite and non-24kHz audio counted as failures. No number here comes from an earlier run that repeated four prompts fifteen times.

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
    <strong>Figure 1.</strong> (a) p95 time to first audio and (b) p95 longest playback gap against closed-loop concurrency, 60 requests per point. Error bars are bootstrap 95% confidence intervals over requests. Dashed lines are the SLOs. Data: <code>capacity.csv</code>.
  </figcaption>
</figure>

Outputs are deterministic per prompt, so the intervals capture timing noise on one host, not variation across prompts or hosts. The backends fail in opposite ways:

| Backend           |           Load | p95 TTFA [95% CI]          | p95 longest gap | Audio generated / wall second | SLO-safe? |
| ----------------- | -------------: | -------------------------: | --------------: | ----------------------------: | --------- |
| Serialised server |  concurrency 1 | 197ms [196, 198]           |             0ms |                         2.19s | Yes       |
| Serialised server |  concurrency 2 | 4.98s [3.93, 8.49]         |             0ms |                         2.19s | No        |
| vLLM-Omni         |  concurrency 1 | 174ms [167, 222]           |           214ms |                         3.73s | No        |
| vLLM-Omni         | concurrency 16 | 4.85s [4.46, 5.22]         |           1.27s |                        12.66s | No        |

The serialised server never stutters, but it queues: throughput stays at **0.29 requests per second** at every load.

vLLM-Omni reached about **1.66 requests per second** and 12.66 seconds of audio per wall-clock second, 5.7× the request throughput, but stutters at every load, even one request. High throughput and good TTFA can coexist with broken playback.

## Why the serialised server queues

At concurrency one, the serialised server takes a mean **3.44s** per request (coefficient of variation 0.37), so its capacity is 1 / 3.44s ≈ **0.29 RPS**, the plateau in Figure 1.

At 0.25 RPS the server is **86% utilised**, and queueing theory predicts long waits even below capacity. The Pollaczek-Khinchine formula for an M/G/1 queue gives a steady-state mean wait of **12.1s**. The measured mean latency from arrival was 10.9s, reasonable for a three-minute, 60-request run from an empty queue, which at 86% utilisation typically does not reach steady state.

A stronger test avoids that assumption: I replayed the recorded arrival times through a single FIFO server, giving each request its prompt's mean service time and mean TTFA from the concurrency-one run:

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
    <strong>Figure 2.</strong> p95 time to first audio, measured from scheduled arrival, under Poisson load with bootstrap 95% CIs. The dashed line is a single-server FIFO model built from per-prompt mean service times and TTFA at concurrency one; it lies on the serialised server's measurements.
  </figcaption>
</figure>

| Offered RPS | Offered load | Measured p50 / p95 | FIFO model p50 / p95 |
| ----------: | -----------: | -----------------: | -------------------: |
|        0.25 |         86% |      12.1s / 20.3s |        12.1s / 20.3s |
|         0.5 |        172% |      51.7s / 95.6s |        51.4s / 95.3s |
|           1 |        344% |     74.9s / 141.5s |       74.7s / 141.4s |
|           2 |        688% |     86.2s / 164.8s |       86.0s / 164.4s |

Offered load is arrival rate times mean service time; above 100% the queue grows without bound. Nothing in the model is fitted to the open-loop data, yet it matches within 1% at every rate. The client cap does not matter: with 32 requests waiting the server never idles, so where a request waits is irrelevant. Above capacity the match is near arithmetic, since latency is accumulated work; the informative point is 0.25 RPS, where the queue empties and refills.

There is no hidden overhead: each replica is one server serving one stream at a time, with latency set by the queue in front of it.

## Why vLLM-Omni stutters

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

The serialised server sends a 320ms chunk every 144ms, over twice real time, so playback never stalls. vLLM-Omni sends a 320ms first packet, then **2-second blocks** about every 505ms. The first block arrives about **528ms** after the first packet, so the player runs dry for roughly 528 − 320 ≈ 210ms.

The cause is configuration: the client sends `initial_codec_chunk_frames: 4` with each request for a four-frame first packet, and the stage config sets `codec_chunk_frames: 25` for everything after. At 80ms per frame, 25 frames is 2 seconds of audio.

If so, the gap should equal the wait for the first block minus the audio already delivered, independent of first-packet size. The packet-size study varied the first packet from one to eight frames (Figure 4a):

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

For one, two, four and eight frames, the model predicts 448, 368, 208 and 0ms; the measured p95 gaps were 449, 368, 207 and 0ms. Eight frames (640ms of audio) is the smallest tested size that outlasts the 528ms wait, eliminating the gap for a single request.

Under sustained Poisson load at 0.25 RPS, eight-frame packets clearly missed TTFA (**376ms p95**) and marginally missed the gap target (**102ms p95**, 95% CI 33 to 127ms). Larger first packets shift the problem into TTFA.

### Can a client buffer fix it?

A startup buffer delays playback so audio accumulates. I replayed all 1,200 recorded requests through players with buffers from 0 to 500ms; at zero buffer the simulator reproduces the benchmark's recorded gaps to within 10⁻¹² seconds.

In Figure 4b each condition moves diagonally: every millisecond of buffer adds exactly one millisecond of p95 audible start and removes one millisecond of p95 gap, until the gap nears zero and the curve flattens. The only vLLM-Omni condition that ever meets both SLOs is concurrency one, with a buffer between **115 and 125ms** (5ms steps).

The stall is a fixed amount of missing audio; a buffer only moves it to the start, where it adds to startup delay. The fix has to come from the server: a smaller `codec_chunk_frames`, so the first full block arrives sooner. I have not measured what smaller blocks cost in throughput.

## Do the backends sound different?

I transcribed one retained clip per prompt for every packet-study condition with Whisper `small.en`. Intelligibility was equivalent; nearly every mismatch was number formatting ("1845" against "eighteen forty-five"); the rest were a misheard "CUDA" on both backends and one misheard word ("details") on the serialised server.

Both open every clip with **420 to 540ms of silence** before speech (median 420ms for the serialised server, 440ms for vLLM-Omni). That does not change the comparison, but it changes what a listener hears, as discussed in the [streaming post](/posts/qwen-tts-streaming/#what-the-listener-actually-waits-for).

## What does capacity cost?

One A10G serves one stream safely, and vLLM-Omni needs a precisely tuned client buffer to do it. Serving 100 simultaneous streams therefore needs at least **100 A10Gs** before headroom: about **$101 per hour** at the recorded on-demand price of $1.006 per `g5.xlarge`-hour (`us-east-1`, 16 July 2026). Targeting 60% fleet utilisation means dividing by 0.6, which gives **167** A10Gs for 100 streams.

These are arithmetic from one measured replica, not tested fleets, excluding routing, autoscaling delay and failure recovery. Utilisation means the fraction of time each GPU is streaming. High utilisation fits a 300ms TTFA only if a router sends each new request to an idle replica; a lone replica cannot: at 86% utilisation under random arrivals, its median wait was 12 seconds.

At its safe load, one serialised replica processes about **105,000 characters per hour**, or **$9.60 per million characters** if it never idles. Cost scales with 1 / utilisation:

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

Prices are public list prices, excluding free tiers, volume discounts and committed use. The comparison is compute only: no engineering time, redundancy or quality, and hosted voices and Qwen3-TTS are not equivalent products.

At this per-GPU capacity, self-hosting pays off only if you keep GPUs busy, and better per-replica scheduling would lower the curve faster than any pricing negotiation.

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

An upgraded WebSocket stays on one replica: the load balancer spreads new connections but cannot move an active stream. Scale-in therefore needs connection draining, and routing should consider active streams, not just instance count.

I would autoscale on active streams per ready replica, queue depth, oldest-request wait, p95 TTFA, playback gaps, and GPU utilisation and memory.

Requests per minute is a weak primary signal: prompts occupy the GPU for very different times, and Figure 2 shows how sharply latency rises near capacity.

Queues should be bounded. If the predicted wait already rules out a 300ms TTFA, reject immediately rather than accept work that will fail; for the serialised server, the queue model gives that prediction.

New replicas must load the model, capture CUDA graphs and warm up before becoming healthy, so scale-out is slow and the fleet needs warm headroom. Scale-in should be slower and wait for streams to drain.

## What is proved and what is not

**Measured on one A10G:** the capacity, queueing, stall and buffer results above. vLLM-Omni's stall follows from its 25-frame block size, and client buffering shifts it rather than removing it.

**Modelled, not tested:** fleet sizes, per-character costs and the effect of a smaller `codec_chunk_frames`.

**Not yet measured:** multi-replica balancing, autoscaling response, replica failure, public-internet latency, and trimming leading silence.

Next, I would sweep vLLM-Omni's `codec_chunk_frames` under sustained load for the smallest block size that keeps the playback gap under 100ms without collapsing throughput, then run the load test across a real fleet.

## The method generalises

Playback gaps are specific to streaming media, but the method is not:

1. Define user-facing latency and continuity SLOs.
2. Test both fixed concurrency and independent arrivals, and time latency from arrival.
3. Find the largest single-replica load that meets every SLO.
4. Check the result against a queueing model. If it fits, it tells you where the latency comes from.
5. Validate routing, scaling and failures across a real fleet.

For an LLM, use inter-token latency instead of playback gaps. Successful requests and high throughput still do not guarantee a responsive application.

## Reproduce it

Every number and figure here is regenerated from the retained per-request records by [`experiments/blog_analysis`](https://github.com/CallumJMac/qwen-tts-serve/tree/main/experiments/blog_analysis) in qwen-tts-serve: `analysis.py` writes `summary.json` and one CSV per figure, and `figures.py` renders the SVGs. Neither needs a GPU; the records are included. The capacity run is study `20260716T121717Z`, and the packet-size run is `20260722T091317Z`.
