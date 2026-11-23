---
pubDatetime: 2026-11-21T00:00:00Z
modDatetime: 2026-11-22T00:00:00Z
title: "Making My Coding Agents Talk Back"
slug: orca-talkback
featured: true
ogImage: "https://callumjmac.github.io/assets/orca-talkback/cover.png"
description: "orca-talkback gives Claude Code agents in Orca a voice: a spoken summary of every reply within a second, local text-to-speech, and barge-in so you can talk over it."
tags:
  - TTS
  - Voice Agents
  - Claude Code
  - Orca
  - Kokoro
---

I run my coding agents side by side in [Orca](https://github.com/stablyai/orca), a desktop app for running agents such as Claude Code in parallel, and I talk to them with the dictation app Wispr Flow. Talking *to* an agent is solved. I wanted it to talk *back*: tell me what it did and what it needs from me, out loud, like voice mode in the Claude apps.

So I built **orca-talkback**. When an agent in the workspace I am looking at finishes, I hear one to three conversational sentences about its reply, spoken locally on my Mac, starting **about 0.8 seconds** after it finishes. If I start talking, it stops.

It is two small open-source pieces: a [Python daemon](https://github.com/CallumJMac/orca-talkback) and an [Orca plugin](https://github.com/CallumJMac/orca-talkback-plugin).

<figure>
  <img
    src="/assets/orca-talkback/how-it-works.svg"
    alt="Architecture diagram. You dictate to a Claude Code agent in Orca. A Claude Code hook asks the orca-talkback daemon whether Talkback is on and, if so, adds an instruction. The Talkback plugin forwards focus and agent status to the daemon, which has Kokoro speak the reply's spoken line. Echo cancellation and voice detection on the microphone let you stop it by talking."
    width="840"
    height="566"
    loading="lazy"
    decoding="async"
  />
  <figcaption>
    <strong>Figure 1.</strong> The plugin tells the daemon what is on screen and when agents finish. The daemon decides what to say, Kokoro says it, and the microphone listens for you while it does.
  </figcaption>
</figure>

## The agent writes for your ears

Agent replies are written to be read: tables, file paths and commit hashes that sound like a recited changelog. So when Talkback is on for a session, a Claude Code `UserPromptSubmit` hook adds one hidden instruction to each prompt: end the reply with a spoken version, written as a markdown comment.

```text
[//]: # (spoken: I fixed the barge-in bug and pushed it. Want me to tag a release?)
```

Orca's chat view hides link reference definitions like this, so the reply looks unchanged, and the daemon speaks the line. The agent that did the work writes it, so no second model has to summarise anything.

Claude Code's terminal would print the line, so terminal agents are not asked for one. For them, or when an agent forgets, orca-talkback speaks a cleaned-up reply instead, with code, tables and links removed, cut at a sentence boundary.

The Talkback button in Orca's status bar toggles it per session: muted, the hook adds nothing and nothing is spoken. Each Orca chat is its own workspace, so muting one chat mutes only that chat.

<figure>
  <img
    src="/assets/orca-talkback/time-to-first-word.svg"
    alt="Timeline from the agent finishing to the first word: reading the reply from Orca takes 370 milliseconds, synthesising the first sentence 320 milliseconds and the speaker buffer 120 milliseconds, so the first word starts at 810 milliseconds."
    width="539"
    height="155"
    loading="lazy"
    decoding="async"
  />
  <figcaption>
    <strong>Figure 2.</strong> Median of five runs on an M4 Max. Kokoro renders a whole chunk before any of it plays, so the first sentence is synthesised alone and the rest follow while it plays. A long opener therefore delays the first word, which is why the hook asks for one under ten words: a 23-word opener took 1.2 seconds to synthesise instead of 0.3.
  </figcaption>
</figure>

## Talking over it

Barge-in is the hard part: the microphone hears the speakers, so a naive voice detector would stop the agent with its own voice.

orca-talkback feeds everything it plays to WebRTC's AEC3 echo canceller as the reference, so the agent's voice is removed from the microphone signal before Silero VAD looks for speech. The rule that makes this work: the reference must be exactly the audio sent to the speaker, silences included, on one continuous timeline, because AEC3 tracks a steady delay, not one that jumps. So the pause while the next chunk is synthesised is real silence written to both. Padding the reference later from microphone timing would mistake audio still in the output buffer for silence, putting every later sentence ahead of its reference. One canceller lasts the whole reply, so what it learns about the room carries over.

<figure>
  <img
    src="/assets/orca-talkback/echo-cancellation.svg"
    alt="Line chart over a 12 second reply. The microphone hears the agent's voice at a median of minus 37 dBFS; after echo cancellation the median is minus 67 dBFS, with brief jumps at speech onsets."
    width="594"
    height="266"
    loading="lazy"
    decoding="async"
  />
  <figcaption>
    <strong>Figure 3.</strong> The agent's own voice at the microphone and after echo cancellation (with WebRTC's noise suppression), for a four-sentence reply through MacBook speakers, measured every 200 ms. Dashed lines are medians; point by point, cancellation makes it a median of 26.5 dB quieter. The jumps are residual echo at speech onsets while the canceller catches up, which is why barge-in ignores the first half-second of each synthesised chunk.
  </figcaption>
</figure>

Your voice is not in the reference, so it survives cancellation, and about 160 ms of it stops playback with a short fade. In six test replies through MacBook speakers, five never came close to interrupting themselves; the sixth had one quarter-second burst louder than anything the speakers played, most likely room noise. A second voice talking over the reply stopped it in both tests I ran.

## Only when you are looking

A voice that talks while you read something else is worse than none, so orca-talkback only speaks when:

1. **An Orca window is visible.** It checks twice a second: Orca on a second screen counts while you type in another app, but Orca minimised, hidden or buried under your browser cuts it off.
2. **The agent is on screen.** That is the workspace picked in Orca's sidebar, or the chat you just prompted. Orca does not tell plugins which chat is visible, so prompting a chat is the most reliable sign you are looking at it.

Other replies are never spoken, or replayed later.

## Choosing a voice

To pick a voice, I compared the open models on Hugging Face's [Open TTS Leaderboard](https://huggingface.co/spaces/hf-audio/open_tts_leaderboard):

| Model | Naturalness (UTMOS) | Word error rate | Size | License |
|---|---|---|---|---|
| **[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)** | **4.50** | 0.76% | 82M | Apache-2.0 |
| [Inflect-Mini-v3](https://huggingface.co/owensong/Inflect-Mini-v3) | 4.48 | 0.76% | 34M | Apache-2.0 |
| [Reverb-TTS](https://huggingface.co/software-mansion/Reverb-TTS) | 4.45 | **0.65%** | 297M | CC-BY-NC-4.0 |
| [Chatterbox](https://huggingface.co/ResembleAI/chatterbox) | 4.40 | 1.00% | 800M | MIT |

In November 2026, Kokoro had the highest UTMOS of the 56 models on the English Seed-TTS set and a word error rate 0.11 percentage points behind the best. UTMOS is a predicted naturalness score, not a human rating, so its 0.02 lead over Inflect-Mini-v3 is a tie; what decided it was being near the top on both while small and Apache-2.0. It also had the lowest word error rate (0.96%) of the 33 models on [VoiceHub's Seed-TTS-Eval](https://huggingface.co/datasets/kadirnar/voicehub-arena-seed-tts-eval).

## Try it

You need macOS on Apple Silicon, Orca 1.4.248 or newer, and [uv](https://docs.astral.sh/uv/).

```bash
uv tool install --python 3.14 orca-talkback
orca-talkback setup   # adds the Claude Code hook, downloads the voice (about 340 MB), checks your mic
orca plugin install CallumJMac/orca-talkback-plugin
# Orca prints a "Fingerprint: ap1-..." line; re-run the command with --approve and that fingerprint
orca-talkback
```

You can also install the plugin from Orca's Settings, Plugins. Then ask an agent something in an Orca chat. No Homebrew, no API keys: after the one-off voice download everything runs on your Mac, the microphone is only open while it speaks, and nothing is recorded.

It is early, tested with Claude Code agents on an M4 Max, and I would love to hear how it works for you. Issues and pull requests are welcome on [GitHub](https://github.com/CallumJMac/orca-talkback).
