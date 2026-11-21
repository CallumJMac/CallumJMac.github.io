---
pubDatetime: 2026-11-21T00:00:00Z
title: "Making My Coding Agents Talk Back"
slug: orca-talkback
featured: true
description: "orca-talkback gives Claude Code agents in Orca a voice: a spoken summary of every reply within a second, local text-to-speech, and barge-in so you can talk over it."
tags:
  - TTS
  - Voice Agents
  - Claude Code
  - Orca
  - Kokoro
---

I run my coding agents side by side in [Orca](https://github.com/stablyai/orca), and I talk to them with Wispr Flow. Talking *to* an agent is solved. What I wanted was for it to talk *back*: when it finishes, tell me what it did and what it needs from me, out loud, the way voice mode works in the Claude apps.

So I built **orca-talkback**. When an agent in the workspace I am looking at finishes, I hear one to three conversational sentences about its reply, spoken locally on my Mac, starting **about 0.8 seconds** after it finishes. If I start talking, it stops.

It is two small open-source pieces: a [Python daemon](https://github.com/CallumJMac/orca-talkback) and an [Orca plugin](https://github.com/CallumJMac/orca-talkback-plugin).

<figure>
  <img
    src="/assets/orca-talkback/how-it-works.svg"
    alt="Architecture diagram. You dictate to a Claude Code agent in Orca. A Claude Code hook asks the orca-talkback daemon whether Talkback is on and, if so, adds an instruction. The Talkback plugin forwards focus and agent status to the daemon, which has Kokoro speak the reply's spoken line. Echo cancellation and voice detection on the microphone let you stop it by talking."
    width="840"
    height="630"
    loading="lazy"
    decoding="async"
  />
  <figcaption>
    The plugin tells the daemon what is on screen and when agents finish. The daemon decides what to say, Kokoro says it, and the microphone listens for you while it does.
  </figcaption>
</figure>

## The agent writes for your ears

Agent replies are written to be read. Mine are full of tables, file paths and commit hashes, and read aloud they sound like someone reciting a changelog. So orca-talkback asks the agent that did the work to say it differently. When Talkback is on for a session, a Claude Code `UserPromptSubmit` hook adds one hidden instruction to each prompt: end the reply with a spoken version of it, written as a markdown comment.

```text
[//]: # (spoken: I fixed the barge-in bug and pushed it. Want me to tag a release?)
```

Orca's chat view hides link reference definitions like this one, so the reply looks exactly the same. The daemon finds the line and speaks it. The agent knows what it meant, so the spoken version says exactly that, and nothing sits between the reply and the voice.

The Talkback button in Orca's status bar picks the mode per session. On, the agent writes for speaking and reading at once; muted, it writes for reading only and the hook adds nothing. Each Orca chat is its own workspace, so muting a chat mutes just that chat.

<figure>
  <img
    src="/assets/orca-talkback/time-to-first-word.svg"
    alt="Stacked bar: reading the finished reply from Orca takes 370 milliseconds, synthesising the first sentence 320 milliseconds, and the speaker buffer 120 milliseconds, about 0.8 seconds in total."
    width="840"
    height="300"
    loading="lazy"
    decoding="async"
  />
  <figcaption>
    From the agent finishing to the first word, median of five runs on an M4 Max. Synthesis time grows with sentence length, so the hook asks for a short first sentence: a 23-word opener took 1.2 seconds instead of 0.3.
  </figcaption>
</figure>

Claude Code's terminal would print the line, so agents running in a terminal are not asked for one. For them, or when an agent forgets, orca-talkback speaks a cleaned-up version of the reply instead, with code, tables and links removed.

## Talking over it

Barge-in is the hard part. The microphone hears the speakers, so a naive voice detector hears the agent's own voice and stops it mid-sentence.

orca-talkback feeds every sample it plays to WebRTC's AEC3 echo canceller as the reference, so the canceller can subtract the agent's voice from the microphone signal before Silero VAD looks for speech. The one rule that makes this work is that the reference must be exactly what the speaker played, sample for sample. Even the short pause while the next sentence is synthesised is filled with real silence written to both the speaker and the reference, so the two never drift apart.

<figure>
  <img
    src="/assets/orca-talkback/echo-cancellation.svg"
    alt="Line chart over a 12 second reply. The microphone hears the agent's voice around minus 33 decibels; after echo cancellation the same signal sits around minus 66 decibels, a median 26 decibels quieter, with brief jumps at the start of each sentence."
    width="840"
    height="470"
    loading="lazy"
    decoding="async"
  />
  <figcaption>
    The agent's own voice, as the microphone hears it and after echo cancellation. The brief jumps at the start of each sentence are the canceller adapting, which is why barge-in ignores the first half-second of every sentence.
  </figcaption>
</figure>

Your voice is not in the reference, so it passes straight through: about 160 ms of it stops playback with a short fade. In six test replies through MacBook speakers, five never came close to interrupting themselves; the sixth had one quarter-second burst louder than anything the speakers played, most likely room noise. A second voice talking over the reply stopped it in both of the tests I ran.

## Only when you are looking

A voice that talks while you are reading something else is worse than no voice, so orca-talkback only speaks when two things are true:

1. **Orca is the app in front.** It checks twice a second, so switching to your browser cuts it off.
2. **The agent is on screen.** That means the workspace you picked in Orca's sidebar, or the chat you just sent a prompt to. Orca does not tell plugins which chat is visible, and its focus signal follows the sidebar, so sending a chat a prompt is the most reliable sign you are looking at it.

Replies from anywhere else are never spoken, and never replayed later.

## Choosing a voice

Before settling on a text-to-speech model, I compared the open models on Hugging Face's [Open TTS Leaderboard](https://huggingface.co/spaces/hf-audio/open_tts_leaderboard):

| Model | Naturalness (UTMOS) | Word error rate | Size | License |
|---|---|---|---|---|
| **[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)** | **4.50** | 0.76% | 82M | Apache-2.0 |
| [Inflect-Mini-v3](https://huggingface.co/owensong/Inflect-Mini-v3) | 4.48 | 0.76% | 34M | Apache-2.0 |
| [Reverb-TTS](https://huggingface.co/software-mansion/Reverb-TTS) | 4.45 | **0.65%** | 297M | CC-BY-NC-4.0 |
| [Chatterbox](https://huggingface.co/ResembleAI/chatterbox) | 4.40 | 1.00% | 800M | MIT |

Kokoro had the highest naturalness of the 56 English models, an error rate within 0.11 points of the best, and it is small and Apache-2.0. It also had the lowest error rate of 33 models on [VoiceHub's Seed-TTS-Eval](https://huggingface.co/datasets/kadirnar/voicehub-arena-seed-tts-eval).

## Try it

You need macOS on Apple Silicon, Orca 1.4.248 or newer, and [uv](https://docs.astral.sh/uv/).

```bash
uv tool install git+https://github.com/CallumJMac/orca-talkback
orca-talkback setup   # adds the Claude Code hook, downloads the voice, checks your mic
# Orca prints a fingerprint for the plugin's access; re-run with --approve <fingerprint>
orca plugin install CallumJMac/orca-talkback-plugin
orca-talkback
```

Then ask an agent something. There is no Homebrew step and there are no API keys: the voice model is downloaded once, then everything runs on your Mac, the microphone is only open while it speaks, and nothing is recorded.

It is early, tested with Claude Code agents on an M4 Max, and I would love to hear how it works for you. Issues and pull requests are welcome on [GitHub](https://github.com/CallumJMac/orca-talkback).
