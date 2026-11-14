# Mapping human-speech neurons in Qwen TTS (WIP)

Qwen3-TTS sometimes breathes, laughs or pauses on its own. These are the neurons behind those moments:
a set of 400 plus four small trait groups, 417 in all. Turn them up and the moments stop being accidents.

- **Breathing**: it breathes as the words are spoken.
- **Laughing while talking**: the laugh lands inside the words, not as a sound effect between them.
- **Whispering**: word endings soften into a whisper.
- **Natural pauses**: a real gap, then the sentence carries on naturally.
- **Someone else joins in**: on a podcast-style voice, a "yeah", a reply, or a question back from a second speaker.

61 of the 417 are dormant: they fire strongly in under 1% of ordinary speech frames (24 in under 0.1%).

## Model and layers

- Model: `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (voice cloning). The 400 sit in talker layers 20-27; the trait groups in talker layers 17-26.
- Each talker layer has an MLP with 3,072 neurons (`model.talker.model.layers.<L>.mlp`). A neuron is `L<layer>#<index>`:
  its activation is the input to `down_proj`, and its write is column `index` of `down_proj.weight`.
- Breathing on built-in voices uses `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, layers 20-27 (see below).

## How the edit works

**The 400: turned up ×8-12.** In each layer L (20-27) there is one fixed direction `d_L` (unit length, 1,024 values).
For each listed neuron k, the part of its write that points along `d_L` is multiplied by α:

```
g_k  = d_L · W_down[:, k]                    (original weights)
out' = out + (α - 1) · (Σ_k a_k · g_k) · d_L    over the listed neurons k of layer L
     = the same as  W_down[:, k] += (α - 1) · g_k · d_L   for each listed k
```

`a_k` is the neuron's activation, `out` the MLP output. α = 8 to 12 (strength guide below). Nothing else changes.
The directions `d_L` are not in this WIP release.

**Trait groups: nudged up or down.** Each trait neuron's activation is shifted by a fixed amount before `down_proj`:

```
a_k' = a_k + β · s_k · sd_k        s_k = +1 (up) or -1 (down),  sd_k = the neuron's usual spread (neurons.json)
```

β = 1-4 (we used 3 for single traits, 1-4 with all four traits on top of the 400 at ×10).

**Breathing on built-in voices (CustomVoice).** The same push applied to every neuron of layers 20-27 at α = 2:
`out' = out + (α - 1)(out · d_L) d_L`. On built-in voices the 400 at ×10 got stuck (no end of speech) in 13 of 36 tries,
so use the 400 on cloned voices.

## Behaviour → neurons

↑ = nudged up, ↓ = nudged down.

| behaviour | neurons | dormant |
|---|---|---|
| Breathing (built-in voices) | every neuron of L20-27, ×2 | - |
| Laughing while talking | L26#281 ↓, L24#837 ↑, L26#2805 ↓, L25#2430 ↑, L17#1652 ↑ | L26#2805 |
| Someone else joins in | L22#555 ↓, L23#1430 ↑, L22#142 ↓, L24#527 ↑, L21#175 ↑ | L22#555, L21#175 |
| Whispering | L26#673 ↓, L26#1854 ↑, L24#1485 ↓, L24#1501 ↑ | L24#1501 |
| Natural pauses | L21#1474 ↓, L20#1712 ↓, L23#1075 ↑ | L21#1474, L20#1712 |
| Laughing, whispering, pauses, someone joining (all of the above, milder) | the 400 alone at ×8-12 | 55 |

## The 400 by layer

| layer | neurons | dormant | dormant (strict) |
|---|---|---|---|
| L20 | 2 | 1 | 1 |
| L21 | 6 | 0 | 0 |
| L22 | 12 | 0 | 0 |
| L23 | 11 | 0 | 0 |
| L24 | 16 | 0 | 0 |
| L25 | 24 | 0 | 0 |
| L26 | 25 | 1 | 1 |
| L27 | 304 | 53 | 19 |
| **total** | 400 | 55 | 21 |

Full list with each neuron's strong % in [NEURONS.md](NEURONS.md) and [neurons.json](neurons.json).

## Strength guide (the 400)

| α | what you hear |
|---|---|
| ×8 | The effects come through and the voice stays itself. All the laugh, whisper, reply and question clips are ×8. |
| ×10 | Between the two. The level the trait groups were stacked on for longer clips. |
| ×12 | The human sounds come through more strongly. The top of the sweet spot. |
| ×16+ | More human sounds still, but ordinary speech breaks into gibberish. ×24 and ×32 are worse. |

Sweet spot: ×8-12. Results vary by voice and seed: generate a few and keep the best.

## Credit

Inspired by Lazarus (Papailiopoulos et al., [arXiv 2611.15118](https://arxiv.org/abs/2611.15118)), which reprogrammed dormant neurons.

Work in progress. Go find your own.
