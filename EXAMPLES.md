# EXAMPLES — Worked custom-wake-word projects

Three reference projects that exercise the toolkit end-to-end. Use them as templates for your own.

---

## Tofu — robotic toy companion

Files: [`configs/examples/tofu/`](configs/examples/tofu/)

```bash
# Already initialised. Reproduce it for free:
python scripts/init_wake.py --name tofu \
    --phrases "hey tofu,hi tofu,hello tofu,okay tofu" \
    --force
```

**Trigger phrases**: "hey tofu", "hi tofu", "hello tofu", "okay tofu"

**Notable hard-negatives**:
- Rhyme: "doofus", "to-do list", "to-go", "tutu", "two pugs"
- Food context: "tofu burger", "I love tofu", "fried tofu"
- Greeting collisions: "hey there", "hi buddy"

**v0 targets** (see [`PLAN.md`](PLAN.md)):
- FRR ≤ 5%, FAR ≤ 1.0/hour, .tflite ≤ 100 kB, ESP32-S3 inference ≤ 10 ms
- ~$1 of RunPod compute, ~4 days wall time

**Status**: pipeline scaffolded, training not yet run.

---

## Sunny — voice-controlled smart-home greeter

```bash
python scripts/init_wake.py --name sunny \
    --phrases "hey sunny,hi sunny,hello sunny" \
    --counts "6000,3000,2000"

# Auto-suggest hard negatives:
python scripts/suggest_hard_negatives.py --name sunny
```

**Notable collisions** (auto-generated):
- "honey", "sunday", "funny", "money"
- "Sonny" (the name)
- "hey honey" (very high collision rate from couples in voice range)

**Why this example**: shows how a 2-syllable wake word with common rhymes ("honey", "money", "funny") forces you to lean harder on hard-negatives. Expect to need ~3.5k hard-negs (vs. 2.5k for Tofu) and possibly tighten the probability threshold.

---

## Jarvis — virtual butler (classic)

```bash
python scripts/init_wake.py --name jarvis \
    --phrases "jarvis,hey jarvis,okay jarvis" \
    --weights "1.2,1.0,0.8"

python scripts/suggest_hard_negatives.py --name jarvis
```

**Notable collisions**:
- "service", "garbage", "Harris"
- "Travis", "Davis", "tennis"
- News/podcast audio mentioning the name "Jarvis"

**Why this example**: matches the [openWakeWord "Hey Jarvis"](https://github.com/dscripka/openWakeWord/blob/main/docs/models/hey_jarvis.md) reference recipe — useful for **direct benchmark** against their production model. We expect to under-perform their FAR (they used 200k positives + 31k h negatives vs. our 10k + 300 h) but it's a known operating point to calibrate against.

---

## Choosing your own wake word — design rules

Before you `init_wake.py`, check the phrase against these heuristics:

| Rule | Why | Example |
|---|---|---|
| ≥ 2 syllables | 1-syllable wakes have ~10× higher FAR | "hey Joe" → bad; "hey jarvis" → good |
| Distinct phonemes | Avoid words that rhyme with common English | "hi vee" → collides with "TV"; "hey bolt" → cleaner |
| Stressed first syllable | Easier for streaming detector | "tofu" (TÓ-fu) → ok; "tofu" said as "to-FU" → harder |
| Avoid common nouns | "tofu" forces you to handle food contexts | "vermillion" → easier, but unmemorable |
| Avoid names of public figures | News/podcast audio will misfire | "trump", "biden" → never |
| 2-3 syllables ideal | Long enough to be distinctive, short enough to be fast | "hey assistant" is fine; "hey my personal assistant" is too long |

When in doubt, run `suggest_hard_negatives.py` *before* committing to a phrase — if the LLM produces 50+ obvious collisions, pick a different phrase.
