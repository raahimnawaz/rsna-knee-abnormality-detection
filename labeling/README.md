# Hand-labeling workflow

Builds the validation set the competition doesn't give you. Only **58 of 4,407** training
studies carry gold labels, and per-language coverage runs from 28 (English) down to **0**
(French) — not enough to validate a 12-label multilingual extractor. See `../FINDINGS.md`.

## Run it

```
start "" "C:\Users\Raahim\rsna-knee-mri\labeling\labeler.html"
```

Self-contained: no server, no internet. Progress autosaves to browser localStorage on every
keystroke, so you can close the tab and resume. **Export to CSV periodically** — localStorage
is per-browser-profile and clearing site data wipes it.

## Keys

| | |
|---|---|
| `1`–`9`, `0`, `-`, `=` | cycle that label: — → **+** → **?** → **−** → — |
| `→` / `←` | browse without marking |
| `Enter` | mark done, jump to next unlabeled |
| `f` | flag for review |
| `t` | jump to notes |
| `h` | toggle highlighting |

Label states: **+** present · **?** uncertain/hedged · **−** explicitly negated ·
**—** not mentioned. Default is **—**, so you only touch the labels the report discusses.

Keeping "not mentioned" and "explicitly negated" separate costs nothing now and lets you
measure later whether your extractor confuses silence with denial — collapse both to 0 when
you build binary targets.

## What's in the sample

303 items / 283 unique studies, from `sample_for_labeling.py` (seed 20260806):

- **253 stratified** across all 9 languages, with a floor of 20 per language so French and
  Dutch get real coverage, and English capped at 60 (it already has 28 of the 58 gold labels).
- **30 gold studies mixed in blind.** Compare your labels against the official ones
  afterwards. If you disagree systematically, your extractor inherits that disagreement
  across all 4,349 studies — better to find out now.
- **20 duplicates** at distant positions, for intra-rater consistency. If you can't reproduce
  yourself, your ceiling is lower than you think.

Gold and duplicate flags are in `labeling_sample.csv` but deliberately not shown in the UI.

Budget ~6 hours at 60–90 s/report. Do it in sittings; progress persists.

## Highlighting

`glossary.json` holds stem-based terms for all 12 findings plus negation and uncertainty
cues, in all 9 languages. Matching is case-, accent- and encoding-insensitive (see
`FINDINGS.md` §2.2 — Greek uses the micro sign, not Greek mu).

Colours: amber = finding term · green = negation cue · purple = uncertainty cue. The number
beside each label is how many glossary terms fired for it — an attention aid, **not a
prediction**. Read the sentence.

The glossary is reusable: it seeds the rule-based extractor that cross-checks the LLM in
`../PLAN.md` §2.

## Files

| | |
|---|---|
| `sample_for_labeling.py` | builds the stratified sample → `labeling_sample.csv` |
| `build_labeler.py` | inlines sample + glossary → `labeler.html` |
| `glossary.json` | multilingual term lists |
| `bulgarian_patch.py`, `greek_bg_patch.py` | glossary corrections, applied; kept for provenance |

Regenerate after editing the glossary:

```
python build_labeler.py
```

Re-running `sample_for_labeling.py` reshuffles item IDs and **invalidates saved progress** —
export your CSV first.

## When done

Export the CSV, then compare against the 30 blind gold studies per-label, and check the 20
duplicate pairs for self-consistency. Those two numbers tell you how much to trust every
downstream pseudo-label.
