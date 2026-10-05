# Reimplementation requirements

This project reimplements the paper's learning and behavior boundary with current PyTorch and Transformers APIs.

## Required behavior

1. Use one BERT-family backbone for all predictions.
2. Predict sentence intent (`conversation` or `action`) from the shared sequence representation.
3. Jointly predict token-level `action`, `position`, and `target` labels from the shared token representations.
4. Train all four heads in one optimization step. Support the paper's frozen-backbone regime and an explicit fine-tuning option.
5. Accept JSONL with character-offset entity spans and align those spans to fast-tokenizer offsets.
6. Save model weights, backbone/tokenizer, and label vocabularies together.
7. Turn action predictions into a dispatch plan only after checking required entities, target existence, and scene affordances.
8. Route conversation predictions separately instead of inventing an action.

## Data boundaries

- The paper's 1,200-sentence controlled-room dataset is unavailable and is not reconstructed or claimed.
- Users may author domain utterances, generate reviewed synthetic paraphrases, or adapt a suitably licensed public intent/slot dataset.
- Labels describe the user's deployment scene. Public slot datasets do not automatically contain the paper's action/position/target ontology.

## Acceptance checks

- Dataset validation rejects overlapping or out-of-range spans and actions without action/target entities.
- Dispatcher tests demonstrate allowed, missing-entity, unknown-target, and unsupported-affordance paths.
- Source files compile without downloading a backbone; training/inference are documented but not run in verification.

## Bundled fictional avatar substitution

Two newly generated fictional CC0 humanoids replace the original avatar assets in the browser demo. They provide a 53-bone rig and named ARKit/viseme targets. Motion retargeting adapts source joints to their bind pose; speaking envelopes approximate mouth motion rather than phoneme alignment. The optional recorded BEAT companion inspects public motion, face and audio files prepared locally, independently of the paper's learned algorithm. No dataset recordings or trained weights are bundled.

## Local recorded co-speech integration

The browser application retrieves prepared BEAT body-motion clips with `automatic` mode: a current public-demo adapter; it is not a method claimed by the intent/action paper. The first `python scripts/start_demo.py` run fetches a small official BVH/TextGrid sample, constructs a nine-clip bank, and fits the local retrieval artifact under ignored `outputs/beat-library/`. Install `scripts/requirements-demo.txt` first. Preparation code and method dependencies are vendored in this repository; no sibling clone, original institute library, full dataset, or pretrained weights are bundled. The intent/entity interpretation and affordance-checked action dispatch remain this application's core; action/navigation poses are separate. The separate recorded-motion companion remains available for local motion/face/audio inspection.
