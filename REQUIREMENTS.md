# Reimplementation requirements

This project reimplements the paper's learning and behaviour boundary with current PyTorch and Transformers APIs.

## Required behavior

1. Use the paper's Table 1 vocabulary as the label space: Subject 2, Action 14, Position 6 and Target 11 classes, each including None as listed. Store it in `src/context_behavior/resources/ontology.json`.
2. Encode text with one frozen BERT-family backbone. Predict Subject from [CLS], and predict Action, Position and Target with sentence-level classifiers over the ontology classes. Train all four heads in one optimisation step with summed cross-entropy. Fine-tuning the backbone is an explicit, reported extension.
3. Accept JSONL rows of `{text, subject, action, position, target}` class labels. Allow target-less actions (Walk, Run, Idle, Stand up, Sit, Lay, Put). Convert the earlier span JSONL through the ontology synonym map.
4. Save a self-contained checkpoint: backbone config and weights, tokenizer, ontology, head weights and training metrics. Inference and the demo server load it once without network access.
5. Combine Action + Position + Target with scene object metadata (name, class, position, facing direction, affordances) into a behaviour plan. Reject unsupported combinations with a reason.
6. Route Subject None (small talk and negated requests) to a pluggable dialogue client with a fixed fallback reply, instead of inventing an action.

## Data boundaries

- The paper's 1,200-sentence controlled-room dataset is unavailable and is neither reconstructed nor claimed.
- The bundled starter dataset (425 sentences, CC0 1.0) was written and label-checked for this repository. Its validation accuracy is informational, not a reproduction of the paper's results.
- Labels describe the deployment scene. Extend the data with reviewed sentences for your own objects, and keep the Table 1 classes or update the ontology deliberately.

## Acceptance checks

- Ontology tests assert the exact Table 1 class lists.
- Data validation accepts target-less actions and rejects conversation rows with entities, Virtual Human rows without an Action, and actions that need a Target but lack one.
- Model tests run forward and training on a tiny randomly initialised local BERT, with no downloads, and reload the self-contained checkpoint.
- Planner tests cover Sit + On + Chair → `sit_on` at seat height, lie on bed, open/close/switch state, bring/hold/put prop moves, and rejected combinations with reasons.
- Rule-fallback tests keep the audit inputs: "Don't open the window" is conversation, "Turn the lamp off, it is on" is Turn off, and "Put the pillow on the bed" puts the pillow.
- Dialogue tests cover the OpenAI-compatible client, the local command client and the fallback. Server tests check that the model is loaded once.
- `scripts/verify.py` trains the tiny backbone on the starter split through the CLI, reloads it with the inference CLI, and plans known-valid combinations.

## Bundled fictional avatar substitution

Two newly generated fictional CC0 humanoids replace the original avatar assets in the browser demo. They provide a 53-bone rig and named ARKit/viseme targets. Motion retargeting adapts source joints to their bind pose; mouth shapes follow a rule-based text-to-phoneme-to-viseme track timed to speech playback, an approximation rather than forced phoneme alignment. The optional recorded BEAT companion inspects public motion, face and audio files prepared locally, independently of the paper's learned algorithm. No dataset recordings or trained weights are bundled.

## Local recorded co-speech integration

The browser application retrieves prepared BEAT body-motion clips with `automatic` mode: a current public-demo adapter; it is not a method claimed by the intent/action paper. The first `python scripts/start_demo.py` run fetches a small official BVH/TextGrid sample, constructs a nine-clip bank, and fits the local retrieval artifact under ignored `outputs/beat-library/`. Install `scripts/requirements-demo.txt` first. Preparation code and method dependencies are vendored in this repository; no sibling clone, original institute library, full dataset, or pretrained weights are bundled. The intent/entity interpretation and affordance-checked action dispatch remain this application's core; action/navigation poses are separate. The separate recorded-motion companion remains available for local motion/face/audio inspection.
