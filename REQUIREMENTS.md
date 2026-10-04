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

