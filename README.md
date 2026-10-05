# Auto-generating Virtual Human Behavior by Understanding User Contexts

**Hanseob Kim, Ghazanfar Ali, Seungwon Kim, Gerard J. Kim, Jae-In Hwang**

**IEEE VR Abstracts and Workshops · 2021** · Published

[Paper / publisher](https://doi.org/10.1109/vrw52623.2021.00178) · [Project page](https://ghazanfarali.com/research/context-aware-behavior/) · [BibTeX](CITATION.bib) · [Requirements](REQUIREMENTS.md) · [Code & setup](#implementation-and-usage)

> Natural-language context activates grounded virtual-human actions.

![Method diagram from Figure 2 of the context-aware-behavior paper](paper-assets/method.png)

*Original method figure from the paper: Figure 2, PDF page 2. Extracted for this research introduction; the diagram describes the original system, not verification of this reimplementation.*

## Why this research

Fixed trigger phrases make virtual-human actions brittle. Joint language understanding can distinguish conversation from an action request and identify the entities needed to act in a known environment.

A BERT-based model jointly classifies whether a sentence requests conversation or action and extracts action entities. An interaction module turns those predictions into virtual-human behaviors in a controlled room scenario.

## Method at a glance

**Natural-language input** → **Sentence + entity classifier** → **Grounded room behavior**

| | Research system |
|---|---|
| Input | Natural-language conversation and action requests |
| Method | Joint sentence classification and entity classification |
| Output | Action intent, extracted entities, and virtual-human behavior |

## Evidence and scope

Pilot study of perceived naturalness and user experience

**Attribution:** These findings describe the paper or manuscript, not results obtained with this repository's code.

**Study context:** Controlled room scenario with a fixed object and interaction set.

**Limitations:** The controlled object and action inventory limits generalization to arbitrary environments.

## Explore the implementation

A shared BERT backbone with intent/entity heads, labeled-data validation, training/inference and scene-affordance checks before action dispatch.

This repository contains independently written research code. The institute's original source, datasets and trained models are not distributed. Public-data preparation, commands, assumptions and checks are documented below and in [REQUIREMENTS.md](REQUIREMENTS.md).

## Resources and citation

Read the paper through its [publisher record](https://doi.org/10.1109/vrw52623.2021.00178). PDFs are hosted by publishers or preprint archives rather than stored in this repository.

Please cite the research paper when using its ideas; [download the BibTeX citation](CITATION.bib). The implementation has its own documented scope.

## Implementation and usage

<!-- implementation-guide -->

This repository reimplements the core method from **“Auto-generating Virtual Human Behavior by Understanding User Contexts”** by Hanseob Kim, Ghazanfar Ali, Seungwon Kim, Gerard J. Kim, and Jae-In Hwang, IEEE VR Abstracts and Workshops 2021, pp. 591–592. DOI: [10.1109/VRW52623.2021.00178](https://doi.org/10.1109/VRW52623.2021.00178).

The original institute code and 1,200-sentence room dataset are unavailable. This independent implementation provides the paper's essential shared BERT backbone, sentence conversation/action head, three token-level entity heads, joint loss, and grounded action dispatch. It does not include the paper's model weights, participant data, Unity room, animations, speech services, or reported study results.

### Immediate browser demo

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
python scripts/prepare_viewer.py
python -m context_behavior.demo
```

Open `http://127.0.0.1:8762`. The default room uses authored phrase rules and a procedural character; trained BERT inference uses the checkpoint workflow below.

### Detailed setup and checks

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest -q
```

The first training run downloads the selected Hugging Face backbone. See the official [BERT model documentation](https://huggingface.co/docs/transformers/model_doc/bert) and [token-classification guide](https://huggingface.co/docs/transformers/main/tasks/token_classification).

### Procedural verification

Run `python scripts/verify.py` after installation. It saves a small local BERT/tokenizer, trains the normal JSONL pipeline for two CPU epochs, reloads the checkpoint through the inference CLI, and validates the prediction/dispatch output schema. Because randomly initialized weights are not expected to be accurate, a separate known-valid parse checks scene grounding deterministically. Inspect the dataset, checkpoint, prediction, and grounded result under `outputs/verify/`. For real data, keep the same JSONL and scene contracts below and replace the local backbone and training records.

### Prepare domain data

Training input is UTF-8 JSONL. Spans use Python character offsets (`start` inclusive, `end` exclusive), must not overlap, and must exactly match `value`:

```json
{"text":"Please put the pillow on the bed","intent":"action","entities":[{"start":7,"end":10,"type":"action","value":"put"},{"start":15,"end":21,"type":"target","value":"pillow"},{"start":22,"end":24,"type":"position","value":"on"}]}
{"text":"How are you today?","intent":"conversation","entities":[]}
```

Author a balanced inventory around objects and affordances that actually exist in your scene, then add human-reviewed paraphrases. Keep train/validation/test speakers or templates separate to avoid paraphrase leakage. Validate before training:

```powershell
context-behavior-data validate .\data\train.jsonl
```

For public starting material, the [SNIPS NLU benchmark](https://github.com/snipsco/nlu-benchmark) supplies intent/slot utterances under Apache-2.0. Its domains are not room actions: review its license, map only compatible utterances into this contract, and author the action/position/target labels for your scene. Synthetic utterances must be reviewed because a generator can create impossible or mislabeled actions. No dataset is bundled.

### Train and infer

The default freezes the shared backbone as described in the paper and learns all four heads jointly:

```powershell
context-behavior-train --train .\data\train.jsonl --output .\artifacts\room-model --epochs 5
```

Add `--fine-tune-backbone` to update the backbone. That is an extension, so report it when comparing experiments.

Create `scene.json` to bind language labels to real application capabilities:

```json
{
  "objects": [
    {"id": "lamp", "affordances": ["turn_on", "turn_off"]},
    {"id": "drawer", "affordances": ["open", "close"]}
  ]
}
```

Then run:

```powershell
context-behavior-infer --model .\artifacts\room-model --scene .\scene.json "Please turn on the lamp"
```

Inference reports the predicted intent/entities and a dispatch decision. The dispatcher refuses missing entities, unknown targets, and unsupported affordances. A host application should route accepted commands to its animation controller and conversation intents to its chosen dialogue system.

### Local 3D room

Run `python scripts/prepare_viewer.py` once to fetch a pinned Three.js module into ignored `static/vendor/`. Start `python -m context_behavior.demo` and open `http://127.0.0.1:8762`. The default interpreter is **explicit authored phrase rules** over `demo/scene.json`; the interface labels it that way and never presents it as a trained model. For shared BERT intent/entity inference, train with the JSONL workflow above and start `python -m context_behavior.demo --model artifacts/room-model`. The room changes only when `ActionDispatcher` accepts an action, and every object exposes its current state and affordances. The procedural character is an integration renderer; the paper's Unity character is not distributed.

`python scripts/verify.py` trains a tiny randomly initialized BERT checkpoint to check pipeline wiring; its predictions are not accuracy evidence. Use reviewed scene-specific training and held-out validation data before relying on model output. The wearable MR agent paper is the framework lineage for this work, not a runtime dependency of this standalone repository.

### Optional local speech

Browser speech is selected by default. To enable **Local Kokoro** and audio transcription, install `python -m pip install -e ".[speech]"`. Set `KOKORO_MODEL_DIR` to a user-prepared folder containing `config.json`, `kokoro-v1_0.pth`, and `voices/af_heart.pt` from [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M). Follow the [Kokoro phonemizer setup](https://github.com/hexgrad/kokoro), including espeak-ng where needed. Set `WHISPER_MODEL_DIR` to a locally prepared faster-whisper small model directory containing `model.bin`. In PowerShell, set `$env:KOKORO_MODEL_DIR='C:\path\to\kokoro'` and `$env:WHISPER_MODEL_DIR='C:\path\to\whisper-small'` before starting the room. Record or upload audio to fill the utterance field; action routing still passes through the scene-affordance checks. Missing paths produce explicit errors and never trigger model downloads.

<!-- avatar-recorded-motion:start -->
## Bundled characters and recorded public motion

The browser demos include Rowan and Mira, two new fictional GLB characters built with MPFB and MakeHuman community assets under CC0 1.0. See [avatar licensing and provenance](static/avatars/LICENSE.md). Use the character selector in the stage. The shared renderer supports body bones, ARKit facial channels, and approximate speaking motion.

Recorded motion is adapted to the characters' proportions. Palm landmarks set hand orientation; finger curl uses bounded hinge bends and preserves the character's finger spacing. Distal bends are estimated from the preceding joint when fingertip landmarks are absent. Use the companion's hand close-up views to inspect the result.

The [recorded BEAT motion companion](static/recorded-motion.html) opens at `/static/recorded-motion.html` while the demo server is running. It plays locally selected motion, face, and WAV files on the bundled characters; this is recorded public-data inspection, separate from the paper implementation. No BEAT recording, dataset archive, or trained model is bundled. Install the one preparation dependency and fetch a small official sample into ignored `outputs/beat-demo/`:

```sh
python -m pip install numpy
python scripts/beat_demo/fetch_modalities.py --speaker 1 --sequence 1_wayne_0_1_1 --include-bvh --max-bytes 25000000 --output-dir outputs/beat-demo/source
python scripts/beat_demo/prepare_bvh.py --bvh outputs/beat-demo/source/1_wayne_0_1_1.bvh --output outputs/beat-demo/sample/1_wayne_0_1_1-raw-motion.json --frames 120
python scripts/beat_demo/prepare_modalities.py --sequence 1_wayne_0_1_1 --source outputs/beat-demo/source --output outputs/beat-demo/sample --frames 120
```

Open the companion and select `outputs/beat-demo/sample/1_wayne_0_1_1-raw-motion.json`, `1_wayne_0_1_1-face.json`, and `1_wayne_0_1_1.wav`. The downloader caps each original file at 25 MB; the prepared clip contains up to 120 frames. The viewer uses local files and does not upload them. For other BEAT takes, substitute a matching official speaker and sequence ID.

If you already have OmniMo's processed 52-joint Unity humanoid data, use that normalized motion instead:

```sh
python scripts/beat_demo/prepare.py --dataset /path/to/processed/beat --speaker 1 --take 1_wayne_0_1_1 --output outputs/beat-demo/sample/1_wayne_0_1_1-motion.json --max-frames 120
```

Select the resulting `*-motion.json` in the companion. Its metadata carries the humanoid joint mapping and source-to-avatar coordinate conversion. The viewer fits source FK directions from the avatar's bind pose, following the spine explicitly at branching joints. This avoids applying incompatible source bone twist to the MPFB skin; it does not reproduce exact performer twist. The adapter supports Unity proximal/intermediate/distal finger names. Raw BVH remains a public-data alternative; do not mix the two skeleton conventions.
<!-- avatar-recorded-motion:end -->
