"""Start the context-aware behavior room.

Uses the trained starter checkpoint in outputs/starter-model when present; otherwise
the labelled rule fallback. --train-starter first trains that checkpoint on the bundled
authored starter dataset (frozen BERT, so a few minutes on CPU; the first run downloads
the backbone, bert-base-uncased by default).
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STARTER_MODEL = ROOT / 'outputs' / 'starter-model'
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--port', type=int, default=8080)
parser.add_argument('--skip-beat', action='store_true', help='Skip preparation; use the existing local cache or authored fixture when absent')
parser.add_argument('--model', type=Path, help='Checkpoint folder from context-behavior-train (default: outputs/starter-model when present)')
parser.add_argument('--train-starter', action='store_true', help='Train outputs/starter-model on the bundled starter dataset before starting')
parser.add_argument('--backbone', default='google-bert/bert-base-uncased', help='Hugging Face id or local folder used by --train-starter')
parser.add_argument('--rules', action='store_true', help='Force the labelled rule fallback even if a checkpoint exists')
parser.add_argument('--dialogue', choices=['fixed', 'openai', 'command'])
parser.add_argument('--dialogue-url')
parser.add_argument('--dialogue-model')
parser.add_argument('--dialogue-command')
args = parser.parse_args()
os.chdir(ROOT)
os.environ['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + os.environ.get('PYTHONPATH', '')
prepare = ROOT / 'scripts' / 'prepare_viewer.py'
if prepare.exists() and not all((ROOT / 'static' / 'vendor' / name).is_file() for name in ('three.module.js', 'GLTFLoader.js', 'BufferGeometryUtils.js')):
    subprocess.run([sys.executable, str(prepare)], check=True)

if not args.skip_beat:
    subprocess.run([sys.executable, str(ROOT/'scripts/prepare_beat_demo.py')], check=True)

if args.train_starter:
    print(f'Training the starter checkpoint with {args.backbone} (frozen encoder) ...', flush=True)
    subprocess.run([sys.executable, '-m', 'context_behavior.train', '--output', str(STARTER_MODEL), '--backbone', args.backbone], check=True)

command = [sys.executable, '-m', 'context_behavior.demo', '--port', str(args.port)]
model = args.model or (STARTER_MODEL if (STARTER_MODEL / 'heads.pt').is_file() else None)
if model and not args.rules:
    command += ['--model', str(model)]
    print(f'Interpreter: trained checkpoint {model}', flush=True)
else:
    print('Interpreter: labelled rule fallback (run with --train-starter to train the BERT starter model).', flush=True)
for name in ('dialogue', 'dialogue_url', 'dialogue_model', 'dialogue_command'):
    if getattr(args, name):
        command += ['--' + name.replace('_', '-'), getattr(args, name)]
print(f'Open http://127.0.0.1:{args.port}/ — the bundled room and starter examples are ready.', flush=True)
raise SystemExit(subprocess.call(command))
