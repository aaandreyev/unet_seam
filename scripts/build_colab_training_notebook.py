from __future__ import annotations
"""Regenerates colab/seam_residual_corrector_train_eval_colab.ipynb.

The checked-in notebook is maintained to match this builder; re-run
`python scripts/build_colab_training_notebook.py` only when you update
`build_notebook()` and want the ipynb on disk to reflect it.
"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "colab" / "seam_residual_corrector_train_eval_colab.ipynb"

EMBED_FILES = [
    "configs/model_resunet_v1.yaml",
    "configs/train_synth_v1.yaml",
    "configs/eval_v1.yaml",
    "configs/export_v1.yaml",
    "src/__init__.py",
    "src/data/__init__.py",
    "src/data/manifest.py",
    "src/data/preprocess.py",
    "src/data/strip_geometry.py",
    "src/data/corruptions.py",
    "src/data/synthetic_strip_dataset.py",
    "src/data/cached_strip_dataset.py",
    "src/models/__init__.py",
    "src/models/blocks.py",
    "src/models/resunet.py",
    "src/losses/__init__.py",
    "src/losses/lowfreq.py",
    "src/losses/perceptual.py",
    "src/losses/residual_guard.py",
    "src/losses/seam_losses.py",
    "src/metrics/__init__.py",
    "src/metrics/bootstrap.py",
    "src/metrics/deltae.py",
    "src/metrics/lowfreq_metrics.py",
    "src/metrics/reports.py",
    "src/metrics/seam_metrics.py",
    "src/train/__init__.py",
    "src/train/checkpoint.py",
    "src/train/ema.py",
    "src/train/scheduler.py",
    "src/train/train_loop.py",
    "src/utils/__init__.py",
    "src/utils/device.py",
    "src/utils/image_io.py",
    "src/utils/phash.py",
    "src/utils/seed.py",
    "scripts/train_resunet.py",
    "scripts/run_eval.py",
    "scripts/export_safetensors.py",
    "scripts/verify_export.py",
]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text}


def file_map_literal() -> str:
    file_map = {}
    for rel in EMBED_FILES:
        file_map[rel] = (ROOT / rel).read_text(encoding="utf-8")
    return json.dumps(file_map, ensure_ascii=False, indent=2)


def build_notebook() -> dict:
    files_literal = file_map_literal()
    cells = [
        md(
            "# Seam Residual Corrector v1 Colab Notebook\n\n"
            "Готовый ноутбук для **training -> validation -> eval -> export -> verify_export**.\n\n"
            "Локально нужно заранее собрать training bundle через `scripts/build_final_training_bundle.py`, "
            "загрузить `.tar.gz` на Google Drive и указать путь в параметрах ниже."
        ),
        code(
            "# 0. PARAMS\n"
            "from pathlib import Path\n"
            "import os\n\n"
            "DATASET_BUNDLE_DRIVE_PATH = '/content/drive/MyDrive/unet_seam/seam_residual_corrector_training_bundle.tar.gz'\n"
            "DRIVE_RUNS_DIR = '/content/drive/MyDrive/unet_seam_runs'\n"
            "RUN_NAME = 'seam_residual_corrector_v1_run001'\n"
            "USE_RAMDISK = True\n"
            "RAMDISK_SIZE_GB = 48\n"
            "COPY_ARCHIVE_TO_RAM_FIRST = True\n"
            "SYNC_INTERVAL_SEC = 180\n"
            "TRAIN_BATCH_SIZE = 32\n"
            "TRAIN_EPOCHS = 20\n"
            "TRAIN_NUM_WORKERS = 16\n"
            "VAL_BATCH_SIZE = 8\n"
            "PRIMARY_CHECKPOINT = 'best_boundary_ciede2000.pt'\n"
            "PROJECT_ROOT = Path('/content/seam_runtime')\n"
            "LOCAL_OUTPUTS = PROJECT_ROOT / 'outputs'\n"
            "LOCAL_CHECKPOINTS = LOCAL_OUTPUTS / 'checkpoints'\n"
            "LOCAL_EVAL = LOCAL_OUTPUTS / 'eval_reports'\n"
            "LOCAL_EXPORTS = LOCAL_OUTPUTS / 'exports'\n"
            "LOCAL_LOGS = LOCAL_OUTPUTS / 'logs'\n"
            "LOCAL_DATA_ROOT = Path('/content/dataset_bundle')\n"
            "DRIVE_RUN_DIR = Path(DRIVE_RUNS_DIR) / RUN_NAME\n"
            "DRIVE_CKPT_DIR = DRIVE_RUN_DIR / 'checkpoints'\n"
            "DRIVE_EVAL_DIR = DRIVE_RUN_DIR / 'eval_reports'\n"
            "DRIVE_EXPORT_DIR = DRIVE_RUN_DIR / 'exports'\n"
            "DRIVE_LOG_DIR = DRIVE_RUN_DIR / 'logs'\n"
        ),
        code(
            "# 1. MOUNT DRIVE\n"
            "from google.colab import drive\n"
            "drive.mount('/content/drive')\n"
            "print('Drive mounted')\n"
        ),
        code(
            "# 2. INSTALL / RUNTIME CHECKS\n"
            "import os, sys, subprocess, importlib.util, platform, json\n"
            "pkgs = ['pyyaml','scipy','scikit-image','safetensors','tqdm','lpips','psutil']\n"
            "subprocess.run(['apt-get', 'update', '-qq'], check=False)\n"
            "subprocess.run(['apt-get', 'install', '-y', '-qq', 'pigz'], check=False)\n"
            "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + pkgs, check=True)\n"
            "import torch\n"
            "device = 'cuda' if torch.cuda.is_available() else 'mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else 'cpu'\n"
            "print(json.dumps({'python': sys.executable, 'platform': platform.platform(), 'torch': torch.__version__, 'device': device}, ensure_ascii=False))\n"
        ),
        code(
            "# 3. OPTIONAL RAMDISK\n"
            "import os, psutil, subprocess, shutil\n"
            "RAM_ROOT = Path('/content/ramdisk')\n"
            "if USE_RAMDISK:\n"
            "    mem = psutil.virtual_memory()\n"
            "    eff = min(RAMDISK_SIZE_GB, max(8, int((mem.available / (1024**3)) * 0.5)))\n"
            "    RAM_ROOT.mkdir(parents=True, exist_ok=True)\n"
            "    mounted = subprocess.run(['mountpoint', str(RAM_ROOT)], capture_output=True).returncode == 0\n"
            "    if not mounted:\n"
            "        subprocess.run(['mount', '-t', 'tmpfs', '-o', f'size={eff}G,mode=777', 'tmpfs', str(RAM_ROOT)], check=True)\n"
            "    DATA_ROOT = RAM_ROOT / 'dataset_bundle'\n"
            "else:\n"
            "    DATA_ROOT = LOCAL_DATA_ROOT\n"
            "DATA_ROOT.mkdir(parents=True, exist_ok=True)\n"
            "LOCAL_OUTPUTS.mkdir(parents=True, exist_ok=True)\n"
            "for p in [LOCAL_CHECKPOINTS, LOCAL_EVAL, LOCAL_EXPORTS, LOCAL_LOGS]:\n"
            "    p.mkdir(parents=True, exist_ok=True)\n"
            "print('DATA_ROOT =', DATA_ROOT)\n"
        ),
        code(
            "# 4. EXTRACT DATASET BUNDLE FROM DRIVE\n"
            "import shutil, tarfile, time\n"
            "from tqdm.auto import tqdm\n\n"
            "src = Path(DATASET_BUNDLE_DRIVE_PATH)\n"
            "if not src.exists():\n"
            "    raise FileNotFoundError(src)\n"
            "archive_local = src\n"
            "if COPY_ARCHIVE_TO_RAM_FIRST and USE_RAMDISK:\n"
            "    archive_local = RAM_ROOT / src.name\n"
            "    with src.open('rb') as fin, archive_local.open('wb') as fout:\n"
            "        total = src.stat().st_size\n"
            "        with tqdm(total=total, unit='B', unit_scale=True, desc='copy_archive') as pbar:\n"
            "            while True:\n"
            "                chunk = fin.read(8 * 1024 * 1024)\n"
            "                if not chunk:\n"
            "                    break\n"
            "                fout.write(chunk)\n"
            "                pbar.update(len(chunk))\n"
            "if DATA_ROOT.exists():\n"
            "    shutil.rmtree(DATA_ROOT)\n"
            "DATA_ROOT.mkdir(parents=True, exist_ok=True)\n"
            "with tarfile.open(archive_local, 'r:*') as tf:\n"
            "    members = tf.getmembers()\n"
            "    for m in tqdm(members, desc='extract_bundle', dynamic_ncols=True):\n"
            "        tf.extract(m, DATA_ROOT)\n"
            "print('bundle extracted to', DATA_ROOT)\n"
        ),
        code(
            "# 5. WRITE PROJECT FILES INTO RUNTIME\n"
            "import json, os\n"
            f"PROJECT_FILES = {files_literal}\n"
            "for rel, text in PROJECT_FILES.items():\n"
            "    path = PROJECT_ROOT / rel\n"
            "    path.parent.mkdir(parents=True, exist_ok=True)\n"
            "    path.write_text(text, encoding='utf-8')\n"
            "sys.path.insert(0, str(PROJECT_ROOT))\n"
            "print('project files written:', len(PROJECT_FILES))\n"
        ),
        code(
            "# 6. VALIDATE BUNDLE LAYOUT\n"
            "import json\n"
            "required = [\n"
            "    DATA_ROOT / 'manifests/strip_train_cache.jsonl',\n"
            "    DATA_ROOT / 'manifests/strip_val_cache.jsonl',\n"
            "    DATA_ROOT / 'outputs/strip_cache/train',\n"
            "    DATA_ROOT / 'outputs/strip_cache/val',\n"
            "]\n"
            "for p in required:\n"
            "    if not p.exists():\n"
            "        raise FileNotFoundError(p)\n"
            "print(json.dumps({\n"
            "    'train_cache_dirs': len([p for p in (DATA_ROOT / 'outputs/strip_cache/train').iterdir() if p.is_dir()]),\n"
            "    'val_cache_dirs': len([p for p in (DATA_ROOT / 'outputs/strip_cache/val').iterdir() if p.is_dir()]),\n"
            "}, ensure_ascii=False))\n"
        ),
        code(
            "# 7. BUILD RUNTIME CONFIGS (обязателен перед 10, либо 10 сам вызовет тот же скрипт, если yml нет)\n"
            "import subprocess, sys\n"
            "subprocess.check_call(\n"
            "    [sys.executable, str(PROJECT_ROOT / 'scripts' / 'write_colab_runtime_yamls.py'),\n"
            "     '--project-root', str(PROJECT_ROOT), '--data-root', str(DATA_ROOT),\n"
            "     '--train-batch-size', str(TRAIN_BATCH_SIZE), '--train-epochs', str(TRAIN_EPOCHS),\n"
            "     '--train-num-workers', str(TRAIN_NUM_WORKERS), '--primary-checkpoint', PRIMARY_CHECKPOINT],\n"
            "    cwd=str(PROJECT_ROOT),\n"
            ")\n"
            "print('runtime configs ->', PROJECT_ROOT / 'runtime_configs')\n"
        ),
        code(
            "# 8. OPTIONAL RESUME FROM DRIVE LAST CHECKPOINT\n"
            "import shutil\n"
            "resume_path = None\n"
            "DRIVE_CKPT_DIR.mkdir(parents=True, exist_ok=True)\n"
            "drive_last = DRIVE_CKPT_DIR / 'last.pt'\n"
            "if drive_last.exists():\n"
            "    LOCAL_CHECKPOINTS.mkdir(parents=True, exist_ok=True)\n"
            "    local_resume = LOCAL_CHECKPOINTS / 'resume_last.pt'\n"
            "    shutil.copy2(drive_last, local_resume)\n"
            "    resume_path = local_resume\n"
            "print('resume_path =', resume_path)\n"
        ),
        code(
            "# 9. BACKGROUND SYNC TO DRIVE (eval/exports on Drive only after you run cells 11–12; checkpoints sync during train if this cell ran before 10)\n"
            "import threading, time, shutil\n"
            "SYNC_STOP = threading.Event()\n"
            "for p in [DRIVE_RUN_DIR, DRIVE_CKPT_DIR, DRIVE_EVAL_DIR, DRIVE_EXPORT_DIR, DRIVE_LOG_DIR]:\n"
            "    p.mkdir(parents=True, exist_ok=True)\n"
            "def sync_tree(src: Path, dst: Path) -> int:\n"
            "    n = 0\n"
            "    if not src.exists():\n"
            "        return 0\n"
            "    for path in src.rglob('*'):\n"
            "        if not path.is_file():\n"
            "            continue\n"
            "        rel = path.relative_to(src)\n"
            "        out = dst / rel\n"
            "        out.parent.mkdir(parents=True, exist_ok=True)\n"
            "        if not out.exists() or out.stat().st_mtime < path.stat().st_mtime or out.stat().st_size != path.stat().st_size:\n"
            "            try:\n"
            "                shutil.copy2(path, out)\n"
            "                n += 1\n"
            "            except OSError as e:\n"
            "                print('sync error', path, '->', out, e, flush=True)\n"
            "    return n\n"
            "def sync_all_to_drive() -> None:\n"
            "    a = sync_tree(LOCAL_CHECKPOINTS, DRIVE_CKPT_DIR)\n"
            "    b = sync_tree(LOCAL_EVAL, DRIVE_EVAL_DIR)\n"
            "    c = sync_tree(LOCAL_EXPORTS, DRIVE_EXPORT_DIR)\n"
            "    d = sync_tree(LOCAL_LOGS, DRIVE_LOG_DIR)\n"
            "    print('sync tick: ckpt', a, 'eval', b, 'export', c, 'logs', d, '->', DRIVE_RUN_DIR, flush=True)\n"
            "def sync_loop():\n"
            "    while True:\n"
            "        sync_all_to_drive()\n"
            "        if SYNC_STOP.wait(SYNC_INTERVAL_SEC):\n"
            "            break\n"
            "sync_all_to_drive()\n"
            "sync_thread = threading.Thread(target=sync_loop, daemon=True)\n"
            "sync_thread.start()\n"
            "print('background sync started (first tick already ran)')\n"
        ),
        code(
            "# 10. TRAIN\n"
            "import os, sys, subprocess, threading\n"
            "if not (PROJECT_ROOT / 'runtime_configs' / 'train.yaml').is_file():\n"
            "    print('Нет runtime_configs — запускаю scripts/write_colab_runtime_yamls.py (как в ячейке 7)…')\n"
            "    subprocess.check_call(\n"
            "        [sys.executable, str(PROJECT_ROOT / 'scripts' / 'write_colab_runtime_yamls.py'),\n"
            "         '--project-root', str(PROJECT_ROOT), '--data-root', str(DATA_ROOT),\n"
            "         '--train-batch-size', str(TRAIN_BATCH_SIZE), '--train-epochs', str(TRAIN_EPOCHS),\n"
            "         '--train-num-workers', str(TRAIN_NUM_WORKERS), '--primary-checkpoint', PRIMARY_CHECKPOINT],\n"
            "        cwd=str(PROJECT_ROOT),\n"
            "    )\n"
            "env = os.environ.copy()\n"
            "env['PYTHONPATH'] = str(PROJECT_ROOT)\n"
            "env['PYTHONUNBUFFERED'] = '1'\n"
            "cmd = [sys.executable, '-u', '-m', 'scripts.train_resunet', '--config', str(PROJECT_ROOT / 'runtime_configs/train.yaml')]\n"
            "if resume_path is not None:\n"
            "    cmd += ['--resume', str(resume_path)]\n"
            "print('TRAIN CMD:', ' '.join(map(str, cmd)))\n"
            "print('Дальше: loading, train_start, epoch_begin, train_iter_begin (затем пауза до первого train_step — нормально: 1-й батч + воркеры + GPU).')\n"
            "def _stream_cmd(cmd, cwd, env):\n"
            "    # Colab: inherited stdio often shows nothing; PIPE + a thread that prints to the cell is reliable. Main thread only p.wait() (not read()).\n"
            "    p = subprocess.Popen(\n"
            "        cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,\n"
            "    )\n"
            "    def _pump():\n"
            "        if p.stdout is not None:\n"
            "            for line in p.stdout:\n"
            "                print(line, end='', flush=True)\n"
            "    t = threading.Thread(target=_pump, daemon=True)\n"
            "    t.start()\n"
            "    rc = p.wait()\n"
            "    t.join(timeout=30)\n"
            "    if rc != 0:\n"
            "        print(f'\\n[exit {rc}] Причина — в Traceback/ошибке ВЫШЕ. CalledProcessError — только итог.', flush=True)\n"
            "        raise subprocess.CalledProcessError(rc, cmd)\n"
            "_stream_cmd(cmd, str(PROJECT_ROOT), env)\n"
        ),
        code(
            "# 10b. TensorBoard (после старта train в 10; можно перезапускать)\n"
            "import subprocess, sys, time, shutil\n"
            "from google.colab import output\n"
            "\n"
            "subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"protobuf<5\", \"tensorboard\"], check=True)\n"
            "\n"
            "LOGDIR = PROJECT_ROOT / \"outputs\" / \"logs\" / \"tensorboard\"\n"
            "LOGDIR.mkdir(parents=True, exist_ok=True)\n"
            "subprocess.Popen(\n"
            "    [sys.executable, \"-m\", \"tensorboard.main\", \"--logdir\", str(LOGDIR), \"--port\", \"6006\", \"--bind_all\"],\n"
            "    start_new_session=True,\n"
            "    stdout=subprocess.DEVNULL,\n"
            "    stderr=subprocess.STDOUT,\n"
            ")\n"
            "time.sleep(5)\n"
            "DRIVE_TB = DRIVE_RUN_DIR / \"tensorboard\"\n"
            "DRIVE_TB.mkdir(parents=True, exist_ok=True)\n"
            "for path in LOGDIR.rglob(\"*\"):\n"
            "    if path.is_file():\n"
            "        rel = path.relative_to(LOGDIR)\n"
            "        out = DRIVE_TB / rel\n"
            "        out.parent.mkdir(parents=True, exist_ok=True)\n"
            "        try:\n"
            "            shutil.copy2(path, out)\n"
            "        except OSError:\n"
            "            pass\n"
            "print(\"LOGDIR =\", LOGDIR, \"| копия на Drive:\", DRIVE_TB)\n"
            "output.serve_kernel_port_as_window(6006, path=\"/\")\n"
            "print(\"Если встроенный просмотр серый — открой ссылку в новой вкладке.\")"
        ),
        code(
            "# 11. EVAL\n"
            "import os, sys, subprocess\n"
            "if '_stream_cmd' not in globals():\n"
            "    import threading\n"
            "    def _stream_cmd(cmd, cwd, env):\n"
            "        p = subprocess.Popen(\n"
            "            cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,\n"
            "        )\n"
            "        def _pump():\n"
            "            if p.stdout is not None:\n"
            "                for line in p.stdout:\n"
            "                    print(line, end='', flush=True)\n"
            "        t = threading.Thread(target=_pump, daemon=True)\n"
            "        t.start()\n"
            "        rc = p.wait()\n"
            "        t.join(timeout=30)\n"
            "        if rc != 0:\n"
            "            print(f'\\n[exit {rc}] See Traceback above.', flush=True)\n"
            "            raise subprocess.CalledProcessError(rc, cmd)\n"
            "env = os.environ.copy()\n"
            "env['PYTHONPATH'] = str(PROJECT_ROOT)\n"
            "env['PYTHONUNBUFFERED'] = '1'\n"
            "cmd = [sys.executable, '-u', '-m', 'scripts.run_eval', '--config', str(PROJECT_ROOT / 'runtime_configs/eval.yaml')]\n"
            "print('EVAL CMD:', ' '.join(map(str, cmd)))\n"
            "_stream_cmd(cmd, str(PROJECT_ROOT), env)\n"
        ),
        code(
            "# 12. EXPORT + VERIFY\n"
            "import os, sys, subprocess\n"
            "if '_stream_cmd' not in globals():\n"
            "    import threading\n"
            "    def _stream_cmd(cmd, cwd, env):\n"
            "        p = subprocess.Popen(\n"
            "            cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,\n"
            "        )\n"
            "        def _pump():\n"
            "            if p.stdout is not None:\n"
            "                for line in p.stdout:\n"
            "                    print(line, end='', flush=True)\n"
            "        t = threading.Thread(target=_pump, daemon=True)\n"
            "        t.start()\n"
            "        rc = p.wait()\n"
            "        t.join(timeout=30)\n"
            "        if rc != 0:\n"
            "            print(f'\\n[exit {rc}] See Traceback above.', flush=True)\n"
            "            raise subprocess.CalledProcessError(rc, cmd)\n"
            "env = os.environ.copy()\n"
            "env['PYTHONPATH'] = str(PROJECT_ROOT)\n"
            "env['PYTHONUNBUFFERED'] = '1'\n"
            "export_cmd = [sys.executable, '-u', '-m', 'scripts.export_safetensors', '--config', str(PROJECT_ROOT / 'runtime_configs/export.yaml')]\n"
            "verify_cmd = [sys.executable, '-u', '-m', 'scripts.verify_export', '--config', str(PROJECT_ROOT / 'runtime_configs/export.yaml')]\n"
            "print('EXPORT CMD:', ' '.join(map(str, export_cmd)))\n"
            "_stream_cmd(export_cmd, str(PROJECT_ROOT), env)\n"
            "print('VERIFY CMD:', ' '.join(map(str, verify_cmd)))\n"
            "_stream_cmd(verify_cmd, str(PROJECT_ROOT), env)\n"
        ),
        code(
            "# 13. FINAL SYNC + SUMMARY\n"
            "SYNC_STOP.set()\n"
            "sync_all_to_drive()\n"
            "print('Drive checkpoint files:', sorted(p.name for p in DRIVE_CKPT_DIR.glob('*')))\n"
            "print('Drive export files:', sorted(p.name for p in DRIVE_EXPORT_DIR.glob('*')))\n"
            "print('Drive eval runs:', sorted(p.name for p in DRIVE_EVAL_DIR.glob('*')))\n"
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
