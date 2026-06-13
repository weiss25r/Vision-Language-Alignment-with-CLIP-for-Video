#!/usr/bin/env python3
"""
install_torch.py
----------------
Installa torch, torchvision e torchaudio con la variante corretta
per la piattaforma corrente (CUDA 12.x, CUDA 11.8, CPU, Apple MPS).

Uso:
    python install_torch.py            # auto-detect
    python install_torch.py --cuda 12  # forza CUDA 12.x
    python install_torch.py --cuda 11  # forza CUDA 11.8
    python install_torch.py --cpu      # forza CPU-only
"""

import subprocess
import sys
import platform
import argparse

# ── Versioni target ──────────────────────────────────────────────────────────
TORCH_VERSION = "2.3.1"          # cambia qui se vuoi un'altra versione
TORCH_PACKAGES = [
    f"torch=={TORCH_VERSION}",
    f"torchvision",
    f"torchaudio",
]

# ── Index URLs per variante ──────────────────────────────────────────────────
INDEX_URLS = {
    "cu121": "https://download.pytorch.org/whl/cu121",   # CUDA 12.1
    "cu118": "https://download.pytorch.org/whl/cu118",   # CUDA 11.8
    "cpu":   "https://download.pytorch.org/whl/cpu",     # CPU-only
    # macOS MPS: usa l'indice default PyPI (torch include già MPS)
    "mps":   None,
}


def detect_cuda_version() -> str | None:
    """Restituisce '12', '11', o None se CUDA non trovata."""
    try:
        out = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.DEVNULL).decode()
        if "release 12" in out:
            return "12"
        if "release 11" in out:
            return "11"
    except FileNotFoundError:
        pass

    try:
        out = subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL).decode()
        if "CUDA Version: 12" in out:
            return "12"
        if "CUDA Version: 11" in out:
            return "11"
    except FileNotFoundError:
        pass

    return None


def choose_variant(args: argparse.Namespace) -> str:
    """Sceglie la variante in base agli argomenti o all'auto-detect."""
    if args.cpu:
        print("[install_torch] Modalità forzata: CPU-only")
        return "cpu"

    if args.cuda == "12":
        print("[install_torch] Modalità forzata: CUDA 12.x")
        return "cu121"

    if args.cuda == "11":
        print("[install_torch] Modalità forzata: CUDA 11.8")
        return "cu118"

    # macOS → MPS (nessun indice speciale)
    if platform.system() == "Darwin":
        print("[install_torch] macOS rilevato → variante MPS (default PyPI)")
        return "mps"

    # Auto-detect CUDA su Linux/Windows
    cuda = detect_cuda_version()
    if cuda == "12":
        print("[install_torch] CUDA 12.x rilevata → cu121")
        return "cu121"
    if cuda == "11":
        print("[install_torch] CUDA 11.8 rilevata → cu118")
        return "cu118"

    print("[install_torch] Nessuna GPU/CUDA rilevata → CPU-only")
    return "cpu"


def install(variant: str) -> None:
    index_url = INDEX_URLS[variant]

    cmd = [sys.executable, "-m", "pip", "install"] + TORCH_PACKAGES

    if index_url:
        cmd += ["--index-url", index_url]

    print(f"\n[install_torch] Eseguo: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("\n[install_torch] ❌ Installazione fallita.")
        sys.exit(result.returncode)
    else:
        print("\n[install_torch] ✅ torch installato correttamente.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Installa PyTorch per la piattaforma corrente.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cuda", choices=["11", "12"], help="Forza versione CUDA (11 o 12)")
    group.add_argument("--cpu", action="store_true", help="Forza installazione CPU-only")
    args = parser.parse_args()

    variant = choose_variant(args)
    install(variant)


if __name__ == "__main__":
    main()
