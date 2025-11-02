#!/usr/bin/env python3
"""
Smart Installer with Size Alert + Confirmation
Run: python install.py
"""

import subprocess
import sys
import os
from typing import List, Dict

# === Package Size Database (approximate disk usage after install) ===
PACKAGE_SIZES = {
    # Core
    "sentence-transformers": 450,   # Includes torch + model cache
    "scikit-learn": 40,
    "python-Levenshtein": 15,
    "nltk": 30,
    "numpy": 25,

    # Optional
    "tabulate": 1,
    "matplotlib": 60,
    "python-docx": 5,
}

def format_size(mb: float) -> str:
    return f"{mb:.0f} MB" if mb >= 1 else f"{mb * 1024:.0f} KB"

def run_cmd(cmd: str):
    print(f"\n[+] Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[!] Failed: {cmd}")
        sys.exit(1)
    print(f"[Success] {cmd}")

def confirm_install(core_size: float, opt_size: float, install_opt: bool) -> bool:
    print("\n" + "="*60)
    print(" PACKAGE INSTALLATION SUMMARY")
    print("="*60)
    print(f"Core packages (required):     {format_size(core_size)}")
    if install_opt:
        print(f"Optional packages:            {format_size(opt_size)}")
    print(f"{'Total' if install_opt else 'Total (core only)'}: {format_size(core_size + (opt_size if install_opt else 0))}")
    print("-"*60)
    print("Packages will be installed via pip.")
    print("BERT model (~90 MB) will download on first run.")
    print("="*60)

    while True:
        choice = input("\nContinue with installation? (Y/n): ").strip().lower()
        if choice in ['y', 'yes', '']:
            return True
        elif choice in ['n', 'no']:
            return False
        else:
            print("Please enter Y or N.")

def main():
    print("Plagiarism Checker - Smart Installer")
    print("=" * 50)

    # === 1. Core Packages ===
    core_pkgs = ["sentence-transformers", "scikit-learn", "python-Levenshtein", "nltk", "numpy"]
    core_size = sum(PACKAGE_SIZES.get(p, 0) for p in core_pkgs)

    # === 2. Optional Packages ===
    opt_pkgs = ["tabulate", "matplotlib", "python-docx"]
    opt_size = sum(PACKAGE_SIZES.get(p, 0) for p in opt_pkgs)

    print(f"Core packages ({len(core_pkgs)}): {', '.join(core_pkgs)}")
    print(f"Optional packages ({len(opt_pkgs)}): {', '.join(opt_pkgs)} (recommended for tables & plots)")

    # === 3. Ask user ===
    install_opt = True
    while True:
        opt_choice = input(f"\nInstall optional packages? (Y/n): ").strip().lower()
        if opt_choice in ['y', 'yes', '']:
            install_opt = True
            break
        elif opt_choice in ['n', 'no']:
            install_opt = False
            break
        else:
            print("Please enter Y or N.")

    # === 4. Confirm total size ===
    if not confirm_install(core_size, opt_size, install_opt):
        print("\nInstallation cancelled by user.")
        sys.exit(0)

    # === 5. Upgrade pip ===
    print("\n[1/4] Upgrading pip...")
    run_cmd(f"{sys.executable} -m pip install --upgrade pip")

    # === 6. Install core ===
    print(f"\n[2/4] Installing {len(core_pkgs)} core packages...")
    for pkg in core_pkgs:
        run_cmd(f"{sys.executable} -m pip install {pkg}")

    # === 7. Install optional (if chosen) ===
    if install_opt:
        print(f"\n[3/4] Installing {len(opt_pkgs)} optional packages...")
        for pkg in opt_pkgs:
            run_cmd(f"{sys.executable} -m pip install {pkg}")
    else:
        print("\n[3/4] Skipping optional packages.")

    # === 8. Download NLTK data ===
    print("\n[4/4] Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        print("[Success] NLTK data ready.")
    except Exception as e:
        print(f"[!] NLTK failed: {e}")
        
    # === Final Message ===
    print("\n" + "="*60)
    print("INSTALLATION COMPLETE!")
    print("="*60)
    print("Now run:")
    print("   python plag_check.py plag_data/ --export report.csv --visualize")
    print("\nFirst run will download BERT model (~90 MB).")
    print("="*60)

if __name__ == "__main__":
    main()