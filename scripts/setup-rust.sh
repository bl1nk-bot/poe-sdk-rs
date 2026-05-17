#!/usr/bin/env bash
set -e

# ติดตั้ง Rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# โหลด PATH
export PATH="$HOME/.cargo/bin:$PATH"
source "$HOME/.cargo/env"

# ติดตั้ง component ที่ต้องใช้
rustup component add rustfmt clippy

# ติดตั้ง C compiler และ build tools (Debian/Ubuntu)
sudo apt update
sudo apt install -y build-essential pkg-config
