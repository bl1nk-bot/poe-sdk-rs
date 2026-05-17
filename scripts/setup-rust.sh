#!/usr/bin/env bash
set -e

# ติดตั้ง Rustup (ใช้ --quiet ถ้าต้องการลด output)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# โหลด PATH ทันที
export PATH="$HOME/.cargo/bin:$PATH"
source "$HOME/.cargo/env"

# ติดตั้ง component ที่ต้องใช้
rustup component add rustfmt clippy

# ติดตั้ง C compiler และ build tools
# ตรวจสอบสิทธิ์ root และเรียก apt โดยไม่ต้องใช้ sudo ถ้าเป็น root อยู่แล้ว
if [ "$EUID" -eq 0 ]; then
    apt update
    apt install -y build-essential pkg-config
else
    sudo apt update
    sudo apt install -y build-essential pkg-config
fi

echo "✅ ติดตั้ง Rust และ dependencies สำเร็จ"