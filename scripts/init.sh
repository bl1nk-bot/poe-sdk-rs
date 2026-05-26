#!/bin/bash
# scripts/init.sh - Development environment initialization for Bash

echo "🚀 Initializing bl1z development environment..."

# 1. Configure Local Git Commit Template
git config local.commit.template .gitmessage
echo "✅ Local git commit template configured."

# 2. Verify Rust Toolchain
if command -v cargo >/dev/null 2>&1; then
    echo "✅ Found Rust $(rustc --version)"
else
    echo "❌ Rust not found. Please install from https://rustup.rs/"
fi

# 3. Create required directories
mkdir -p target/tmp

# 4. Set script permissions
chmod +x scripts/*.sh

echo "🎉 Environment ready for development!"
