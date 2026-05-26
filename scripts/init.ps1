# scripts/init.ps1 - Development environment initialization for PowerShell

Write-Host "🚀 Initializing bl1z development environment..." -ForegroundColor Cyan

# 1. Configure Local Git Commit Template
git config local.commit.template .gitmessage
Write-Host "✅ Local git commit template configured." -ForegroundColor Green

# 2. Verify Rust Toolchain
if (Get-Command cargo -ErrorAction SilentlyContinue) {
    $version = cargo --version
    Write-Host "✅ Found Rust $version" -ForegroundColor Green
} else {
    Write-Host "❌ Rust not found. Please install from https://rustup.rs/" -ForegroundColor Red
}

# 3. Create required directories
New-Item -ItemType Directory -Force -Path "target/tmp" | Out-Null

Write-Host "🎉 Environment ready for development!" -ForegroundColor Cyan
