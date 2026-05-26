#!/bin/bash
# scripts/release.sh - Version and Changelog updater
# Usage: ./scripts/release.sh <phase_number> "<summary>"

PHASE=$1
SUMMARY=$2

if [[ -z "$PHASE" ]] || [[ -z "$SUMMARY" ]]; then
    echo "Usage: ./scripts/release.sh <phase_number> \"<summary>\""
    echo "Example: ./scripts/release.sh 8 \"Implement access chaining support\""
    exit 1
fi

VERSION="0.2.$PHASE"
DATE=$(date +%Y-%m-%d)

echo "📦 Preparing Release version $VERSION (Phase $PHASE)..."

# 1. Update Cargo.toml
sed -i "s/^version = \".*\"/version = \"$VERSION\"/" Cargo.toml
echo "✅ Updated Cargo.toml to $VERSION"

# 2. Update CHANGELOG.md
# Insert new entry after ## [Unreleased] or at the top
CHANGELOG_ENTRY="## [$VERSION] - $DATE\n\n### Added\n- (Phase $PHASE): $SUMMARY\n"

if grep -q "## \[Unreleased\]" CHANGELOG.md; then
    sed -i "/## \[Unreleased\]/a \\\n$CHANGELOG_ENTRY" CHANGELOG.md
else
    # If no Unreleased header, insert after the main header
    sed -i "/# Changelog/a \\\n$CHANGELOG_ENTRY" CHANGELOG.md
fi
echo "✅ Updated CHANGELOG.md"

# 3. Commit Report
echo -e "\n📝 Recent commits for Phase $PHASE:"
git log --oneline --grep="Phase $PHASE" -n 10

echo -e "\n🚀 Done! Run 'cargo check' to verify the environment."
