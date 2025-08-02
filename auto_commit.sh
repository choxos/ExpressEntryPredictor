#!/bin/bash

# Express Entry Predictor - Auto Commit Script
# This script automatically commits and pushes changes to GitHub

echo "🚀 Express Entry Predictor - Auto Commit"
echo "========================================"

# Check if there are any changes
if [[ -z $(git status --porcelain) ]]; then
    echo "✅ No changes to commit. Repository is up to date."
    exit 0
fi

# Show current status
echo "📋 Current git status:"
git status --short

# Add all changes
echo ""
echo "📝 Adding all changes..."
git add .

# Get a timestamp for the commit message
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Create a commit message based on what changed
CHANGED_FILES=$(git diff --cached --name-only | wc -l | tr -d ' ')
COMMIT_MSG="Auto-commit: Updated $CHANGED_FILES files - $TIMESTAMP

Changes include:
$(git diff --cached --name-status | head -10)
$(if [ $(git diff --cached --name-only | wc -l) -gt 10 ]; then echo "... and more files"; fi)

Auto-generated commit for Express Entry Predictor updates."

# Commit the changes
echo "💾 Committing changes..."
git commit -m "$COMMIT_MSG"

# Push to GitHub
echo "⬆️  Pushing to GitHub..."
if git push origin main; then
    echo ""
    echo "✅ SUCCESS! All changes have been committed and pushed to GitHub."
    echo "🌐 Repository: https://github.com/choxos/ExpressEntryPredictor"
    echo "📊 Commit: $(git rev-parse --short HEAD)"
else
    echo ""
    echo "❌ ERROR: Failed to push to GitHub. Please check your connection and try again."
    exit 1
fi

echo ""
echo "🎉 Auto-commit completed successfully!" 