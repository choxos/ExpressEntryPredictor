#!/bin/bash

# 🚀 Automatic GitHub Push Script for Express Entry Predictor
# This script automatically commits and pushes changes to GitHub

echo "🔍 Checking for changes..."

# Check if there are any changes
if [[ -z $(git status --porcelain) ]]; then
    echo "✅ No changes to commit. Repository is up to date."
    exit 0
fi

echo "📁 Changes detected. Processing..."

# Add all changes
git add .

# Get current timestamp
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

# Default commit message if none provided
if [ -z "$1" ]; then
    COMMIT_MSG="🔄 Auto-update: Changes on $TIMESTAMP

📝 Modified files:
$(git diff --cached --name-only | head -10 | sed 's/^/• /')

🤖 Automatically committed by auto_push.sh"
else
    COMMIT_MSG="$1"
fi

echo "💬 Commit message:"
echo "$COMMIT_MSG"
echo ""

# Commit changes
git commit -m "$COMMIT_MSG"

if [ $? -eq 0 ]; then
    echo "✅ Changes committed successfully"
    
    # Push to GitHub
    echo "🚀 Pushing to GitHub..."
    git push origin main
    
    if [ $? -eq 0 ]; then
        echo "🎉 Successfully pushed to GitHub!"
        echo "📊 Repository updated: https://github.com/choxos/ExpressEntryPredictor"
        
        # Show summary
        echo ""
        echo "📈 PUSH SUMMARY:"
        echo "• Timestamp: $TIMESTAMP"
        echo "• Branch: main"
        echo "• Files changed: $(git diff HEAD~1 --name-only | wc -l | xargs)"
        echo "• Lines added/removed: $(git diff HEAD~1 --stat | tail -1)"
    else
        echo "❌ Failed to push to GitHub"
        exit 1
    fi
else
    echo "❌ Failed to commit changes"
    exit 1
fi 