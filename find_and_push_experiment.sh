#!/bin/bash
# find_and_push_experiment.sh

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <commit_hash> [branch_name]"
    echo "Example: $0 abc123def456"
    echo "Example: $0 abc123def456 experiment-success"
    exit 1
fi

COMMIT_HASH=$1
BRANCH_NAME=${2:-$(git branch --show-current)}

# Check if commit exists
if ! git cat-file -e "$COMMIT_HASH" 2>/dev/null; then
    echo "Error: Commit $COMMIT_HASH not found"
    exit 1
fi

# Show the diff for review
echo "=== Commit Details ==="
git show --stat "$COMMIT_HASH"
echo ""

# Check if diff file exists
DIFF_FILE="ts_logs/${COMMIT_HASH}.diff"
if [ -f "$DIFF_FILE" ]; then
    echo "=== Experiment Log Entry ==="
    grep -A 10 "$COMMIT_HASH" ts_logs/experiment_log.txt || echo "No log entry found"
    echo ""
fi

# Ask for confirmation
echo "Do you want to:"
echo "1) Cherry-pick this commit to current branch ($BRANCH_NAME)"
echo "2) Create new branch from this commit"
echo "3) Reset current branch to this commit (⚠️  DESTRUCTIVE)"
echo "4) Just show the commit (no changes)"
echo "5) Cancel"

read -p "Choose option (1-5): " choice

case $choice in
    1)
        echo "Cherry-picking commit $COMMIT_HASH..."
        git cherry-pick "$COMMIT_HASH"
        echo "Pushing to $BRANCH_NAME..."
        git push origin "$BRANCH_NAME"
        echo "✅ Success! Commit cherry-picked and pushed."
        ;;
    2)
        NEW_BRANCH="experiment-$(date +%Y%m%d-%H%M%S)"
        echo "Creating new branch: $NEW_BRANCH"
        git checkout -b "$NEW_BRANCH" "$COMMIT_HASH"
        echo "Pushing new branch..."
        git push origin "$NEW_BRANCH"
        echo "✅ Success! New branch $NEW_BRANCH created and pushed."
        ;;
    3)
        echo "⚠️  WARNING: This will reset your current branch to $COMMIT_HASH"
        echo "All uncommitted changes will be lost!"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            git reset --hard "$COMMIT_HASH"
            echo "Force pushing to $BRANCH_NAME..."
            git push --force origin "$BRANCH_NAME"
            echo "✅ Branch reset and force pushed."
        else
            echo "Cancelled."
        fi
        ;;
    4)
        git show "$COMMIT_HASH"
        ;;
    5)
        echo "Cancelled."
        ;;
    *)
        echo "Invalid option."
        exit 1
        ;;
esac