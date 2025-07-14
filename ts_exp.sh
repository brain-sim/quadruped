#!/bin/bash

# ts_exp.sh - Task Spooler Experiment Wrapper
# This wrapper commits changes, runs experiments on specific commits,
# and logs diffs for easy tracking and reproducibility

set -e  # Exit on any error

# Check if we're in a git repository
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo "Error: Not in a git repository"
    exit 1
fi

# Check if task spooler arguments were provided
if [ $# -eq 0 ]; then
    echo "Error: No task spooler command provided"
    echo "Usage: $0 <ts_arguments>"
    echo "Example: $0 python train.py --lr 0.01"
    echo "Example: $0 -L gpu python train.py"
    exit 1
fi

# Create ts_logs directory if it doesn't exist
TS_LOGS_DIR="ts_logs"
mkdir -p "$TS_LOGS_DIR"

# Check if there are any changes to commit
if git diff --quiet && git diff --cached --quiet; then
    echo "No changes to commit. Using current HEAD."
    COMMIT_HASH=$(git rev-parse HEAD)
    COMMIT_MSG="ts_exp: Using existing commit $(date '+%Y-%m-%d %H:%M:%S')"
else
    # Auto-generate commit message with timestamp and command
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    COMMIT_MSG="ts_exp: $* - $TIMESTAMP"
    
    # Add all changes
    echo "Adding changes to git..."
    git add .
    
    # Commit changes
    echo "Committing changes with message: $COMMIT_MSG"
    git commit -m "$COMMIT_MSG"
    
    # Get the commit hash
    COMMIT_HASH=$(git rev-parse HEAD)
fi

echo "Commit hash: $COMMIT_HASH"

# Store current branch name to return to it later
CURRENT_BRANCH=$(git branch --show-current)

# Create diff file name
DIFF_FILE="$TS_LOGS_DIR/${COMMIT_HASH}.diff"

# Store the diff of this commit
echo "Storing diff in: $DIFF_FILE"
git show "$COMMIT_HASH" > "$DIFF_FILE"

# Create experiment log entry
LOG_FILE="$TS_LOGS_DIR/experiment_log.txt"
echo "===========================================" >> "$LOG_FILE"
echo "Experiment: $TIMESTAMP" >> "$LOG_FILE"
echo "Commit: $COMMIT_HASH" >> "$LOG_FILE"
echo "Branch: $CURRENT_BRANCH" >> "$LOG_FILE"
echo "Command: ts $*" >> "$LOG_FILE"
echo "Diff file: $DIFF_FILE" >> "$LOG_FILE"
echo "===========================================" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Checkout to the specific commit
echo "Checking out commit $COMMIT_HASH..."
git checkout "$COMMIT_HASH"

# Function to cleanup and return to original branch
cleanup() {
    echo "Returning to branch: $CURRENT_BRANCH"
    git checkout "$CURRENT_BRANCH"
}

# Set trap to ensure we return to original branch on exit
trap cleanup EXIT

# Run the task spooler command with all arguments
echo "Running task spooler command: ts $*"
echo "This will execute on commit: $COMMIT_HASH"
echo "Diff stored in: $DIFF_FILE"
ts "$@"

echo "Task spooler job queued successfully on commit $COMMIT_HASH"
echo "Check results and if good, you can cherry-pick or merge this commit: $COMMIT_HASH"
echo "Diff available at: $DIFF_FILE" 