# PowerShell script to remove STREAMLIT_SECRETS.md from git history
# This uses git filter-branch with a Windows-compatible approach

$env:FILTER_BRANCH_SQUELCH_WARNING = "1"

# Remove the file from all commits
git filter-branch --force --index-filter `
  "git rm --cached --ignore-unmatch STREAMLIT_SECRETS.md" `
  --prune-empty --tag-name-filter cat -- --all

# Clean up backup refs
git for-each-ref --format="%(refname)" refs/original/ | ForEach-Object { git update-ref -d $_ }

# Force garbage collection
git reflog expire --expire=now --all
git gc --prune=now --aggressive

Write-Host "File removed from git history. Verify with: git log --all --oneline -- STREAMLIT_SECRETS.md"

