# Fix Git Commands Hanging in Cursor Agents

## Problem
Git commands hang/freeze when executed by Cursor agents (non-interactive processes).

## Root Cause
`credential.helper=manager-core` (Windows Credential Manager Core) waits for interactive user input that never comes in non-interactive contexts.

## Solution

### For Python/Agent Code
Set these environment variables before running any git commands:

```python
import os

# Set these BEFORE running git commands
os.environ['GIT_TERMINAL_PROMPT'] = '0'
os.environ['GIT_ASKPASS'] = ''
os.environ['GIT_CREDENTIAL_HELPER'] = ''

# Now git commands will work without hanging
import subprocess
subprocess.run(['git', 'status'])
```

### For PowerShell/Batch Scripts
```powershell
$env:GIT_TERMINAL_PROMPT = "0"
$env:GIT_ASKPASS = ""
$env:GIT_CREDENTIAL_HELPER = ""
git status
```

## Quick Fix Command

Run this before any git commands:
```bash
# PowerShell
$env:GIT_TERMINAL_PROMPT = "0"; $env:GIT_ASKPASS = ""; $env:GIT_CREDENTIAL_HELPER = ""; git status
```

## Verification

To test if it's fixed:
1. Set the environment variables above
2. Run: `git status`
3. If it completes in <2 seconds → Fixed ✅
4. If it hangs → Still broken ❌

## Alternative: Repository-Level Fix

If you want to disable credential helper for this repo only:

```bash
git config credential.helper ""
```

Note: This may not always work. Environment variables are more reliable.

## Important Notes

- These environment variables only affect the current process/session
- Your interactive git usage (outside agents) will still work normally
- The credential helper (`manager-core`) will still work when you run git manually

