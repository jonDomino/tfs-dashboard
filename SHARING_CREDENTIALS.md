# Sharing BigQuery Credentials with Colleagues

## Option 1: Share Credentials File Securely (Easiest)

### Steps:
1. **You**: Share `meatloaf.json` via a secure channel:
   - Encrypted email
   - Password manager (1Password, LastPass, etc.)
   - Secure file sharing (Google Drive with restricted access)
   - Slack/Discord DM (less secure but convenient)

2. **Colleague**: 
   - Place `meatloaf.json` in the project root directory
   - The code will automatically find and use it

### Security Note:
- The file contains sensitive credentials
- Never commit it to git (already excluded in `.gitignore`)
- Consider rotating credentials if shared file is ever compromised

---

## Option 2: Use Environment Variable (More Flexible)

### Steps:
1. **Colleague**: Set environment variable pointing to credentials:
   
   **Windows (PowerShell):**
   ```powershell
   $env:BIGQUERY_CREDENTIALS_PATH = "C:\path\to\meatloaf.json"
   ```
   
   **Windows (Command Prompt):**
   ```cmd
   set BIGQUERY_CREDENTIALS_PATH=C:\path\to\meatloaf.json
   ```
   
   **Linux/Mac:**
   ```bash
   export BIGQUERY_CREDENTIALS_PATH="/path/to/meatloaf.json"
   ```

2. **Colleague**: Run dashboard:
   ```bash
   streamlit run streamlit_app.py
   ```

### Benefits:
- Credentials can be stored outside the project directory
- Multiple team members can use different credential files
- More secure than hardcoding paths

---

## Option 3: Use Google Cloud Application Default Credentials

If your colleague has `gcloud` CLI installed and authenticated:

### Steps:
1. **Colleague**: Authenticate with gcloud:
   ```bash
   gcloud auth application-default login
   ```

2. **You**: Grant colleague access to the service account:
   - Go to Google Cloud Console
   - IAM & Admin → Service Accounts
   - Find the service account (jondomino@meatloaf-427522.iam.gserviceaccount.com)
   - Add colleague's Google account as "Service Account User"
   - Grant "BigQuery Data Viewer" or "BigQuery Job User" role

3. **Colleague**: The code will automatically use Application Default Credentials if `meatloaf.json` is not found

---

## Option 4: Create Shared Service Account (Best for Teams)

### Steps:
1. **You**: Create a new service account in Google Cloud:
   - IAM & Admin → Service Accounts → Create Service Account
   - Name it something like "tfs-dashboard-shared"
   - Grant it BigQuery access

2. **You**: Download the JSON key and share securely (Option 1)

3. **Team**: Everyone uses the same shared credentials file

### Benefits:
- Centralized access control
- Easy to revoke access if someone leaves
- Can track usage by service account

---

## Recommended Approach

For quick setup: **Option 1** (share file securely)

For production/teams: **Option 3** or **Option 4** (IAM-based access)

---

## Security Best Practices

1. ✅ Never commit credentials to git
2. ✅ Use secure channels for sharing
3. ✅ Rotate credentials periodically
4. ✅ Use least-privilege IAM roles
5. ✅ Monitor BigQuery usage/access logs
6. ✅ Revoke access when team members leave

