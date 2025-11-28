# Git Setup Instructions

## For You (Sharing the Project)

### 1. Initialize Git Repository

```bash
cd C:\Users\jonDomino\Desktop\Projects\models\cbb_2025\sandbox\dashboard
git init
```

### 2. Add All Files (meatloaf.json will be excluded by .gitignore)

```bash
git add .
```

### 3. Make Initial Commit

```bash
git commit -m "Initial commit: TFS Kernel Dashboard"
```

### 4. Create Remote Repository

Choose one:
- **GitHub**: Create a new repository at github.com
- **GitLab**: Create a new project at gitlab.com
- **Bitbucket**: Create a new repository at bitbucket.org

### 5. Add Remote and Push

```bash
# Replace with your actual repository URL
git remote add origin https://github.com/yourusername/tfs-dashboard.git
git branch -M main
git push -u origin main
```

## For Your Colleague (Getting the Project)

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/tfs-dashboard.git
cd tfs-dashboard
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Credentials

Create `meatloaf.json` in the root directory with BigQuery service account credentials.

### 4. Run the Dashboard

```bash
streamlit run streamlit_app.py
```

## Important Security Note

⚠️ **Never commit `meatloaf.json`** - it contains sensitive credentials. The `.gitignore` file is configured to exclude it, but double-check before pushing:

```bash
git status
# Make sure meatloaf.json is NOT listed
```

If it appears, remove it:
```bash
git rm --cached meatloaf.json
git commit -m "Remove credentials file"
```

