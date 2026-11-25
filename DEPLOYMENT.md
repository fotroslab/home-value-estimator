# Deployment Guide — Home Value Range Estimator

> **Status:** Not yet deployed (no push without your approval). This guide is click-through, browser-only — no VS Code or Git CLI required if you use the GitHub web UI + Streamlit Community Cloud web UI.

## Option A — Streamlit Community Cloud (recommended, free)

This app is lightweight (no GPU, models < 2 MB, `requirements.txt` pinned) and fits Streamlit Community Cloud’s free tier.

### Current limits (verified 2026-11-25 from Streamlit docs)

- Free for personal/portfolio apps with a GitHub account.
- Apps sleep after inactivity and wake on next visit (cold start 30–60 s). No 24/7 SLA.
- Resource quota: ~1 GB RAM, 1 vCPU shared. Our app uses ~300 MB peak.
- Build from `requirements.txt` or `pyproject.toml`; Python 3.10–3.13 supported.
- Custom domain not included on free tier.
- Public apps are visible to anyone with the URL; do not commit secrets or private data (we don’t — dataset is CC0, no `secrets.toml`).

If quotas or terms have changed since this doc was written, check https://docs.streamlit.io/streamlit-community-cloud and update this file.

### Steps (browser only)

1. **Push to GitHub (web UI)**
   - Create a new public repo on GitHub (e.g., `home-value-estimator-pilot`). Do not initialize with a README if you already have this folder.
   - On your local machine (or via GitHub web “Add file → Upload files”), upload the contents of `home-value-estimator-pilot/` (not the outer folder). Ensure `app.py` is at repo root, `requirements.txt` is present, and `models/` + `data/sample_properties.csv` are included (they are small and allowed; `.gitignore` already excludes `data/raw/`).
   - Commit via the web UI.

   *If you prefer the CLI:*
   ```bash
   git init
   git add .
   git commit -m "Initial: Allegheny County pilot"
   git branch -M main
   git remote add origin https://github.com/<you>/home-value-estimator-pilot.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Community Cloud**
   - Go to https://share.streamlit.io/ → Sign in with GitHub.
   - Click “Create app” → “Deploy a public app from GitHub”.
   - Select your repo and branch (`main`), set Main file path to `app.py`.
   - Click “Deploy”. First build takes 3–6 minutes (installs `requirements.txt`).
   - When green, open the URL (e.g., `https://<you>-home-value-estimator-pilot.streamlit.app`) in an incognito window to verify — no login should be required for the demo.

3. **Verify**
   - Load a preset, click “Estimate price range”, check that low/central/high render and SHAP loads.
   - Check “Model performance & diagnostics” expander loads metrics, SHAP bar, and residual plot.
   - Test an invalid input (living area 100) — you should see a validation error, not a price.

4. **Share**
   - Add the live URL to the top of `README.md` and to your portfolio site. Keep the `models/metrics.json` link so viewers can see the holdout numbers.

### Updating after deploy

- Push to `main` on GitHub → Streamlit auto-redeploys (1–2 min). No redeploy button needed.

---

## Option B — Hugging Face Spaces (Streamlit)

- Also free, good alternative if you already use HF.
- Create a new Space → SDK “Streamlit” → upload files or connect GitHub.
- `requirements.txt` is auto-installed. Add `app.py` as entry point.
- Same limits: sleeps on inactivity on free tier.

## Option C — Render / Railway free tier

Possible but Streamlit Community Cloud is simplest for this stack. Use Render only if you need a persistent custom domain on free tier.

## What NOT to do

- Do not use **Publish to web** (Power BI) — this is not a PBIX dashboard.
- Do not call a local CSV a “PBIX” or an `html` export a “native Power BI dataset”.
- Do not claim 24/7 uptime or push to my GitHub without explicit approval.

## Local parity check before you deploy

```bash
pip install -r requirements.txt
python train.py        # regenerates models/metrics.json
pytest tests/ -v       # all 6 should pass
streamlit run app.py   # test locally at http://localhost:8501
```

If any step fails, fix before deploying — the cloud build will fail the same way.

## After you deploy

Send me the live URL and I will add it to `README.md` + `app.py` footer + portfolio index, and re-run a no-login check.
