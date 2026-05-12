# Blacklane Quality Dashboard

Interactive Streamlit dashboard for exploring the three quality pillars (Safety, Reliability, Peace of mind) and the LSP Scorecard with A/B/C/D tier classification, built on the Q1 2019 EMEA dataset.

## What it shows

Four tabs, each driven by sidebar filters (city, transfer type, car class, VIP segment):

1. **Overview** — top-line metrics anchored in the "reclaim your time" framing: Cell B rate, completion rate, revenue at risk, rating coverage. Plus the Cell B zero-km vs some-km split (the two-stories diagnostic) and Cell B rate by city.

2. **Quality pillars** — Safety / Reliability / Peace of mind, each with its measurable KPIs from the dataset. Includes the rating-bias diagnostic (coverage by class and by VIP segment) showing where the 17% headline is structurally biased.

3. **LSP Scorecard** — the main operational tool:
   - **Tier classification** based on Cell B rate: A (<3%) Standard · B (3-8%) Monitor · C (8-15%) Improve · D (≥15%) Review
   - Sortable scorecard table with Cell B/C rates, contribution per ride, rating coverage, avg chauffeur rating
   - Tier filter chips, minimum-rides threshold slider
   - Bubble-chart visualization (volume share × Cell B rate, bubble size = revenue at risk)
   - **Per-LSP drill-down**: tier badge, airport-pickup-specific Cell B rate vs peer median, city distribution, class mix pie

4. **Chauffeur audit** — Q1's anomalous-cluster finding: chauffeurs running high Cell B rates with adjustable thresholds. Warns when flagged chauffeurs also have perfect ratings (the gaming-or-shared-account pattern).

## Tier thresholds

```
Tier A · Standard      → Cell B rate < 3%
Tier B · Monitor       → 3% ≤ Cell B < 8%
Tier C · Improve       → 8% ≤ Cell B < 15%
Tier D · Review        → Cell B ≥ 15%
```

These thresholds drive the QBR conversation per LSP. Tier A = standard partnership terms. Tier B = quarterly monitoring meetings. Tier C = 90-day improvement plan with named targets. Tier D = deactivation timeline.

## Setup (local)

```bash
# Clone or copy the dashboard folder, then:
cd dashboard
python -m venv .venv
source .venv/bin/activate           # macOS / Linux
# .venv\Scripts\activate            # Windows

pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Data file

The dashboard expects an Excel file with two sheets:

- `EMEA Q1 19 Accepted Tours`
- `EMEA Q1 19 Rejected Tours`

You can either:

- **Upload the file via the sidebar** (file uploader at the top of the left panel), or
- **Place `data.xlsx` in the same folder as `app.py`** — the dashboard will load it automatically as a fallback.

The Stockholm currency correction (SEK → EUR ÷ 11) is applied at load time. All downstream metrics use corrected values.

## Deployment options

**Streamlit Community Cloud** (free, fastest)
1. Push this folder to a GitHub repo (any visibility)
2. Sign in at https://streamlit.io/cloud with the same GitHub account
3. New app → point at `app.py` in the repo
4. Upload `data.xlsx` to the repo OR add a Secrets entry, OR have viewers upload via the sidebar
5. Get a shareable `*.streamlit.app` URL

**Internal hosting** (Nexio infrastructure or any VM)
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```
Put it behind a reverse proxy (nginx, Caddy) with basic auth or your existing SSO.

**Docker**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

## Customization notes

The most likely places to modify:

- **Tier thresholds** — top of `app.py`, constants `TIER_D_THRESHOLD`, `TIER_C_THRESHOLD`, `TIER_B_THRESHOLD`. Change to match how the Blacklane ops team would actually score LSPs.
- **Color palette** — also at the top, `COLOR_*` constants. Currently operational (red/amber/green/ink) rather than brand colors. Swap to Blacklane navy if preferred.
- **Currency correction** — `SEK_TO_EUR = 11`. If the field-semantics conversation with finance reveals other markets with similar issues, add them in `load_data()`.
- **Minimum-rides thresholds** — the scorecard defaults to ≥100 rides per LSP; the chauffeur audit defaults to ≥30 rides per chauffeur. Both are user-adjustable via sliders in-app.

## Caveats baked into the dashboard

The dashboard surfaces several data-quality flags from the analysis:

- **Currency correction**: applied at ingestion. Stockholm SEK fields divided by 11.
- **Cell B definitional split**: the dashboard separates Cell B (pickup logged, no-show — operational) from Cell C (no pickup, no-show — true chauffeur no-show) and labels them distinctly.
- **Zero-km share of Cell B**: surfaced as a top-line metric because it's the cleanest operational signature ("chauffeur waited, customer never appeared").
- **Rating coverage bias**: the rating coverage metric is shown alongside VIP/non-VIP breakdown so the structural 2.1% VIP gap is visible, not hidden in the 17% aggregate.
- **Shared chauffeur-name pattern**: the chauffeur audit tab includes an explicit caveat about identity verification before deactivation decisions.

## Field-definition uncertainty (still pending verification)

The contribution-per-ride calculation uses `Avg. Gross Revenue − Avg. Winning Price`. If the field-semantics conversation with finance reveals these aren't post-event settlement amounts, the contribution figures need re-interpretation. The dashboard surfaces the headline numbers; the operational decisions that depend on them (like the contractual lever in Goal 1) should wait until finance confirms.
