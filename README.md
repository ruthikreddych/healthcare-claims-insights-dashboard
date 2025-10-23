# Healthcare Claims Insights Dashboard

This project demonstrates a complete workflow for analysing and visualising a large
set of synthetic healthcare claims. It includes a data generation pipeline, a
predictive model and a lightweight dashboard that can be hosted on any static
site provider (such as GitHub Pages). No backend server is required once the
data are pre‑aggregated, making it easy to deploy the dashboard without
maintaining a Python runtime.

## Features

- **Synthetic dataset** – `generate_dashboard_data.py` synthesises half a million
  de‑identified health insurance claims. Each record contains a provider, payer,
  service category, claim amount, status (approved/denied), submission and
  payment dates, denial reason and a readmission flag.
- **Predictive modelling** – a logistic regression model is trained to
  forecast readmission. Because the underlying label is intentionally
  generated from a non‑linear threshold on claim amounts, the model achieves
  roughly 82 % accuracy on a hold‑out set. Evaluation metrics (accuracy,
  precision, recall, F1 score and confusion matrix) are written to
  `logistic_metrics.json`.
- **Pre‑aggregated metrics** – the large raw dataset is grouped by
  provider, payer and service category. For each combination the following
  statistics are computed and stored in `claims_aggregated.json`:

  * total claims
  * approved and denied claims (and denial rate)
  * average turnaround time in days
  * actual and predicted readmission rates (weighted averages)

- **Interactive dashboard** – `dashboard.html` loads the aggregated data and
  model metrics using JavaScript. Users can filter by provider, payer and
  service category. Summary cards update in real time and bar charts (via
  Plotly) visualise denial rates and readmission rates. Model performance is
  displayed with easy‑to‑read metrics and a confusion matrix.

## Getting started

1. **Clone the repository and navigate into it**

   ```bash
   git clone https://github.com/<your‑username>/healthcare‑dashboard.git
   cd healthcare‑dashboard/healthcare_dashboard
   ```

2. **Create a virtual environment (optional but recommended)**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**

   Only a small set of libraries are required for data generation and modelling:

   ```bash
   pip install pandas numpy scikit-learn
   ```

   The dashboard itself uses only client‑side libraries (Bootstrap and Plotly)
   loaded from a CDN, so no extra installation is needed to view it.

4. **Generate the data**

   Run the data pipeline to create the raw CSV, aggregated JSON and model
   metrics. These files will be written into the current directory:

   ```bash
   python generate_dashboard_data.py
   ```

   The script will print progress messages and report the logistic regression
   accuracy. You should see an accuracy close to 0.82 on the large synthetic
   dataset.

5. **View the dashboard locally**

   Open `dashboard.html` in your web browser. Because it uses relative paths
   to load the JSON files, you can either double‑click it or serve the
   directory with a simple HTTP server:

   ```bash
   python -m http.server 8000
   # Then visit http://localhost:8000/dashboard.html
   ```

   Filtering controls along the top allow you to explore denial patterns,
   payer performance and readmission rates interactively.

## Deploying as a live dashboard

To host the dashboard via GitHub Pages:

1. Make sure `dashboard.html`, `claims_aggregated.json` and
   `logistic_metrics.json` are committed to the root of your repository (or
   whatever folder you choose to publish).
2. Enable GitHub Pages in your repository settings and select the branch
   (typically `main` or `gh‑pages`) and folder containing your dashboard files.
3. After the site is published, navigate to the provided URL – you should
   see the same interactive dashboard without any 404 errors.

The raw dataset (`claims_raw.csv`) is relatively large (~500 k rows) and is
not needed for the dashboard, so you may choose to exclude it from your
GitHub Pages deployment or add it to `.gitignore` if you do not wish to
version control it.

## License

This project uses only synthetic data and is provided under the MIT License.