"""
generate_dashboard_data.py
-------------------------

This script synthesises a large mock healthcare claims dataset, trains a
logistic regression model to predict the likelihood of patient readmission,
and then produces an aggregated data file for a simple dashboard.

The resulting aggregated dataset contains summary metrics by provider,
payer and service category, such as claim counts, denial rates,
average turnaround time and both actual and predicted readmission rates.
It also writes a JSON file containing evaluation metrics for the trained
logistic regression model.

The primary objective of this script is to support a lightweight, static
web-based dashboard that can be hosted via GitHub Pages or a similar
service. Because the dashboard operates entirely in the browser it uses
pre‑aggregated data; individual claim records are not shipped to the
client in order to keep payload sizes manageable.

Usage::

    python generate_dashboard_data.py

This will create the following files in the current working directory:

    * claims_raw.csv            – the full synthetic claims dataset (500k rows)
    * claims_aggregated.json    – aggregated metrics used by the dashboard
    * logistic_metrics.json     – evaluation metrics for the trained model

Both JSON files are consumed by the accompanying dashboard.html and
should be committed to your repository alongside that page.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split


def generate_dataset(
    n_claims: int = 500_000,
    n_providers: int = 30,
    n_payers: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a synthetic claims dataset.

    The generated dataset roughly mirrors the structure of a typical
    healthcare claims database. Providers, payers and service categories
    are randomly assigned from fixed lists. Claim amounts follow a
    log‑normal distribution to simulate the skewed nature of healthcare
    costs. Submission dates span from the start of 2024 to the end of
    September 2025, and payment dates for approved claims occur within a
    month of submission.

    A readmission flag is synthesised using a logistic function of
    log‑scaled claim amount and service category effects. This creates a
    non‑linear relationship that logistic regression can learn.

    Parameters
    ----------
    n_claims : int
        Number of claim records to generate.
    n_providers : int
        Number of unique providers.
    n_payers : int
        Number of unique payers.
    random_state : int
        Seed for reproducible randomness.

    Returns
    -------
    pd.DataFrame
        A DataFrame with synthetic claim records.
    """
    rng = np.random.default_rng(random_state)

    # Define categorical values
    providers = [f"Provider {i + 1}" for i in range(n_providers)]
    # Use uppercase letters for payers (e.g. A, B, …)
    payers = [f"Payer {chr(ord('A') + i)}" for i in range(n_payers)]
    service_categories = [
        "Inpatient",
        "Outpatient",
        "Emergency",
        "Preventive",
        "Diagnostic",
        "Surgery",
    ]

    # Create claim identifiers
    claim_id = np.arange(n_claims)
    provider_id = rng.choice(providers, size=n_claims)
    payer = rng.choice(payers, size=n_claims)
    service_category = rng.choice(service_categories, size=n_claims)

    # Generate claim amounts using a log‑normal distribution. The mean
    # parameter of 9 yields values around exp(9) ≈ 8k but with heavy
    # dispersion to reflect the wide variation in medical bills.
    claim_amount = rng.lognormal(mean=9.0, sigma=0.6, size=n_claims)

    # Assign submission dates uniformly between Jan 1, 2024 and Sep 30, 2025
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 9, 30)
    total_seconds = int((end_date - start_date).total_seconds())
    # Random seconds offset for each claim
    offset_seconds = rng.integers(low=0, high=total_seconds, size=n_claims)
    claim_submission_date = np.array(
        [start_date + timedelta(seconds=int(o)) for o in offset_seconds],
        dtype='datetime64[ns]',
    )

    # Determine whether each claim is approved or denied (80% approval rate)
    approved_mask = rng.random(n_claims) < 0.8
    claim_status = np.where(approved_mask, "Approved", "Denied")

    # Payment dates: approved claims get a payment date between 1 and 30 days after submission
    payment_days = rng.integers(low=1, high=31, size=n_claims)
    claim_payment_date = np.empty(n_claims, dtype='datetime64[ns]')
    for idx, approved in enumerate(approved_mask):
        if approved:
            claim_payment_date[idx] = claim_submission_date[idx] + np.timedelta64(
                payment_days[idx], 'D'
            )
        else:
            # Use NaT (Not a Time) to represent missing payment date
            claim_payment_date[idx] = np.datetime64('NaT')

    # Denial reason assigned only when claim is denied
    denial_reasons_list = [
        "Incomplete Information",
        "Coding Error",
        "Policy Not Covered",
        "Eligibility Issue",
        "Duplicate Claim",
    ]
    denial_reason = np.where(
        approved_mask,
        "",  # no denial reason for approved claims
        rng.choice(denial_reasons_list, size=n_claims),
    )

    # ---------------------------------------------------------------------
    # Determine readmission flag
    #
    # For this dataset we intentionally construct the readmission target
    # to have a simple non‑linear dependency on claim amount that a
    # logistic regression using only linear features cannot perfectly
    # capture. Claims with an amount below the median of all claim
    # amounts have a low chance (5%) of readmission, whereas claims
    # above or equal to the median carry a high chance (85%). This
    # produces a mixture of two populations and yields a logistic
    # regression accuracy of roughly ~82% on large samples.
    median_amount = np.median(claim_amount)
    # Vector of probabilities: 0.05 for low‑amount claims, 0.85 for high
    prob_readmission = np.where(claim_amount < median_amount, 0.05, 0.85)
    readmission = (rng.random(n_claims) < prob_readmission).astype(int)

    # Assemble DataFrame
    df = pd.DataFrame(
        {
            "claim_id": claim_id,
            "provider_id": provider_id,
            "payer": payer,
            "service_category": service_category,
            "claim_amount": claim_amount,
            "claim_status": claim_status,
            "claim_submission_date": claim_submission_date,
            "claim_payment_date": claim_payment_date,
            "denial_reason": denial_reason,
            "readmission": readmission,
        }
    )

    return df


def train_logistic_model(df: pd.DataFrame) -> tuple[dict, LogisticRegression]:
    """Train a logistic regression on the synthetic dataset.

    The features used are claim amount and the one‑hot encoded service
    category. No claim status or date information is fed to the model to
    avoid leakage. The target is the readmission flag.

    Returns
    -------
    metrics : dict
        Dictionary containing accuracy, precision, recall, f1 score and
        confusion matrix (as a nested list) of the trained model on a
        hold‑out test set.
    model : LogisticRegression
        The trained scikit‑learn logistic regression estimator.
    """
    # Prepare feature matrix X and target y
    X = df[["claim_amount", "service_category"]].copy()
    # One‑hot encode the service_category column
    X = pd.get_dummies(X, columns=["service_category"], drop_first=False)
    y = df["readmission"]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Instantiate and fit logistic regression
    model = LogisticRegression(max_iter=200, solver="lbfgs")
    model.fit(X_train, y_train)

    # Predictions on test set
    y_pred = model.predict(X_test)
    # Derive probability estimates for potential use in aggregated metrics
    y_prob = model.predict_proba(X_test)[:, 1]

    # Compute evaluation metrics
    acc = float(accuracy_score(y_test, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.astype(int).tolist(),
    }
    return metrics, model


def aggregate_dashboard_data(df: pd.DataFrame, model: LogisticRegression) -> pd.DataFrame:
    """Aggregate claim and model data for dashboard consumption.

    New columns for predicted readmission probability and class are added
    before grouping. Turnaround days are computed from the difference
    between payment and submission dates (missing values yield NaN).

    The resulting DataFrame contains one row per provider/payer/service
    category combination with the following metrics:

        * total_claims
        * approved_claims
        * denied_claims
        * denial_rate
        * avg_turnaround_days
        * actual_readmission_rate
        * predicted_readmission_rate

    Returns
    -------
    pd.DataFrame
        Aggregated metrics per provider/payer/service category.
    """
    # Create features to pass into the model for prediction on all rows
    X_full = df[["claim_amount", "service_category"]].copy()
    X_full = pd.get_dummies(X_full, columns=["service_category"], drop_first=False)
    # Align columns with those used during training
    # If any category is missing, add it with zeros
    for col in model.feature_names_in_:
        if col not in X_full.columns:
            X_full[col] = 0
    X_full = X_full[model.feature_names_in_]
    # Predicted probabilities and binary predictions
    pred_prob = model.predict_proba(X_full)[:, 1]
    pred_class = (pred_prob >= 0.5).astype(int)

    df = df.copy()
    df["pred_readmission_prob"] = pred_prob
    df["pred_readmission"] = pred_class
    # Compute turnaround days; NaT values yield NaN when casting to float
    df["turnaround_days"] = (
        df["claim_payment_date"] - df["claim_submission_date"]
    ).dt.days.astype(float)

    # Perform groupby aggregation
    agg = (
        df.groupby(["provider_id", "payer", "service_category"])
        .agg(
            total_claims=("claim_id", "count"),
            approved_claims=("claim_status", lambda x: (x == "Approved").sum()),
            denied_claims=("claim_status", lambda x: (x == "Denied").sum()),
            denial_rate=("claim_status", lambda x: (x == "Denied").mean()),
            avg_turnaround_days=("turnaround_days", "mean"),
            actual_readmission_rate=("readmission", "mean"),
            predicted_readmission_rate=("pred_readmission", "mean"),
        )
        .reset_index()
    )

    # Ensure numeric columns are Python float or int for JSON serialisation
    num_cols = [
        "total_claims",
        "approved_claims",
        "denied_claims",
        "denial_rate",
        "avg_turnaround_days",
        "actual_readmission_rate",
        "predicted_readmission_rate",
    ]
    for col in num_cols:
        agg[col] = agg[col].astype(float) if agg[col].dtype.kind != 'O' else agg[col]

    return agg


def main() -> None:
    """Entry point for dataset generation and model training."""
    # Generate synthetic claims
    print("Generating synthetic claims dataset...")
    df = generate_dataset()
    print(f"Generated {len(df)} claims")

    # Train logistic regression model
    print("Training logistic regression model...")
    metrics, model = train_logistic_model(df)
    print(f"Model accuracy: {metrics['accuracy']:.3f}")

    # Aggregate data for dashboard
    print("Aggregating data for dashboard...")
    agg = aggregate_dashboard_data(df, model)
    print(f"Aggregated into {len(agg)} rows")

    # Write raw dataset to CSV (for reproducibility)
    raw_path = Path("claims_raw.csv")
    print(f"Saving raw dataset to {raw_path}...")
    df.to_csv(raw_path, index=False)

    # Write aggregated dataset to JSON
    agg_path = Path("claims_aggregated.json")
    print(f"Saving aggregated data to {agg_path}...")
    # Convert to dictionary for JSON serialisation
    agg_records = agg.to_dict(orient="records")
    with agg_path.open("w", encoding="utf-8") as f:
        json.dump(agg_records, f, indent=2)

    # Write model metrics to JSON
    metrics_path = Path("logistic_metrics.json")
    print(f"Saving logistic model metrics to {metrics_path}...")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Data generation complete.")


if __name__ == "__main__":
    main()