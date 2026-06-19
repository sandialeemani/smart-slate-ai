# AI-Powered Operational Management Platform — Slate Kitchen & Café

CSCI323 — Modern Artificial Intelligence · University of Wollongong in Dubai · Spring 2026
Tutor: Dr. Abdalla Elnikiti · Submitted: 21 June 2026

An AI-driven operational management platform built for a UAE café, combining demand forecasting, inventory management, and dynamic pricing into a single decision-support pipeline. Developed using synthetic data modelled on a real café's menu, pricing, and operating patterns.

---

## Overview

Slate Kitchen & Café faced three linked operational problems: unpredictable demand making staffing and prep difficult, reactive (rather than predictive) inventory restocking, and full-price perishables being written off at closing time instead of sold at a markdown. This project addresses all three with one integrated pipeline:

Historical Sales Data

↓

Demand Forecasting Model  (Logistic Regression / KNN — busy vs. quiet day classification)

↓

Inventory Management System  (Low-Stock Prediction · Expiry-Risk Prediction · Supplier Order Recommendation)

↓

Dynamic Pricing Engine  (time-based markdown cascade on expiry-risk items)

↓

Management Dashboard

## Results at a glance

| Metric | Result |
|---|---|
| Demand classification accuracy | 94.3% (F1 = 0.93) |
| Low-stock alerts generated | 2,267 |
| Inventory expiry-risk flags | 290 |
| Overall ingredient wastage rate | 2.40% |
| Waste units avoided (dynamic pricing) | 2,758 |
| Revenue recovered via discounting | AED 34,809 |
| Understaffed shifts identified | 92 |

Full methodology, evaluation, limitations, and ethical considerations are in [`/report`](./report).

## Repository structure

.

├── notebooks/

│   ├── 01_demand_forecasting.ipynb       # Logistic Regression + KNN classification

│   ├── 02_inventory_management.ipynb     # Low-stock & expiry-risk models, reorder logic

│   └── 03_dynamic_pricing.ipynb          # Discount-tier pricing engine

├── data/

│   ├── sales_transactions.csv            # 64,855 synthetic transaction records

│   ├── staffing.csv                      # 1,448 daily staffing records

│   └── inventory_records.csv             # 8,326 daily inventory records

├── report/

│   └── CSCI323_Project_Final_Report.pdf

├── presentation/

│   └── slate-kitchen-deck.pptx

└── README.md

*(Adjust paths above to match your actual folder names if they differ.)*

## Tech stack

- **Python 3** — pandas, numpy, scikit-learn, matplotlib, seaborn
- **Models** — Logistic Regression, K-Nearest Neighbors, Random Forest Classifier
- **Environment** — Google Colab (file upload via `files.upload()`)

## Getting started

1. Clone the repo:
```bash
   git clone https://github.com/<your-org>/<your-repo>.git
   cd <your-repo>
```
2. Open any notebook in `notebooks/` in Google Colab or Jupyter.
3. Upload the corresponding CSV(s) from `data/` when prompted (or mount the repo if running locally).
4. Run all cells top to bottom — each notebook is self-contained and reproduces the metrics reported in the final report.

Notebooks should be run in order (`01` → `02` → `03`), since the inventory and pricing modules consume the busy-day classification output from the demand forecasting model.

## Methodology summary

- **Demand Forecasting:** Daily revenue/covers aggregated from 6 months of transaction + staffing data, 18 engineered features (calendar flags, revenue/covers lags, one-hot encoded day name), median-split binary target, 80/20 chronological train-test split (`shuffle=False` to prevent leakage), `StandardScaler` fit on training data only.
- **Inventory Management:** Random Forest classifiers for low-stock and expiry-risk prediction, fed by the demand model's busy-day output; reorder quantities calculated as `(Expected Consumption + Safety Stock) − Current Stock`.
- **Dynamic Pricing:** Rules-based markdown cascade (20% at 6 PM → 30% at 7 PM → 40% at 8 PM) triggered on expiry-risk-flagged perishable bakery items.

See **Appendix B** of the final report for the full data preprocessing summary.

## Team & roles

| Name | Role |
|---|---|
| Ryan Sethi | Project Lead |
| Akram Kamhawi | Demand Forecasting Model |
| Ahmed Mohammed Sifat Ahmed | Inventory Management System |
| David Samir | Dynamic Pricing Engine |
| Sandiya Kumari | Evaluation Report & Presentation Lead |

## Limitations

- Dataset is synthetic and spans only 6 months — insufficient for full annual seasonality (Ramadan, summer slowdown, December peak only partially represented).
- No external signals (weather, competitor pricing, local events) incorporated into the forecasting model.
- Inventory reorder thresholds are fixed in the synthetic data rather than dynamically optimised.
- See Section 13 of the final report for the complete limitations discussion.

## License

Academic coursework submitted for CSCI323, University of Wollongong in Dubai. Code is shared here for reproducibility and grading purposes — add an explicit license (e.g. MIT) if you intend this to be reused beyond the course.

## Acknowledgements

Developed for CSCI323 — Modern Artificial Intelligence, Spring 2026, under Dr. Abdalla Elnikiti. See the final report's References section for all literature cited.
