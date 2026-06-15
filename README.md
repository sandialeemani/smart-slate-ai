# Smart Operations AI — Slate Kitchen & Café
### CSCI323 Modern Artificial Intelligence | UOWD | Spring 2026

AI-powered system for demand forecasting, inventory management,
and dynamic pricing at a UAE café. Built by a group of 5 students
as the final project for CSCI323.

---

## Team

| Member | Role |
|--------|------|
| Member 1 | Project lead, problem definition, executive summary |
| Member 2 | Demand forecasting model (LSTM/XGBoost) |
| Member 3 | Inventory management & stock alert system |
| Member 4 | Dynamic pricing engine |
| Sandiya Kumari (8990682) | Evaluation, report, presentation, repo |

---

## Project Structure
smart-slate-ai/

│

├── data/

│   ├── slate_sales_transactions.csv

│   ├── slate_staffing.csv

│   ├── slate_inventory_daily.csv

│   └── slate_dynamic_pricing.csv

│

├── notebooks/

│   ├── 01_demand_forecasting.ipynb      # Member 2

│   ├── 02_inventory_management.ipynb   # Member 3

│   └── 03_dynamic_pricing.ipynb        # Member 4

│

├── src/

│   ├── forecasting/

│   │   └── model.py

│   ├── inventory/

│   │   └── alert_system.py

│   └── pricing/

│       └── engine.py

│

├── results/

│   ├── evaluation_metrics.csv

│   ├── charts/

│   └── confusion_matrices/

│

├── report/

│   └── CSCI323_Slate_Kitchen_Report.pdf

│

├── presentation/

│   └── CSCI323_Slate_Kitchen_Slides.pptx

│

├── requirements.txt

└── README.md

---

## Problem Statement

Slate Kitchen & Café faces three operational challenges common to UAE
F&B businesses:

- **Perishable waste** — bakery and dairy items discarded unsold at day-end
- **Staffing mismatches** — demand spikes not predicted, leading to
  understaffed shifts
- **Reactive inventory** — ingredients running out before reorder is triggered

This system addresses all three with AI models trained on 6 months of
operational data (November 2024 – April 2025).

---

## Models & Results

### Module 1 — Demand Forecasting (LSTM/XGBoost)
- **Task:** Predict hourly transaction volume and daily footfall
- **Test R²:** 0.85 | **RMSE:** 24.1 transactions/hour
- **Peak hours detected:** 10 AM (8,624 transactions) and 1 PM (9,200)

### Module 2 — Inventory Management
- **Task:** Low-stock alerting and expiry risk classification
- **Alerts raised:** 2,267 low-stock events across 181 days
- **Wastage rate:** 2.40% of ingredient throughput
- **Critical finding:** Chicken Breast understocked on 103/181 days

### Module 3 — Dynamic Pricing Engine
- **Task:** Time-based discount rules for perishable bakery items
- **Waste units avoided:** 2,758 over 6 months
- **Revenue recovered:** AED 34,809
- **Trigger rate:** 74.5% of evaluated pricing slots

---

## Setup & Installation

### Requirements

- Python 3.9+
- See `requirements.txt` for full dependency list

```bash
# Clone the repository
git clone https://github.com/[your-team-username]/smart-slate-ai.git
cd smart-slate-ai

# Install dependencies
pip install -r requirements.txt
```

### Run the notebooks

```bash
jupyter notebook notebooks/
```

Open each notebook in order:
1. `01_demand_forecasting.ipynb`
2. `02_inventory_management.ipynb`
3. `03_dynamic_pricing.ipynb`

Each notebook is self-contained and includes data loading,
preprocessing, model training, and evaluation.

---

## Data

The dataset is synthetic, generated to represent realistic operational
patterns for a UAE café. It does not contain real customer data.

All four CSV files should be placed in the `data/` directory before
running the notebooks. The files are included in this repository.

Column descriptions for each dataset are documented at the top of
the relevant notebook.

---

## Requirements
pandas>=1.5.0

numpy>=1.23.0

scikit-learn>=1.1.0

tensorflow>=2.10.0       # for LSTM (Module 1)

xgboost>=1.7.0           # alternative forecaster (Module 1)

matplotlib>=3.6.0

seaborn>=0.12.0

jupyter>=1.0.0

---

## Ethical Considerations

- All data is synthetic; no real customer PII is included
- A real deployment would require compliance with UAE Federal
  Decree-Law No. 45 of 2021 (Personal Data Protection Law)
- The staffing module is designed as a decision-support tool only;
  human manager override is required at all times
- Dynamic pricing discounts are applied to end-of-day perishables
  only and are intended to reduce food waste, not to engage in
  predatory pricing

---

## Submission

- **Report:** `/report/CSCI323_Slate_Kitchen_Report.pdf`
- **Presentation:** `/presentation/CSCI323_Slate_Kitchen_Slides.pptx`
- **Video demo:** [link to be added before submission]

---

## Acknowledgements

We thank the CSCI323 course instructors — Dr. Milan Dordevic,
Dr. Abdullah El Nokiti, Ms. Asma Damankesh, and
Ms. Syama Kurungodathil — for their guidance throughout this project.

---

*UOWD · CSCI323 · Spring 2026*
