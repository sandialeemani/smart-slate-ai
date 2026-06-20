# Smart Operations AI for Slate Kitchen & Café
 
**CSCI323 Modern Artificial Intelligence | University of Wollongong Dubai | Spring 2026**

An AI-powered operational management platform developed for Slate Kitchen and Cafe, a food and beverage business operating in Dubai, UAE. The system addresses three core operational inefficiencies through machine learning: unpredictable customer demand, reactive inventory management, and end-of-day perishable food waste. The platform comprises three integrated modules covering demand forecasting, inventory management, and dynamic pricing, built on six months of synthetic operational data modelled on real cafe operations.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Team](#team)
- [Repository Structure](#repository-structure)
- [Datasets](#datasets)
- [Module 1: Demand Forecasting](#module-1-demand-forecasting)
- [Module 2: Inventory Management System](#module-2-inventory-management-system)
- [Module 3: Dynamic Pricing Engine](#module-3-dynamic-pricing-engine)
- [Results Summary](#results-summary)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Limitations](#limitations)
- [Ethical Considerations](#ethical-considerations)
- [References](#references)

---

## Project Overview

Slate Kitchen and Cafe operates in the UAE food and beverage sector, a market valued at approximately AED 104 billion and growing at over 6 percent annually. Like many small and medium enterprises in this sector, the business relied on manual decision-making processes and WhatsApp-based staff communication, leaving it exposed to demand uncertainty, overstocking of perishable goods, and lost revenue from end-of-day food write-offs.

This project applies supervised machine learning to convert historical sales, staffing, and inventory records into proactive operational decisions. The three modules operate as a sequential pipeline: the demand forecast informs inventory consumption estimates, and expiry risk flags from the inventory module trigger discounting decisions in the pricing engine.

The dataset covers 181 operating days from November 2024 to April 2025, comprising 64,855 sales transactions, 1,448 staffing records, 8,326 inventory records, and 1,457 dynamic pricing events. All data is synthetic, generated to reflect realistic operational patterns for a Dubai cafe including weekend peaks, UAE public holidays, and the tourist peak season from November to February.

---

## Team

| Name | Student ID | Role |
|---|---|---|
| Ryan Sethi | 8891229 | Project Lead |
| Akram Kamhawi | 8870226 | Demand Forecasting Model |
| Ahmed Mohammed Sifat Ahmed | 8787700 | Inventory Management System |
| David Samir | 8220220 | Dynamic Pricing Engine |
| Sandiya Kumari | 8990682 | Evaluation Report and Presentation Lead |

- **Institution:** University of Wollongong Dubai (UOWD)
- **Course:** CSCI323 Modern Artificial Intelligence
- **Tutor:** Dr. Abdalla Elnikiti
- **Submission Date:** 21 June 2026

---

## Repository Structure

```
smart-slate-ai/
├── notebooks/
│   ├── demand_forecasting_Model.ipynb   # Module 1: Logistic Regression and KNN classifier
│   ├── Low_Stock_Predictor.ipynb        # Module 2: Random Forest low-stock and expiry risk
│   └── Dynamic_Price_Engine.ipynb       # Module 3: Rule-based dynamic pricing engine
├── data/
│   ├── slate_sales_transactions.csv     # 64,855 transaction records, Nov 2024 - Apr 2025
│   ├── slate_staffing.csv               # 1,448 daily staffing records
│   ├── slate_inventory_daily.csv        # 8,326 daily inventory records
│   └── slate_dynamic_pricing.csv        # 1,457 dynamic pricing events
├── README.md
└── requirements.txt
```

---

## Datasets

All four datasets are synthetic and were generated to represent realistic operational patterns for a UAE-based cafe. They are modelled on real menu items, real ingredient names, and real pricing in AED. The data does not contain any personal customer information.

### `slate_sales_transactions.csv`

| Column | Description |
|---|---|
| `transaction_id` | Unique identifier for each transaction |
| `date` | Date of transaction in `YYYY-MM-DD` format |
| `time` | Time of transaction in `HH:MM:SS` format |
| `day_of_week` | Day name Monday through Sunday |
| `day_type` | Weekday or weekend classification |
| `item_name` | Menu item ordered |
| `category` | Menu category such as Burgers, Eggs and Co., Omelets |
| `quantity` | Number of units ordered |
| `unit_price_aed` | Price per unit in AED |
| `discount_pct` | Discount applied, between 0 and 40 percent |
| `total_revenue_aed` | Final revenue after discount |

**Coverage:** 64,855 records | November 2024 to April 2025 | Average 358 transactions per day

### `slate_staffing.csv`

| Column | Description |
|---|---|
| `date` | Date of staffing record |
| `shift_type` | Morning, afternoon, or evening |
| `role` | Staff role such as Waiter, Line Cook, Kitchen Helper |
| `headcount` | Number of staff scheduled |
| `estimated_covers` | Estimated customer footfall for the day |
| `peak_flag` | Flag indicating a predicted peak period |

**Coverage:** 1,448 records | 181 operating days

### `slate_inventory_daily.csv`

| Column | Description |
|---|---|
| `date` | Date of inventory record |
| `ingredient` | Ingredient name |
| `opening_stock` | Stock level at start of day |
| `received_qty` | Quantity received from supplier |
| `usage_qty` | Quantity consumed in operations |
| `wastage_qty` | Quantity wasted or discarded |
| `closing_stock` | Stock level at end of day |
| `par_level` | Minimum acceptable stock level |
| `reorder_point` | Threshold triggering a reorder alert |
| `low_stock_alert` | Binary flag, 1 if stock fell below reorder point |
| `expiry_risk_flag` | Binary flag, 1 if item flagged as expiry risk |

**Coverage:** 8,326 records across all tracked ingredients

### `slate_dynamic_pricing.csv`

| Column | Description |
|---|---|
| `date` | Date of pricing event |
| `time_slot` | Time discount was triggered: 18:00, 19:00, or 20:00 |
| `item_name` | Bakery item subject to discount |
| `base_price_aed` | Original full price in AED |
| `discount_pct` | Discount tier applied: 20, 30, or 40 percent |
| `units_sold_at_discount` | Units sold under the discounted price |
| `revenue_from_discount_aed` | Revenue generated from discounted sales |
| `waste_units_avoided` | Units that would have been discarded without discounting |

**Coverage:** 1,457 pricing events across Croissants, Assorted Cakes, and Chocolate Brownies

---

## Module 1: Demand Forecasting

**Notebook:** `notebooks/demand_forecasting_Model.ipynb`

### Objective

Classify each operating day as a high-revenue busy day or low-revenue quiet day to support staffing and ingredient preparation decisions. The question being answered is not the exact revenue figure but whether the day will require elevated operational readiness.

### Approach

Two classification algorithms covered in CSCI323 were trained and compared: Logistic Regression and K-Nearest Neighbors with k set to 5.

- **Data sources used:** `slate_sales_transactions.csv` and `slate_staffing.csv`
- **Aggregation:** 64,855 individual transactions and 1,448 staffing records were aggregated to 181 daily observations. After lag feature creation removed the first several rows, 174 usable daily records remained.

### Feature Engineering

Eighteen features were engineered from the date column and aggregated daily data:

- `day_of_week` — Integer 0 (Monday) to 6 (Sunday)
- `month` — Month number 1 to 12
- `day_of_month` — Day within the month
- `is_weekend` — 1 if Friday, Saturday, or Sunday
- `is_friday` — 1 if Friday specifically, the busiest day in the UAE
- `peak_season` — 1 if November, December, January, or February
- `is_holiday` — 1 if UAE public holiday such as National Day or Eid Al Fitr
- `revenue_lag1` — Total revenue from the previous day
- `revenue_lag7` — Total revenue from exactly one week prior
- `revenue_rolling7` — Seven-day rolling average revenue
- `covers_lag1` — Customer footfall from the previous day
- `day_name_Monday` … `day_name_Sunday` — One-hot encoded day name as seven binary columns

### Target Variable

The median daily revenue across the six-month period was AED 12,438. Days above the median were labelled class 1 (high revenue / busy). Days below were labelled class 0 (low revenue / quiet). This produced a balanced dataset with approximately 50 percent of each class.

### Train and Test Split

Data was split chronologically with 80 percent for training (139 days) and 20 percent for testing (35 days). Shuffle was set to `False` to preserve time ordering and prevent data leakage. `StandardScaler` was fitted on the training partition only and then applied to the test partition.

### Results

| Model | Accuracy | Precision | Recall | F1-Score | Training | Test |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9429 | 0.8667 | 1.0000 | 0.9286 | 139 days | 35 days |
| KNN (k=5) | 0.9429 | 0.8667 | 1.0000 | 0.9286 | 139 days | 35 days |

Both models achieved identical results. Recall of 1.0 means every high-revenue day in the test set was correctly identified with no misses. Precision of 0.87 indicates the model occasionally flagged a quiet day as busy, which is the less costly error in a cafe context, as mild overpreparation is preferable to underpreparation. **Logistic Regression is selected as the preferred model due to interpretability.**

### Observed Demand Patterns

The morning peak runs from 9 to 10 AM with 8,624 transactions recorded at 10 AM across the full dataset. The lunch peak runs from 12 to 1 PM with 9,200 transactions at 1 PM. A secondary shoulder period exists from 5 to 6 PM. Revenue peaked in December 2024 at AED 433,269 and reached its lowest point in April 2025 at AED 339,456, consistent with Ramadan trading hour reductions reducing daytime footfall.

---

## Module 2: Inventory Management System

**Notebook:** `notebooks/Low_Stock_Predictor.ipynb`

### Objective

Transform daily inventory records and demand predictions into actionable stock management decisions. The system predicts low-stock events before they occur and flags items at risk of expiry, enabling proactive reordering rather than reactive restocking.

### Component 1: Low Stock Prediction

A Random Forest Classifier was trained to predict whether any ingredient would fall below its reorder threshold on a given day.

- **Features used:** `opening_stock`, `received_qty`, `usage_qty`, `closing_stock`, `par_level`, `reorder_point`
- **Target variable:** `low_stock_alert` — 0 = stock sufficient, 1 = reorder required

| Metric | Score |
|---|---|
| Accuracy | 100% |
| Precision | 100% |
| Recall | 100% |
| F1-Score | 100% |

### Component 2: Expiry Risk Prediction

A second Random Forest Classifier was trained to identify inventory items at risk of expiring before being consumed.

- **Features used:** `opening_stock`, `usage_qty`, `wastage_qty`, `closing_stock`
- **Target variable:** `expiry_risk_flag` — 0 = low risk, 1 = high risk

| Metric | Score |
|---|---|
| Accuracy | 98% |
| Precision | 98% |
| Recall | 98% |
| F1-Score | 98% |

The lower recall observed for the minority expiry-risk class indicates a small number of high-risk products may remain undetected, representing an opportunity for future model refinement.

### Component 3: Supplier Order Recommendation

Recommended order quantities are calculated using the following formula:

```
Recommended Order Quantity = (Expected Consumption + Safety Stock) - Current Stock
```

**Example:** Current milk inventory is 15 litres. Predicted consumption is 20 litres. Safety stock is 5 litres.
Recommended Order Quantity = 20 + 5 − 15 = **10 litres**

### System Performance

| Metric | Value |
|---|---|
| Total inventory records processed | 8,326 |
| Low-stock alerts correctly triggered | 2,267 |
| Expiry risk flags raised | 290 |
| Overall ingredient wastage rate | 2.40% |
| Total wastage quantity over 6 months | 8,300 units |
| Total ingredient cost over 6 months | AED 1,033,149 |

### Top Wastage by Ingredient

| Ingredient | Total Wastage (units) |
|---|---|
| Lemon Juice | 4,671.59 |
| Eggs | 1,114.25 |
| Brioche Buns | 393.62 |
| Hashbrown (frozen) | 339.67 |
| Beef Patty (Wagyu) | 276.60 |

Lemon Juice accounts for a disproportionate share of total wastage and is likely a portion-sizing or shelf-life management issue rather than a demand prediction failure.

### Low-Stock Alerts by Ingredient (Top 8)

| Ingredient | Alert Days |
|---|---|
| Chicken Breast | 103 |
| Brioche Buns | 97 |
| Ribeye Steak | 97 |
| Beef Brisket | 91 |
| Hashbrown (frozen) | 85 |
| Beef Patty (Wagyu) | 84 |
| Eggs | 84 |
| Salmon Fillet | 75 |

Chicken Breast triggered low-stock alerts on 103 out of 181 operating days. As it is a core ingredient for several top-revenue menu items, the business recommendation is to raise the par level and reorder point for this ingredient immediately.

---

## Module 3: Dynamic Pricing Engine

**Notebook:** `notebooks/Dynamic_Price_Engine.ipynb`

### Objective

Recover revenue from perishable bakery items that would otherwise be discarded unsold at end of day. The engine applies time-based discount tiers to slow-selling perishables as closing time approaches, converting potential waste into discounted revenue.

### Pricing Logic

The engine calculates hours remaining until store closure at 22:00 and applies the following discount tiers:

| Time | Hours to Close | Discount |
|---|---|---|
| 18:00 | 4 hours | 20% off base price |
| 19:00 | 3 hours | 30% off base price |
| 20:00 | 2 hours | 40% off base price |

Discounts are only triggered when the expiry risk flag from Module 2 is set and unsold stock remains. If stock reaches zero at any point, no further pricing logic is applied.

**Items covered:** Croissant Plain Butter, Assorted Cakes, Chocolate Brownies with Ice Cream

### Business Assumptions

Cost of Goods Sold is assumed at 30 percent of base price. Waste disposal cost is assumed at AED 1.00 per discarded unit.

### Revenue Impact by Pricing Tier

| Time | Discount | Units Sold | Revenue (AED) | Waste Units Avoided |
|---|---|---|---|---|
| 18:00 | 20% | 1,149 | 15,988.00 | 1,149 |
| 19:00 | 30% | 927 | 11,487.70 | 927 |
| 20:00 | 40% | 682 | 7,333.20 | 682 |

### Performance by Item

| Item | Waste Units Avoided | Revenue Recovered (AED) |
|---|---|---|
| Assorted Cakes | 946 | 10,234.50 |
| Chocolate Brownies with Ice Cream | 906 | 18,082.40 |
| Croissant Plain Butter | 906 | 6,492.00 |

Chocolate Brownies generated the highest revenue recovery despite similar waste-avoidance volume to Croissants, owing to its higher base price.

### Overall Engine Performance

| Metric | Value |
|---|---|
| Total pricing slots evaluated | 1,457 |
| Discounts triggered | 1,085 (74.5%) |
| Total waste units avoided | 2,758 |
| Revenue recovered via discounting | AED 34,809 |
| Waste disposal savings | AED 2,758 |
| **Net business value created** | **AED 37,567** |
| Average discount applied | 29.1% |

---

## Results Summary

| Metric | Baseline | AI Result | Notes |
|---|---|---|---|
| Demand classification accuracy | No forecasting system | 0.9429 accuracy, F1-Score 0.9286 | 35-day held-out test set |
| Ingredient wastage rate | Significantly above 1–2% industry benchmark | 2.40% | Source: Loxton, 2026 |
| Low-stock days (Chicken Breast) | Untracked | 103 of 181 days flagged | Proactive reordering enabled |
| Perishable waste units avoided | 0 (no system) | 2,758 units | Over 6-month period |
| Revenue recovered from discounting | AED 0 | AED 34,809 | New revenue stream created |
| Understaffed shifts flagged | Not tracked | 92 shifts (6.4%) | Scheduling risk identified |

---

## Installation and Setup

**Requirements:** Python 3.8 or higher and Google Colab recommended, or a local Jupyter environment.

Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

Or install from the requirements file:

```bash
pip install -r requirements.txt
```

`requirements.txt` contents:

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

---

## Usage

### Running in Google Colab

Each notebook is self-contained and can be run in Google Colab. Upload the relevant CSV files when prompted.

**Module 1 — Demand Forecasting:**
1. Open `demand_forecasting_Model.ipynb` in Colab.
2. Upload `slate_sales_transactions.csv` and `slate_staffing.csv` when the file upload prompt appears.
3. Run all cells in order. Outputs include confusion matrices, model comparison charts, a k-tuning curve, and exported CSV files.

**Module 2 — Inventory Management:**
1. Open `Low_Stock_Predictor.ipynb` in Colab.
2. Upload `slate_inventory_daily.csv` when prompted.
3. Run all cells in order. Outputs include classification reports for both the low-stock and expiry risk models, and the recommended order quantity for a sample ingredient.

**Module 3 — Dynamic Pricing Engine:**
1. Open `Dynamic_Price_Engine.ipynb` in Colab.
2. Upload `slate_dynamic_pricing.csv` when prompted.
3. Run all cells in order. Outputs include revenue impact by pricing tier and the full business value summary.

### Running Locally

```bash
git clone https://github.com/sandialeemani/smart-slate-ai
cd smart-slate-ai
pip install -r requirements.txt
jupyter notebook
```

Place the CSV files in the same directory as the notebooks, or update the file path variables at the top of each notebook.

---

## Limitations

- **Data limitations:** All datasets are synthetic. Real-world deployment would introduce noise, missing values, sensor failures, and irregular events such as Ramadan trading hours and mall closure events that the synthetic data does not fully replicate. All reported performance metrics should be interpreted as indicative rather than deployable benchmarks.
- **Dataset window:** The six-month window from November 2024 to April 2025 is insufficient to capture full annual seasonality. The Dubai summer slowdown from June to August is entirely absent. A minimum of 18 to 24 months of real data would be needed for robust seasonal modelling.
- **Demand model:** The forecasting model does not incorporate external signals relevant to Dubai restaurant footfall, including nearby competitor promotions, weather, social media virality events, or real-time venue activity nearby.
- **Inventory model:** Reorder points and par levels were fixed in the synthetic dataset. In real deployment these thresholds would need dynamic calibration using historical stockout data and supplier lead times.
- **Pricing model:** Fixed time-based discount tiers can cause strategic customer waiting behaviour, where customers delay purchases in anticipation of end-of-day discounts, depressing full-price revenue earlier in the day.
- **Computational:** The project was developed in Google Colab with free-tier resources. Production deployment would require a persistent cloud instance with scheduled retraining pipelines.

---

## Ethical Considerations

- **Data privacy:** Any real deployment would require compliance with UAE Federal Decree-Law No. 45 of 2021 on Personal Data Protection. Transaction-level customer data must be anonymised, and appropriate consent or legitimate interest provisions must be established before processing.
- **Staffing fairness:** The system identified 92 understaffed shifts disproportionately affecting Kitchen Helpers, Waiters, and Line Cooks. AI-driven scheduling systems that generate systematic errors for frontline workers raise fairness concerns. Human override capability must be preserved in any deployment.
- **Pricing equity:** If discounts are communicated only through digital channels or a loyalty application, customers without smartphones may be excluded. A dual-channel approach combining in-store digital signage with app-based notifications is recommended to ensure equitable access.
- **Environmental alignment:** The waste reduction outcomes of this system align with the UAE Food Waste Pledge and the UAE Net Zero 2050 strategy. The UAE Ministry of Climate Change and Environment targets a 50 percent reduction in food waste by 2030 (MOCCAE, 2023).

---

## References

Ferreira, K.J., Lee, B.H.A. and Simchi-Levi, D. (2016) 'Analytics for an online retailer: demand forecasting and price optimization', *Manufacturing and Service Operations Management*, 18(1), pp. 69–88.

Harris, C.R., Millman, K.J., van der Walt, S.J. et al. (2020) 'Array programming with NumPy', *Nature*, 585, pp. 357–362.

Hochreiter, S. and Schmidhuber, J. (1997) 'Long short-term memory', *Neural Computation*, 9(8), pp. 1735–1780.

Loxton, L. (2026) *UAE gets serious about food waste*. AB Magazine, ACCA Global. Available at: [abmagazine.accaglobal.com](https://abmagazine.accaglobal.com/global/articles/2026/jan/business/uae-gets-serious-about-food-waste.html) (Accessed: 10 June 2026).

McKinney, W. (2010) 'Data structures for statistical computing in Python', *Proceedings of the 9th Python in Science Conference*, pp. 56–61.

Mitchell, T.M. (2021) *Machine learning: a probabilistic perspective*. Cambridge, MA: MIT Press.

OpenAI (2026) *ChatGPT (GPT-5.5)* [Large language model]. Available at: [chatgpt.com](https://chatgpt.com/) (Accessed: 16 June 2026). Used as a supplementary tool to assist with understanding project requirements and distributing tasks. All final decisions, analysis, and submitted work were reviewed and completed by the authors.

Pedregosa, F., Varoquaux, G., Gramfort, A. et al. (2011) 'Scikit-learn: machine learning in Python', *Journal of Machine Learning Research*, 12, pp. 2825–2830.

Pinnacle Publications (2024) *Global food waste statistics and economic impact*. Available at: [pinnaclepublications.com](https://www.pinnaclepublications.com) (Accessed: 10 June 2026).

UAE Ministry of Climate Change and Environment (2023) *National food waste reduction framework*. Abu Dhabi: MOCCAE.

UAE Official Gazette (2021) *Federal Decree-Law No. 45 of 2021 on Personal Data Protection*. Available at: [u.ae](https://u.ae/en/information-and-services/justice-safety-and-the-law/handling-personal-data-in-the-uae) (Accessed: 10 June 2026).

---

*Submitted as part of CSCI323 Modern Artificial Intelligence, University of Wollongong Dubai, Spring 2026.*
