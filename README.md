Smart Operations AI for Slate Kitchen and Cafe
CSCI323 Modern Artificial Intelligence | University of Wollongong Dubai | Spring 2026
An AI-powered operational management platform developed for Slate Kitchen and Cafe, a food and beverage business operating in Dubai, UAE. The system addresses three core operational inefficiencies through machine learning: unpredictable customer demand, reactive inventory management, and end-of-day perishable food waste. The platform comprises three integrated modules covering demand forecasting, inventory management, and dynamic pricing, built on six months of synthetic operational data modelled on real cafe operations.


Table of Contents

Project Overview
Team
Repository Structure
Datasets
Module 1: Demand Forecasting
Module 2: Inventory Management System
Module 3: Dynamic Pricing Engine
Results Summary
Installation and Setup
Usage
Limitations
Ethical Considerations
References


Project Overview
Slate Kitchen and Cafe operates in the UAE food and beverage sector, a market valued at approximately AED 104 billion and growing at over 6 percent annually. Like many small and medium enterprises in this sector, the business relied on manual decision-making processes and WhatsApp-based staff communication, leaving it exposed to demand uncertainty, overstocking of perishable goods, and lost revenue from end-of-day food write-offs.
This project applies supervised machine learning to convert historical sales, staffing, and inventory records into proactive operational decisions. The three modules operate as a sequential pipeline: the demand forecast informs inventory consumption estimates, and expiry risk flags from the inventory module trigger discounting decisions in the pricing engine.
The dataset covers 181 operating days from November 2024 to April 2025, comprising 64,855 sales transactions, 1,448 staffing records, 8,326 inventory records, and 1,457 dynamic pricing events. All data is synthetic, generated to reflect realistic operational patterns for a Dubai cafe including weekend peaks, UAE public holidays, and the tourist peak season from November to February.

Team
NameStudent IDRoleRyan Sethi8891229Project LeadAkram Kamhawi8870226Demand Forecasting ModelAhmed Mohammed Sifat Ahmed8787700Inventory Management SystemDavid Samir8220220Dynamic Pricing EngineSandiya Kumari8990682Evaluation Report and Presentation Lead
Institution: University of Wollongong Dubai (UOWD)

Course: CSCI323 Modern Artificial Intelligence

Tutor: Dr. Abdalla Elnikiti

Submission Date: 21 June 2026

Repository Structure
smart-slate-ai/
│
├── notebooks/
│   ├── demand_forecasting_Model.ipynb       # Module 1: Logistic Regression and KNN classifier
│   ├── Low_Stock_Predictor.ipynb            # Module 2: Random Forest low-stock and expiry risk
│   └── Dynamic_Price_Engine.ipynb           # Module 3: Rule-based dynamic pricing engine
│
├── data/
│   ├── slate_sales_transactions.csv         # 64,855 transaction records Nov 2024 to Apr 2025
│   ├── slate_staffing.csv                   # 1,448 daily staffing records
│   ├── slate_inventory_daily.csv            # 8,326 daily inventory records
│   └── slate_dynamic_pricing.csv            # 1,457 dynamic pricing events
│
├── outputs/
│   ├── slate_model_metrics.csv              # Model performance metrics export
│   └── slate_forecast_output.csv            # Daily busy/quiet day predictions
│
├── README.md
└── requirements.txt

Datasets
All four datasets are synthetic and were generated to represent realistic operational patterns for a UAE-based cafe. They are modelled on real menu items, real ingredient names, and real pricing in AED. The data does not contain any personal customer information.
slate_sales_transactions.csv
ColumnDescriptiontransaction_idUnique identifier for each transactiondateDate of transaction (YYYY-MM-DD)timeTime of transaction (HH:MM:SS)day_of_weekDay name (Monday through Sunday)day_typeWeekday or weekend classificationitem_nameMenu item orderedcategoryMenu category (Burgers, Eggs and Co, Omelets, etc.)quantityNumber of units orderedunit_price_aedPrice per unit in AEDdiscount_pctDiscount applied (0 to 40 percent)total_revenue_aedFinal revenue after discount
Coverage: 64,855 records | November 2024 to April 2025 | Average 358 transactions per day
slate_staffing.csv
ColumnDescriptiondateDate of staffing recordshift_typeMorning, afternoon, or eveningroleStaff role (Waiter, Line Cook, Kitchen Helper, etc.)headcountNumber of staff scheduledestimated_coversEstimated customer footfall for the daypeak_flagFlag indicating a predicted peak period
Coverage: 1,448 records | 181 operating days
slate_inventory_daily.csv
ColumnDescriptiondateDate of inventory recordingredientIngredient nameopening_stockStock level at start of dayreceived_qtyQuantity received from supplierusage_qtyQuantity consumed in operationswastage_qtyQuantity wasted or discardedclosing_stockStock level at end of daypar_levelMinimum acceptable stock levelreorder_pointThreshold triggering a reorder alertlow_stock_alertBinary flag: 1 if stock fell below reorder pointexpiry_risk_flagBinary flag: 1 if item flagged as expiry risk
Coverage: 8,326 records across all tracked ingredients
slate_dynamic_pricing.csv
ColumnDescriptiondateDate of pricing eventtime_slotTime discount was triggered (18:00, 19:00, or 20:00)item_nameBakery item subject to discountbase_price_aedOriginal full price in AEDdiscount_pctDiscount tier applied (20, 30, or 40 percent)units_sold_at_discountUnits sold under the discounted pricerevenue_from_discount_aedRevenue generated from discounted saleswaste_units_avoidedUnits that would have been discarded without discounting
Coverage: 1,457 pricing events across Croissants, Assorted Cakes, and Chocolate Brownies

Module 1: Demand Forecasting
Notebook: notebooks/demand_forecasting_Model.ipynb
Objective
Classify each operating day as a high-revenue (busy) day or low-revenue (quiet) day to support staffing and ingredient preparation decisions. The question being answered is not the exact revenue figure but whether the day will require elevated operational readiness.
Approach
Two classification algorithms covered in CSCI323 were trained and compared: Logistic Regression and K-Nearest Neighbors (k=5).
Data sources used: slate_sales_transactions.csv and slate_staffing.csv
Aggregation: 64,855 individual transactions and 1,448 staffing records were aggregated to 181 daily observations. After lag feature creation removed the first several rows, 174 usable daily records remained.
Feature Engineering
Eighteen features were engineered from the date column and aggregated daily data:
FeatureDescriptionday_of_weekInteger 0 (Monday) to 6 (Sunday)monthMonth number 1 to 12day_of_monthDay within the monthis_weekend1 if Friday, Saturday, or Sundayis_friday1 if Friday specifically (busiest day in UAE)peak_season1 if November, December, January, or Februaryis_holiday1 if UAE public holiday (National Day, Eid Al Fitr)revenue_lag1Total revenue from the previous dayrevenue_lag7Total revenue from exactly one week priorrevenue_rolling7Seven-day rolling average revenuecovers_lag1Customer footfall from the previous dayday_name_Monday to day_name_SundayOne-hot encoded day name (seven binary columns)
Target Variable
The median daily revenue across the six-month period was AED 12,438. Days above the median were labelled class 1 (high revenue or busy). Days below were labelled class 0 (low revenue or quiet). This produced a balanced dataset with approximately 50 percent of each class.
Train and Test Split
Data was split chronologically with 80 percent for training (139 days) and 20 percent for testing (35 days). Shuffle was set to False to preserve time ordering and prevent data leakage. StandardScaler was fitted on the training partition only and then applied to the test partition.
Results
MetricLogistic RegressionKNN (k=5)Accuracy0.94290.9429Precision0.86670.8667Recall1.00001.0000F1-Score0.92860.9286Training samples139 days139 daysTest samples35 days35 days
Both models achieved identical results. Recall of 1.0 means every high-revenue day in the test set was correctly identified with no misses. Precision of 0.87 indicates the model occasionally flagged a quiet day as busy, which is the less costly error in a cafe context as mild overpreparation is preferable to underpreparation. Logistic Regression is selected as the preferred model due to interpretability.
Observed Demand Patterns
The morning peak runs from 9 to 10 AM with 8,624 transactions recorded at 10 AM across the full dataset. The lunch peak runs from 12 to 1 PM with 9,200 transactions at 1 PM. A secondary shoulder period exists from 5 to 6 PM. Revenue peaked in December 2024 at AED 433,269 and reached its lowest point in April 2025 at AED 339,456, consistent with Ramadan trading hour reductions reducing daytime footfall.

Module 2: Inventory Management System
Notebook: notebooks/Low_Stock_Predictor.ipynb
Objective
Transform daily inventory records and demand predictions into actionable stock management decisions. The system predicts low-stock events before they occur and flags items at risk of expiry, enabling proactive reordering rather than reactive restocking.
Component 1: Low Stock Prediction
A Random Forest Classifier was trained to predict whether any ingredient would fall below its reorder threshold on a given day.
Features used: opening_stock, received_qty, usage_qty, closing_stock, par_level, reorder_point
Target variable: low_stock_alert (0 = stock sufficient, 1 = reorder required)
Performance:
MetricValueAccuracy100%Precision100%Recall100%F1-Score100%
Component 2: Expiry Risk Prediction
A second Random Forest Classifier was trained to identify inventory items at risk of expiring before being consumed.
Features used: opening_stock, usage_qty, wastage_qty, closing_stock
Target variable: expiry_risk_flag (0 = low risk, 1 = high risk)
Performance:
MetricValueAccuracy98%Precision98%Recall98%F1-Score98%
The lower recall observed for the minority expiry-risk class indicates a small number of high-risk products may remain undetected, representing an opportunity for future model refinement.
Component 3: Supplier Order Recommendation
Recommended order quantities are calculated using the following formula:
Recommended Order Quantity = (Expected Consumption + Safety Stock) - Current Stock
Example: Current milk inventory is 15 litres. Predicted consumption is 20 litres. Safety stock is 5 litres. Recommended order quantity = (20 + 5) - 15 = 10 litres.
System Performance
MetricValueTotal inventory records processed8,326Low-stock alerts correctly triggered2,267Expiry risk flags raised290Overall ingredient wastage rate2.40%Total wastage quantity (6 months)8,300 unitsTotal ingredient cost (6 months)AED 1,033,149
Top Wastage by Ingredient
IngredientTotal Wastage (units)Lemon Juice4,671.59Eggs1,114.25Brioche Buns393.62Hashbrown (frozen)339.67Beef Patty (Wagyu)276.60
Lemon Juice accounts for a disproportionate share of total wastage and is likely a portion-sizing or shelf-life management issue rather than a demand prediction failure.
Low-Stock Alerts by Ingredient (Top 8)
IngredientAlert DaysChicken Breast103Brioche Buns97Ribeye Steak97Beef Brisket91Hashbrown (frozen)85Beef Patty (Wagyu)84Eggs84Salmon Fillet75
Chicken Breast triggered low-stock alerts on 103 out of 181 operating days. As it is a core ingredient for several top-revenue menu items, the business recommendation is to raise the par level and reorder point for this ingredient immediately.

Module 3: Dynamic Pricing Engine
Notebook: notebooks/Dynamic_Price_Engine.ipynb
Objective
Recover revenue from perishable bakery items that would otherwise be discarded unsold at end of day. The engine applies time-based discount tiers to slow-selling perishables as closing time approaches, converting potential waste into discounted revenue.
Pricing Logic
The engine calculates hours remaining until store closure (22:00) and applies the following discount tiers:
TimeHours to CloseDiscount Applied18:004 hours20% off base price19:003 hours30% off base price20:002 hours40% off base price
Discounts are only triggered when the expiry risk flag from Module 2 is set and unsold stock remains. If stock reaches zero at any point, no further pricing logic is applied.
Items covered: Croissant (Plain Butter), Assorted Cakes, Chocolate Brownies with Ice Cream
Business Assumptions

Cost of Goods Sold (COGS) is assumed at 30 percent of base price
Waste disposal cost is assumed at AED 1.00 per discarded unit

Revenue Impact by Pricing Tier
Time SlotDiscountUnits SoldRevenue (AED)Waste Units Avoided18:0020%1,14915,988.001,14919:0030%92711,487.7092720:0040%6827,333.20682
Performance by Item
ItemWaste Units AvoidedRevenue Recovered (AED)Assorted Cakes94610,234.50Chocolate Brownies with Ice Cream90618,082.40Croissant (Plain Butter)9066,492.00
Chocolate Brownies generated the highest revenue recovery despite similar waste-avoidance volume to Croissants, owing to its higher base price.
Overall Engine Performance
MetricValueTotal pricing slots evaluated1,457Discounts triggered1,085 (74.5%)Total waste units avoided2,758Revenue recovered via discountingAED 34,809Waste disposal savingsAED 2,758Net business value createdAED 37,567Average discount applied29.1%

Results Summary
KPIBaseline (Pre-AI)AI-Assisted ResultNotesDemand classification accuracyNo forecasting system0.9429 (F1: 0.9286)35-day held-out test setIngredient wastage rateSignificantly above 1 to 2% industry benchmark2.40%Loxton (2026)Low-stock days (Chicken Breast)Untracked103 of 181 days flaggedProactive reordering enabledPerishable waste units avoided0 (no system)2,758 unitsOver 6-month periodRevenue recovered from discountingAED 0AED 34,809New revenue streamUnderstaffed shifts flaggedNot tracked92 shifts (6.4%)Scheduling risk identified

Installation and Setup
Requirements

Python 3.8 or higher
Google Colab (recommended) or local Jupyter environment

Install Dependencies
bashpip install pandas numpy scikit-learn matplotlib seaborn
Or install from the requirements file:
bashpip install -r requirements.txt
requirements.txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0

Usage
Running in Google Colab
Each notebook is self-contained and can be run in Google Colab. Upload the relevant CSV files when prompted.
Module 1 (Demand Forecasting):

Open demand_forecasting_Model.ipynb in Colab
Upload slate_sales_transactions.csv and slate_staffing.csv when the file upload prompt appears
Run all cells in order
Outputs include confusion matrices, model comparison charts, k-tuning curve, and exported CSV files

Module 2 (Inventory Management):

Open Low_Stock_Predictor.ipynb in Colab
Upload slate_inventory_daily.csv when prompted
Run all cells in order
Outputs include classification reports for both the low-stock and expiry risk models, and the recommended order quantity for a sample ingredient

Module 3 (Dynamic Pricing Engine):

Open Dynamic_Price_Engine.ipynb in Colab
Upload slate_dynamic_pricing.csv when prompted
Run all cells in order
Outputs include revenue impact by pricing tier and the full business value summary

Running Locally
bashgit clone https://github.com/sandialeemani/smart-slate-ai
cd smart-slate-ai
pip install -r requirements.txt
jupyter notebook
Place the CSV files in the same directory as the notebooks, or update the file path variables at the top of each notebook.

Limitations
Data limitations: All datasets are synthetic. Real-world deployment would introduce noise, missing values, sensor failures, and irregular events such as Ramadan trading hours and mall closure events that the synthetic data does not fully replicate. All reported performance metrics should be interpreted as indicative rather than deployable benchmarks.
Dataset window: The six-month window from November 2024 to April 2025 is insufficient to capture full annual seasonality. The Dubai summer slowdown from June to August is entirely absent. A minimum of 18 to 24 months of real data would be needed for robust seasonal modelling.
Demand model limitations: The forecasting model does not incorporate external signals relevant to Dubai restaurant footfall, including nearby competitor promotions, weather, social media virality events, or real-time venue activity nearby.
Inventory model limitations: Reorder points and par levels were fixed in the synthetic dataset. In real deployment, these thresholds would need dynamic calibration using historical stockout data and supplier lead times.
Pricing model limitations: Fixed time-based discount tiers can cause strategic customer waiting behaviour, where customers delay purchases in anticipation of end-of-day discounts, depressing full-price revenue earlier in the day.
Computational limitations: The project was developed in Google Colab with free-tier resources. Production deployment would require a persistent cloud instance with scheduled retraining pipelines.

Ethical Considerations
Data privacy: Any real deployment would require compliance with UAE Federal Decree-Law No. 45 of 2021 on Personal Data Protection (PDPL). Transaction-level customer data must be anonymised and appropriate consent or legitimate interest provisions must be established before processing.
Staffing fairness: The system identified 92 understaffed shifts, disproportionately affecting Kitchen Helpers, Waiters, and Line Cooks. AI-driven scheduling systems that generate systematic errors for frontline workers raise fairness concerns. Human override capability must be preserved in any deployment.
Pricing equity: If discounts are communicated only through digital channels or a loyalty application, customers without smartphones may be excluded. A dual-channel approach combining in-store digital signage with app-based notifications is recommended to ensure equitable access.
Environmental alignment: The waste reduction outcomes of this system align with the UAE Food Waste Pledge and the UAE Net Zero 2050 strategy. The UAE Ministry of Climate Change and Environment targets a 50 percent reduction in food waste by 2030 (MOCCAE, 2023).

References
Ferreira, K.J., Lee, B.H.A. and Simchi-Levi, D. (2016) 'Analytics for an online retailer: demand forecasting and price optimization', Manufacturing and Service Operations Management, 18(1), pp. 69-88.
Harris, C.R., Millman, K.J., van der Walt, S.J. et al. (2020) 'Array programming with NumPy', Nature, 585, pp. 357-362.
Hochreiter, S. and Schmidhuber, J. (1997) 'Long short-term memory', Neural Computation, 9(8), pp. 1735-1780.
Loxton, L. (2026) UAE gets serious about food waste. AB Magazine, ACCA Global. Available at: https://abmagazine.accaglobal.com/global/articles/2026/jan/business/uae-gets-serious-about-food-waste.html (Accessed: 10 June 2026).
McKinney, W. (2010) 'Data structures for statistical computing in Python', Proceedings of the 9th Python in Science Conference, pp. 56-61.
Mitchell, T.M. (2021) Machine learning: a probabilistic perspective. Cambridge, MA: MIT Press.
OpenAI (2026) ChatGPT (GPT-5.5) [Large language model]. Available at: https://chatgpt.com/ (Accessed: 16 June 2026). Used as a supplementary tool to assist with understanding project requirements and distributing tasks. All final decisions, analysis, and submitted work were reviewed and completed by the authors.
Pedregosa, F., Varoquaux, G., Gramfort, A. et al. (2011) 'Scikit-learn: machine learning in Python', Journal of Machine Learning Research, 12, pp. 2825-2830.
Pinnacle Publications (2024) Global food waste statistics and economic impact. Available at: https://www.pinnaclepublications.com (Accessed: 10 June 2026).
UAE Ministry of Climate Change and Environment (2023) National food waste reduction framework. Abu Dhabi: MOCCAE.
UAE Official Gazette (2021) Federal Decree-Law No. 45 of 2021 on Personal Data Protection. Available at: https://u.ae/en/information-and-services/justice-safety-and-the-law/handling-personal-data-in-the-uae (Accessed: 10 June 2026).


Submitted as part of CSCI323 Modern Artificial Intelligence, University of Wollongong Dubai, Spring 2026.
