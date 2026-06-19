from datetime import datetime

class DynamicPricingEngine:
    """
    A rule-based engine to apply dynamic discounts to perishable inventory 
    based on the time remaining until store closure.
    """
    def __init__(self, base_price, current_stock, closing_time='22:00'):
        self.base_price = base_price
        self.current_stock = current_stock
        self.closing_time = closing_time

    def calculate_current_price(self, current_time_str):
        """
        Calculates the appropriate discount tier based on current time.
        Returns: (discounted_price, discount_percentage)
        """
        # If sold out, no pricing logic needed
        if self.current_stock <= 0:
            return self.base_price, 0.0
            
        # FIX: Ensure string is only HH:MM to prevent errors if seconds (HH:MM:SS) are included
        current_time_str = current_time_str[:5]
        
        # Parse times to calculate hours until closing
        time_format = '%H:%M'
        current = datetime.strptime(current_time_str, time_format)
        close = datetime.strptime(self.closing_time, time_format)
        
        hours_to_close = (close - current).total_seconds() / 3600.0
        
        # ---------------------------------------------------------
        # PRICING TIERS LOGIC
        # ---------------------------------------------------------
        if hours_to_close <= 2.0:
            discount_pct = 0.40  # Tier 3: 40% off at 20:00 (T-2 Hours)
        elif hours_to_close <= 3.0:
            discount_pct = 0.30  # Tier 2: 30% off at 19:00 (T-3 Hours)
        elif hours_to_close <= 4.0:
            discount_pct = 0.20  # Tier 1: 20% off at 18:00 (T-4 Hours)
        else:
            discount_pct = 0.00  # Base price applies

        # Calculate final price rounded to 2 decimals
        discounted_price = self.base_price * (1 - discount_pct)
        return round(discounted_price, 2), discount_pct

# --- Example Usage (Safe to leave in for your submission) ---
# Testing with seconds included to prove the safeguard works
pastry = DynamicPricingEngine(base_price=10.00, current_stock=5, closing_time='22:00')
price, discount = pastry.calculate_current_price('19:15:00')
print(f"Algorithm Test - Price: AED {price}, Discount: {discount*100}%\n")

import pandas as pd

# Load the dynamic pricing dataset 
file_path = 'slate_dynamic_pricing.xlsx - slate_dynamic_pricing.csv'
dp_df = pd.read_csv("C:\\Users\\david\\OneDrive - University of Wollongong\\Desktop\\University of Wollongong\\Year 3\\Semester 3\\CSCI 323 - Modern Artificial Intelligence\\PROJECT\\MY PART\\slate_dynamic_pricing.csv")

print("="*50)
print("TASK 1: SIMULATE REVENUE IMPACT BY PRICING TIER")
print("="*50)

# Group the data by time slot and discount percentage to summarize tier performance
tier_summary = dp_df.groupby(['time_slot', 'discount_pct']).agg({
    'units_sold_at_discount': 'sum',
    'revenue_from_discount_aed': 'sum',
    'waste_units_avoided': 'sum'
}).reset_index()

# Format the revenue column for readability in the table
tier_summary['revenue_from_discount_aed'] = tier_summary['revenue_from_discount_aed'].apply(lambda x: f"AED {x:,.2f}")

print(tier_summary.to_string(index=False))
print("\n")

print("="*50)
print("TASK 2: REVENUE VS WASTE-DISPOSAL COST (BUSINESS VALUE)")
print("="*50)

# Define business constants & assumptions
COGS_PERCENTAGE = 0.30              # Cost of Goods Sold is assumed at 30% of base price
WASTE_DISPOSAL_COST_PER_UNIT = 1.00 # Assuming AED 1.00 disposal fee per wasted pastry

# Calculate the COGS for each item
dp_df['cogs'] = dp_df['base_price_aed'] * COGS_PERCENTAGE

# Aggregate total metrics across the entire dataset
total_revenue_from_discount = dp_df['revenue_from_discount_aed'].sum()
total_waste_avoided = dp_df['waste_units_avoided'].sum()

# Calculate Financial Impact
total_disposal_savings = total_waste_avoided * WASTE_DISPOSAL_COST_PER_UNIT
total_cogs_recovered = (dp_df['cogs'] * dp_df['waste_units_avoided']).sum()
net_business_value = total_revenue_from_discount + total_disposal_savings

# Output the quantified metrics perfectly formatted
print(f"Total Waste Units Avoided:      {total_waste_avoided:,} units")
print(f"Total Revenue Added:            AED {total_revenue_from_discount:,.2f}")
print(f"Total COGS Recovered:           AED {total_cogs_recovered:,.2f}")
print("-" * 50)
print(f"Waste Disposal Savings:         AED {total_disposal_savings:,.2f}")
print(f"NET BUSINESS VALUE CREATED:     AED {net_business_value:,.2f}")
print("="*50)

