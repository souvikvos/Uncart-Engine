"""
UNCart - Synthetic E-Commerce Return Dataset Generator (Realistic Calibration)
Calibrated with realistic probabilistic human behavior to land at ~82-84% accuracy.
"""

import numpy as np
import pandas as pd
import os

def generate_uncart_dataset(num_samples: int = 6000, random_seed: int = 42) -> pd.DataFrame:
    np.random.seed(random_seed)
    
    order_ids = [f"ORD_{10000 + i}" for i in range(num_samples)]
    
    # 1. Customer Profiles
    customer_age_days = np.random.randint(5, 1200, size=num_samples)
    past_orders_count = np.random.poisson(lam=4, size=num_samples)
    past_orders_count = np.clip(past_orders_count, 0, 30)
    
    # Customer historical return rate
    customer_return_rate = []
    for count in past_orders_count:
        if count == 0:
            customer_return_rate.append(np.random.choice([0.15, 0.25], p=[0.7, 0.3]))
        else:
            rate = np.random.beta(a=2, b=6)
            customer_return_rate.append(round(float(rate), 3))
    customer_return_rate = np.array(customer_return_rate)

    # 2. Product Categories & Pricing
    categories = ['Apparel', 'Footwear', 'Electronics', 'Home & Kitchen', 'Beauty']
    cat_weights = [0.35, 0.20, 0.20, 0.15, 0.10]
    category_choices = np.random.choice(categories, size=num_samples, p=cat_weights)
    
    prices = []
    for cat in category_choices:
        if cat == 'Apparel':
            prices.append(np.random.randint(499, 4500))
        elif cat == 'Footwear':
            prices.append(np.random.randint(899, 7999))
        elif cat == 'Electronics':
            prices.append(np.random.randint(1299, 25000))
        elif cat == 'Home & Kitchen':
            prices.append(np.random.randint(399, 3500))
        else: # Beauty
            prices.append(np.random.randint(249, 1999))
    product_price = np.array(prices)

    # 3. Order Details
    discount_percent = np.random.choice([0, 10, 20, 30, 40, 50, 60], size=num_samples, p=[0.2, 0.25, 0.2, 0.15, 0.1, 0.06, 0.04])
    
    payment_methods = ['Cash on Delivery', 'UPI', 'Credit Card', 'Debit Card']
    payment_weights = [0.40, 0.35, 0.15, 0.10]
    payment_choices = np.random.choice(payment_methods, size=num_samples, p=payment_weights)
    
    quantity = []
    for cat in category_choices:
        if cat in ['Apparel', 'Footwear'] and np.random.rand() < 0.28:
            quantity.append(np.random.choice([2, 3], p=[0.75, 0.25]))
        else:
            quantity.append(np.random.choice([1, 2], p=[0.92, 0.08]))
    quantity = np.array(quantity)

    delivery_tiers = ['Tier 1', 'Tier 2', 'Tier 3']
    tier_choices = np.random.choice(delivery_tiers, size=num_samples, p=[0.45, 0.35, 0.20])
    
    is_festive_sale = np.random.choice([0, 1], size=num_samples, p=[0.75, 0.25])

    # 4. Realistic Probabilistic Formula (Respecting Human Stochasticity)
    logit = -2.1
    logit += customer_return_rate * 4.8
    logit += np.where(category_choices == 'Apparel', 1.25, 0.0)
    logit += np.where(category_choices == 'Footwear', 0.90, 0.0)
    logit += np.where(category_choices == 'Beauty', -0.95, 0.0)
    logit += np.where(category_choices == 'Home & Kitchen', -0.60, 0.0)
    
    logit += np.where(payment_choices == 'Cash on Delivery', 1.35, 0.0)
    logit -= np.where(payment_choices == 'Credit Card', 0.65, 0.0)
    logit += (discount_percent / 100.0) * 1.1
    logit += (quantity - 1) * 1.1
    logit += np.where(tier_choices == 'Tier 3', 0.35, 0.0)
    logit += is_festive_sale * 0.35
    logit -= np.where(customer_age_days > 365, 0.45, 0.0)

    # Convert logit to probability via Sigmoid
    prob_return = 1.0 / (1.0 + np.exp(-logit))
    
    # Real Probabilistic Human Sampling (e.g., an 80% risk item is kept 20% of the time)
    actual_return = (np.random.rand(num_samples) < prob_return).astype(int)

    df = pd.DataFrame({
        'order_id': order_ids,
        'customer_age_days': customer_age_days,
        'past_orders_count': past_orders_count,
        'customer_past_return_rate': customer_return_rate,
        'product_category': category_choices,
        'product_price': product_price,
        'discount_percent': discount_percent,
        'payment_method': payment_choices,
        'item_quantity': quantity,
        'delivery_tier': tier_choices,
        'is_festive_sale': is_festive_sale,
        'is_returned': actual_return
    })

    return df

if __name__ == '__main__':
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, 'uncart_orders_dataset.csv')
    
    print("Generating realistically calibrated dataset for UNCart (ML_29)...")
    df = generate_uncart_dataset(num_samples=6000)
    df.to_csv(output_path, index=False)
    
    return_rate = df['is_returned'].mean() * 100
    print(f"\n[SUCCESS] Dataset created: {output_path}")
    print(f"Total Rows: {len(df)}")
    print(f"Overall Return Rate: {return_rate:.2f}%\n")
    print("Return Rate by Category:")
    print((df.groupby('product_category')['is_returned'].mean() * 100).round(2).to_string())
    print("\nReturn Rate by Payment Method:")
    print((df.groupby('payment_method')['is_returned'].mean() * 100).round(2).to_string())
