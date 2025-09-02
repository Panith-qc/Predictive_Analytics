import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_supply_chain_data(num_skus=20, days=90):
    """
    Generate realistic supply chain data for the forecasting platform
    """
    np.random.seed(42)
    random.seed(42)
    
    # SKU categories and base prices
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books']
    sku_data = []
    
    # Generate SKU master data
    for i in range(num_skus):
        sku_id = f"SKU-{1000 + i}"
        category = random.choice(categories)
        base_price = random.uniform(15, 200)
        
        sku_data.append({
            'sku_id': sku_id,
            'category': category,
            'base_price': base_price,
            'cost': base_price * 0.6,
            'weight': random.uniform(0.1, 5.0),
            'supplier': f"Supplier-{random.randint(1, 10)}"
        })
    
    # Generate time series data
    data = []
    start_date = datetime.now() - timedelta(days=days)
    
    for sku in sku_data:
        # Generate seasonal and trend patterns
        base_demand = random.uniform(50, 500)
        trend = random.uniform(-0.5, 1.0)
        seasonality_amplitude = base_demand * 0.3
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            
            # Add trend
            trend_component = trend * day
            
            # Add seasonality (weekly pattern)
            seasonal_component = seasonality_amplitude * np.sin(2 * np.pi * day / 7)
            
            # Add random noise
            noise = np.random.normal(0, base_demand * 0.1)
            
            # Calculate demand
            demand = max(0, base_demand + trend_component + seasonal_component + noise)
            
            # Calculate other metrics
            revenue = demand * sku['base_price']
            inventory = max(0, random.uniform(demand * 0.5, demand * 3))
            lead_time = random.randint(3, 14)
            
            data.append({
                'date': current_date,
                'sku_id': sku['sku_id'],
                'category': sku['category'],
                'demand': int(demand),
                'revenue': revenue,
                'inventory': int(inventory),
                'price': sku['base_price'] * random.uniform(0.9, 1.1),
                'cost': sku['cost'],
                'lead_time': lead_time,
                'supplier': sku['supplier'],
                'stockout_risk': max(0, min(1, (demand - inventory) / demand)) if demand > 0 else 0
            })
    
    return pd.DataFrame(data)

def generate_competitor_data():
    """Generate competitor intelligence data"""
    competitors = ['CompetitorA', 'CompetitorB', 'CompetitorC', 'CompetitorD']
    
    data = []
    for comp in competitors:
        data.append({
            'competitor': comp,
            'market_share': random.uniform(15, 35),
            'avg_price': random.uniform(40, 60),
            'rating': random.uniform(3.5, 4.8),
            'reviews': random.randint(1000, 10000),
            'growth_rate': random.uniform(-5, 15)
        })
    
    return pd.DataFrame(data)

def generate_forecast_accuracy_data():
    """Generate historical forecast accuracy data"""
    models = ['ARIMA', 'Random Forest', 'Gradient Boosting', 'Neural Network', 'Ensemble']
    
    data = []
    for model in models:
        for month in range(12):
            accuracy = random.uniform(75, 95) if model != 'Ensemble' else random.uniform(90, 98)
            data.append({
                'model': model,
                'month': month + 1,
                'accuracy': accuracy,
                'mae': random.uniform(5, 25),
                'rmse': random.uniform(8, 35)
            })
    
    return pd.DataFrame(data)