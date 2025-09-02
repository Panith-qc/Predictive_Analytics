import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class PurchaseDecisionEngine:
    """AI-powered purchase decision engine"""
    
    def __init__(self):
        self.decision_weights = {
            'demand_forecast': 0.25,
            'inventory_level': 0.20,
            'price_competitiveness': 0.15,
            'profit_margin': 0.15,
            'market_trend': 0.10,
            'seasonality': 0.15
        }
    
    def calculate_demand_score(self, df, sku):
        """Calculate demand-based score (0-100)"""
        sku_data = df[df['sku'] == sku].copy()
        
        if len(sku_data) < 30:
            return 50  # Neutral score for insufficient data
        
        # Recent demand trend
        recent_data = sku_data.tail(30)
        demand_trend = np.polyfit(range(len(recent_data)), recent_data['demand'], 1)[0]
        
        # Demand growth rate
        if len(sku_data) >= 60:
            recent_avg = recent_data['demand'].mean()
            older_avg = sku_data.iloc[-60:-30]['demand'].mean()
            growth_rate = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
        else:
            growth_rate = 0
        
        # Score calculation
        trend_score = min(100, max(0, 50 + demand_trend * 10))
        growth_score = min(100, max(0, 50 + growth_rate * 100))
        
        return (trend_score + growth_score) / 2
    
    def calculate_inventory_score(self, df, sku):
        """Calculate inventory-based score (0-100)"""
        sku_data = df[df['sku'] == sku].copy()
        
        if len(sku_data) == 0:
            return 50
        
        current_stock = sku_data['our_stock'].iloc[-1]
        avg_demand = sku_data['demand'].tail(30).mean()
        
        # Days of inventory
        days_of_inventory = current_stock / avg_demand if avg_demand > 0 else 30
        
        # Optimal range: 15-45 days
        if days_of_inventory < 15:
            score = 100  # High score - need to buy
        elif days_of_inventory > 45:
            score = 20   # Low score - overstocked
        else:
            # Linear interpolation between 15-45 days
            score = 100 - ((days_of_inventory - 15) / 30) * 80
        
        return max(0, min(100, score))
    
    def calculate_price_competitiveness_score(self, df, sku):
        """Calculate price competitiveness score (0-100)"""
        sku_data = df[df['sku'] == sku].copy()
        
        if len(sku_data) == 0:
            return 50
        
        recent_data = sku_data.tail(30)
        our_avg_price = recent_data['our_price'].mean()
        comp_avg_price = recent_data['competitor_price'].mean()
        
        # Price advantage
        price_ratio = our_avg_price / comp_avg_price if comp_avg_price > 0 else 1
        
        # Score: lower price = higher score
        if price_ratio <= 0.9:
            score = 90  # We're significantly cheaper
        elif price_ratio <= 0.95:
            score = 75  # We're moderately cheaper
        elif price_ratio <= 1.05:
            score = 60  # Prices are similar
        elif price_ratio <= 1.1:
            score = 40  # We're moderately more expensive
        else:
            score = 20  # We're significantly more expensive
        
        return score
    
    def calculate_profit_margin_score(self, df, sku):
        """Calculate profit margin score (0-100)"""
        sku_data = df[df['sku'] == sku].copy()
        
        if len(sku_data) == 0:
            return 50
        
        # Estimate cost (assuming 60% of selling price as cost)
        recent_price = sku_data['our_price'].tail(30).mean()
        estimated_cost = recent_price * 0.6
        profit_margin = (recent_price - estimated_cost) / recent_price
        
        # Score based on margin
        if profit_margin >= 0.5:
            score = 100
        elif profit_margin >= 0.3:
            score = 80
        elif profit_margin >= 0.2:
            score = 60
        elif profit_margin >= 0.1:
            score = 40
        else:
            score = 20
        
        return score
    
    def calculate_market_trend_score(self, df, sku):
        """Calculate market trend score (0-100)"""
        sku_data = df[df['sku'] == sku].copy()
        
        if len(sku_data) < 60:
            return 50
        
        # Buy box share trend
        recent_bb = sku_data['buy_box_share'].tail(30).mean()
        older_bb = sku_data['buy_box_share'].iloc[-60:-30].mean()
        bb_trend = (recent_bb - older_bb) / older_bb if older_bb > 0 else 0
        
        # Sales trend
        recent_sales = sku_data['sales'].tail(30).mean()
        older_sales = sku_data['sales'].iloc[-60:-30].mean()
        sales_trend = (recent_sales - older_sales) / older_sales if older_sales > 0 else 0
        
        # Combined score
        trend_score = 50 + (bb_trend * 100) + (sales_trend * 50)
        
        return max(0, min(100, trend_score))
    
    def calculate_seasonality_score(self, df, sku):
        """Calculate seasonality score (0-100)"""
        sku_data = df[df['sku'] == sku].copy()
        
        if len(sku_data) < 90:
            return 50
        
        # Current month
        current_month = datetime.now().month
        
        # Historical performance for current month
        sku_data['month'] = pd.to_datetime(sku_data['date']).dt.month
        current_month_avg = sku_data[sku_data['month'] == current_month]['demand'].mean()
        overall_avg = sku_data['demand'].mean()
        
        seasonal_factor = current_month_avg / overall_avg if overall_avg > 0 else 1
        
        # Score based on seasonal factor
        score = 50 + (seasonal_factor - 1) * 50
        
        return max(0, min(100, score))
    
    def calculate_ml_score(self, df, sku):
        """Calculate comprehensive ML-based purchase score"""
        
        # Calculate individual scores
        demand_score = self.calculate_demand_score(df, sku)
        inventory_score = self.calculate_inventory_score(df, sku)
        price_score = self.calculate_price_competitiveness_score(df, sku)
        margin_score = self.calculate_profit_margin_score(df, sku)
        trend_score = self.calculate_market_trend_score(df, sku)
        season_score = self.calculate_seasonality_score(df, sku)
        
        # Weighted combination
        ml_score = (
            demand_score * self.decision_weights['demand_forecast'] +
            inventory_score * self.decision_weights['inventory_level'] +
            price_score * self.decision_weights['price_competitiveness'] +
            margin_score * self.decision_weights['profit_margin'] +
            trend_score * self.decision_weights['market_trend'] +
            season_score * self.decision_weights['seasonality']
        )
        
        return {
            'ml_score': round(ml_score, 1),
            'component_scores': {
                'demand_forecast': round(demand_score, 1),
                'inventory_level': round(inventory_score, 1),
                'price_competitiveness': round(price_score, 1),
                'profit_margin': round(margin_score, 1),
                'market_trend': round(trend_score, 1),
                'seasonality': round(season_score, 1)
            }
        }
    
    def predict_roi(self, df, sku, investment_amount):
        """Predict ROI for purchase decision"""
        sku_data = df[df['sku'] == sku].copy()
        
        if len(sku_data) == 0:
            return 0
        
        # Recent metrics
        recent_data = sku_data.tail(30)
        avg_price = recent_data['our_price'].mean()
        avg_demand = recent_data['demand'].mean()
        
        # Estimate units that can be purchased
        units_to_purchase = investment_amount / (avg_price * 0.6)  # Assuming 60% cost ratio
        
        # Estimate sales over next 90 days
        estimated_sales = min(units_to_purchase, avg_demand * 90)
        
        # Revenue and profit calculation
        estimated_revenue = estimated_sales * avg_price
        estimated_profit = estimated_revenue - investment_amount
        
        # ROI calculation
        roi = (estimated_profit / investment_amount) * 100 if investment_amount > 0 else 0
        
        return max(-50, min(200, roi))  # Cap between -50% and 200%
    
    def generate_purchase_recommendations(self, df, budget=50000):
        """Generate purchase recommendations for all SKUs"""
        recommendations = []
        
        for sku in df['sku'].unique():
            # Calculate ML score
            score_result = self.calculate_ml_score(df, sku)
            ml_score = score_result['ml_score']
            
            # Estimate investment needed
            sku_data = df[df['sku'] == sku]
            avg_price = sku_data['our_price'].tail(30).mean()
            current_stock = sku_data['our_stock'].iloc[-1]
            avg_demand = sku_data['demand'].tail(30).mean()
            
            # Suggested purchase quantity (to reach 30 days inventory)
            target_inventory = avg_demand * 30
            purchase_quantity = max(0, target_inventory - current_stock)
            investment_needed = purchase_quantity * avg_price * 0.6  # Cost estimate
            
            # Predict ROI
            predicted_roi = self.predict_roi(df, sku, investment_needed)
            
            # Priority classification
            if ml_score >= 80:
                priority = "High"
            elif ml_score >= 60:
                priority = "Medium"
            else:
                priority = "Low"
            
            # Action recommendation
            if ml_score >= 75 and investment_needed <= budget * 0.2:
                action = "Buy Now"
            elif ml_score >= 60:
                action = "Monitor"
            else:
                action = "Hold"
            
            recommendations.append({
                'sku': sku,
                'ml_score': ml_score,
                'priority': priority,
                'investment_needed': investment_needed,
                'predicted_roi': predicted_roi,
                'purchase_quantity': purchase_quantity,
                'action': action,
                'component_scores': score_result['component_scores']
            })
        
        # Sort by ML score
        recommendations = sorted(recommendations, key=lambda x: x['ml_score'], reverse=True)
        
        return recommendations

# Global instance
purchase_decision_engine = PurchaseDecisionEngine()