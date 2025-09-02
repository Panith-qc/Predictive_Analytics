import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class CompetitorIntelligence:
    """Competitor intelligence and monitoring system"""
    
    def __init__(self):
        self.alerts = []
        self.analysis_results = {}
    
    def analyze_competitor_pricing(self, df, sku):
        """Analyze competitor pricing patterns for a specific SKU"""
        sku_data = df[df['sku'] == sku].copy()
        
        if len(sku_data) == 0:
            return None
        
        # Price analysis
        our_avg_price = sku_data['our_price'].mean()
        comp_avg_price = sku_data['competitor_price'].mean()
        
        # Price advantage/disadvantage
        price_advantage = (comp_avg_price - our_avg_price) / our_avg_price * 100
        
        # Volatility analysis
        our_price_volatility = sku_data['our_price'].std() / our_avg_price * 100
        comp_price_volatility = sku_data['competitor_price'].std() / comp_avg_price * 100
        
        # Recent trends
        recent_data = sku_data.tail(30)
        our_recent_trend = np.polyfit(range(len(recent_data)), recent_data['our_price'], 1)[0]
        comp_recent_trend = np.polyfit(range(len(recent_data)), recent_data['competitor_price'], 1)[0]
        
        return {
            'sku': sku,
            'our_avg_price': our_avg_price,
            'competitor_avg_price': comp_avg_price,
            'price_advantage_pct': price_advantage,
            'our_price_volatility': our_price_volatility,
            'competitor_price_volatility': comp_price_volatility,
            'our_price_trend': our_recent_trend,
            'competitor_price_trend': comp_recent_trend,
            'avg_buy_box_share': sku_data['buy_box_share'].mean()
        }
    
    def detect_price_anomalies(self, df, sku, threshold=0.15):
        """Detect unusual price movements"""
        sku_data = df[df['sku'] == sku].copy()
        
        if len(sku_data) < 7:
            return []
        
        # Calculate rolling averages
        sku_data['our_price_ma'] = sku_data['our_price'].rolling(7).mean()
        sku_data['comp_price_ma'] = sku_data['competitor_price'].rolling(7).mean()
        
        # Detect anomalies
        anomalies = []
        
        for idx, row in sku_data.iterrows():
            if pd.isna(row['our_price_ma']) or pd.isna(row['comp_price_ma']):
                continue
                
            # Check for significant price changes
            our_change = abs(row['our_price'] - row['our_price_ma']) / row['our_price_ma']
            comp_change = abs(row['competitor_price'] - row['comp_price_ma']) / row['comp_price_ma']
            
            if our_change > threshold:
                anomalies.append({
                    'date': row['date'],
                    'type': 'our_price_anomaly',
                    'severity': 'high' if our_change > 0.3 else 'medium',
                    'description': f"Our price changed by {our_change*100:.1f}%"
                })
            
            if comp_change > threshold:
                anomalies.append({
                    'date': row['date'],
                    'type': 'competitor_price_anomaly',
                    'severity': 'high' if comp_change > 0.3 else 'medium',
                    'description': f"Competitor price changed by {comp_change*100:.1f}%"
                })
        
        return anomalies
    
    def generate_competitive_insights(self, df, sku):
        """Generate actionable competitive insights"""
        analysis = self.analyze_competitor_pricing(df, sku)
        
        if not analysis:
            return []
        
        insights = []
        
        # Price positioning insights
        if analysis['price_advantage_pct'] > 10:
            insights.append({
                'type': 'opportunity',
                'priority': 'high',
                'insight': f"We are {analysis['price_advantage_pct']:.1f}% cheaper than competitor - consider price increase",
                'action': 'Consider raising price to capture more margin'
            })
        elif analysis['price_advantage_pct'] < -10:
            insights.append({
                'type': 'threat',
                'priority': 'high',
                'insight': f"We are {abs(analysis['price_advantage_pct']):.1f}% more expensive than competitor",
                'action': 'Consider price reduction or value proposition enhancement'
            })
        
        # Buy box share insights
        if analysis['avg_buy_box_share'] < 0.3:
            insights.append({
                'type': 'threat',
                'priority': 'medium',
                'insight': f"Low buy box share ({analysis['avg_buy_box_share']:.1%})",
                'action': 'Improve price competitiveness or product listing optimization'
            })
        elif analysis['avg_buy_box_share'] > 0.7:
            insights.append({
                'type': 'opportunity',
                'priority': 'low',
                'insight': f"Strong buy box share ({analysis['avg_buy_box_share']:.1%})",
                'action': 'Maintain current strategy'
            })
        
        # Volatility insights
        if analysis['competitor_price_volatility'] > analysis['our_price_volatility'] * 2:
            insights.append({
                'type': 'opportunity',
                'priority': 'medium',
                'insight': "Competitor shows high price volatility - opportunity for stable pricing strategy",
                'action': 'Maintain consistent pricing to build customer trust'
            })
        
        return insights
    
    def monitor_market_trends(self, df):
        """Monitor overall market trends across all SKUs"""
        market_analysis = {}
        
        # Overall market metrics
        total_sales = df['sales'].sum()
        avg_price_change = df.groupby('sku').apply(
            lambda x: x['our_price'].pct_change().mean()
        ).mean()
        
        # Category performance
        category_performance = df.groupby('category').agg({
            'sales': 'sum',
            'our_price': 'mean',
            'buy_box_share': 'mean'
        }).round(2)
        
        # Top performing SKUs
        sku_performance = df.groupby('sku').agg({
            'sales': 'sum',
            'buy_box_share': 'mean'
        }).sort_values('sales', ascending=False).head(5)
        
        market_analysis = {
            'total_sales': total_sales,
            'avg_price_change': avg_price_change,
            'category_performance': category_performance,
            'top_skus': sku_performance
        }
        
        return market_analysis

# Global instance
competitor_intelligence = CompetitorIntelligence()