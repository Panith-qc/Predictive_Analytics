import pandas as pd
import numpy as np
from scipy import stats

class SafetyStockOptimizer:
    """Advanced safety stock optimization using Monte Carlo simulation"""
    
    def __init__(self):
        self.simulation_results = {}
        self.recommendations = {}
    
    def calculate_demand_statistics(self, df, sku):
        """Calculate demand statistics for a SKU"""
        sku_data = df[df['sku'] == sku].copy()
        
        if len(sku_data) < 30:
            return None
        
        # Basic statistics
        demand_mean = sku_data['demand'].mean()
        demand_std = sku_data['demand'].std()
        demand_cv = demand_std / demand_mean if demand_mean > 0 else 0
        
        # Lead time analysis (simulated)
        lead_time_mean = 7  # days
        lead_time_std = 2   # days
        
        # Seasonality analysis
        sku_data['month'] = pd.to_datetime(sku_data['date']).dt.month
        seasonal_variation = sku_data.groupby('month')['demand'].mean().std()
        
        return {
            'demand_mean': demand_mean,
            'demand_std': demand_std,
            'demand_cv': demand_cv,
            'lead_time_mean': lead_time_mean,
            'lead_time_std': lead_time_std,
            'seasonal_variation': seasonal_variation,
            'data_points': len(sku_data)
        }
    
    def monte_carlo_simulation(self, stats_dict, service_level=0.95, num_simulations=10000):
        """Run Monte Carlo simulation for safety stock optimization"""
        
        if not stats_dict:
            return None
        
        # Simulation parameters
        demand_mean = stats_dict['demand_mean']
        demand_std = stats_dict['demand_std']
        lead_time_mean = stats_dict['lead_time_mean']
        lead_time_std = stats_dict['lead_time_std']
        
        # Run simulations
        stockout_counts = []
        
        for safety_stock in range(0, int(demand_mean * 2), 5):
            stockouts = 0
            
            for _ in range(num_simulations):
                # Simulate lead time
                lead_time = max(1, np.random.normal(lead_time_mean, lead_time_std))
                
                # Simulate demand during lead time
                lead_time_demand = 0
                for _ in range(int(lead_time)):
                    daily_demand = max(0, np.random.normal(demand_mean, demand_std))
                    lead_time_demand += daily_demand
                
                # Check if stockout occurs
                if lead_time_demand > (demand_mean * lead_time_mean + safety_stock):
                    stockouts += 1
            
            stockout_rate = stockouts / num_simulations
            stockout_counts.append({
                'safety_stock': safety_stock,
                'stockout_rate': stockout_rate,
                'service_level': 1 - stockout_rate
            })
        
        # Find optimal safety stock for desired service level
        optimal_safety_stock = None
        for result in stockout_counts:
            if result['service_level'] >= service_level:
                optimal_safety_stock = result['safety_stock']
                break
        
        return {
            'simulation_results': stockout_counts,
            'optimal_safety_stock': optimal_safety_stock,
            'target_service_level': service_level
        }
    
    def calculate_reorder_point(self, stats_dict, safety_stock):
        """Calculate reorder point"""
        if not stats_dict or safety_stock is None:
            return None
        
        lead_time_demand = stats_dict['demand_mean'] * stats_dict['lead_time_mean']
        reorder_point = lead_time_demand + safety_stock
        
        return {
            'lead_time_demand': lead_time_demand,
            'safety_stock': safety_stock,
            'reorder_point': reorder_point
        }
    
    def optimize_inventory_policy(self, df, sku, service_level=0.95):
        """Optimize complete inventory policy for a SKU"""
        
        # Get demand statistics
        stats_dict = self.calculate_demand_statistics(df, sku)
        
        if not stats_dict:
            return None
        
        # Run Monte Carlo simulation
        simulation_result = self.monte_carlo_simulation(stats_dict, service_level)
        
        if not simulation_result:
            return None
        
        # Calculate reorder point
        reorder_info = self.calculate_reorder_point(
            stats_dict, 
            simulation_result['optimal_safety_stock']
        )
        
        # Economic Order Quantity (simplified)
        annual_demand = stats_dict['demand_mean'] * 365
        holding_cost_rate = 0.25  # 25% per year
        ordering_cost = 50  # $50 per order
        
        if annual_demand > 0:
            eoq = np.sqrt((2 * annual_demand * ordering_cost) / 
                         (stats_dict['demand_mean'] * 0.1 * holding_cost_rate))
        else:
            eoq = stats_dict['demand_mean'] * 30  # 30 days supply
        
        # Risk assessment
        risk_factors = self._assess_risk_factors(stats_dict)
        
        return {
            'sku': sku,
            'demand_statistics': stats_dict,
            'simulation_results': simulation_result,
            'reorder_point': reorder_info,
            'economic_order_quantity': eoq,
            'risk_assessment': risk_factors,
            'recommendations': self._generate_recommendations(
                stats_dict, simulation_result, reorder_info, eoq
            )
        }
    
    def _assess_risk_factors(self, stats_dict):
        """Assess various risk factors"""
        risk_score = 0
        risk_factors = []
        
        # High demand variability
        if stats_dict['demand_cv'] > 0.5:
            risk_score += 2
            risk_factors.append("High demand variability")
        
        # Seasonal variation
        if stats_dict['seasonal_variation'] > stats_dict['demand_mean'] * 0.3:
            risk_score += 1
            risk_factors.append("Significant seasonal variation")
        
        # Limited data
        if stats_dict['data_points'] < 90:
            risk_score += 1
            risk_factors.append("Limited historical data")
        
        risk_level = "Low"
        if risk_score >= 3:
            risk_level = "High"
        elif risk_score >= 2:
            risk_level = "Medium"
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_factors': risk_factors
        }
    
    def _generate_recommendations(self, stats_dict, simulation_result, reorder_info, eoq):
        """Generate actionable recommendations"""
        recommendations = []
        
        # Safety stock recommendation
        if simulation_result['optimal_safety_stock']:
            recommendations.append(
                f"Maintain {simulation_result['optimal_safety_stock']:.0f} units as safety stock"
            )
        
        # Reorder point recommendation
        if reorder_info:
            recommendations.append(
                f"Set reorder point at {reorder_info['reorder_point']:.0f} units"
            )
        
        # Order quantity recommendation
        recommendations.append(f"Optimal order quantity: {eoq:.0f} units")
        
        # Risk-based recommendations
        if stats_dict['demand_cv'] > 0.5:
            recommendations.append("Consider demand forecasting improvements due to high variability")
        
        if stats_dict['seasonal_variation'] > stats_dict['demand_mean'] * 0.3:
            recommendations.append("Implement seasonal inventory planning")
        
        return recommendations

# Global instance
safety_stock_optimizer = SafetyStockOptimizer()