import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Advanced ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import scipy.stats as stats
from scipy.optimize import minimize
import joblib

st.set_page_config(
    page_title="🚀 Amazon Inventory AI System",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: #ffffff;
    }
    
    .main .block-container {
        background: rgba(255, 255, 255, 0.05);
        padding: 2rem;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
        padding: 20px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 10px 0;
    }
    
    h1, h2, h3 {
        color: #ffffff !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)

class AmazonInventoryAI:
    def __init__(self):
        self.demand_model = None
        self.buybox_model = None
        self.price_elasticity_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def generate_amazon_data(self, n_skus=50, days=365):
        """Generate realistic Amazon marketplace data"""
        np.random.seed(42)
        
        # SKU categories
        categories = ['Electronics', 'Home & Kitchen', 'Sports', 'Books', 'Clothing']
        brands = ['Brand_A', 'Brand_B', 'Brand_C', 'Brand_D', 'Brand_E']
        
        data = []
        
        for sku_id in range(1, n_skus + 1):
            category = np.random.choice(categories)
            brand = np.random.choice(brands)
            
            # Base demand influenced by category
            base_demand = {
                'Electronics': 50, 'Home & Kitchen': 30, 'Sports': 25, 
                'Books': 15, 'Clothing': 40
            }[category]
            
            # Generate time series data
            dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
            
            for i, date in enumerate(dates):
                # Seasonality effects
                day_of_year = date.timetuple().tm_yday
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
                
                # Weekly patterns
                weekly_factor = 1.2 if date.weekday() < 5 else 0.8
                
                # Trend
                trend_factor = 1 + (i / days) * 0.1
                
                # Random noise
                noise = np.random.normal(1, 0.2)
                
                # Calculate demand
                demand = int(base_demand * seasonal_factor * weekly_factor * trend_factor * noise)
                demand = max(1, demand)  # Ensure positive demand
                
                # Pricing
                base_price = np.random.uniform(20, 200)
                our_price = base_price * np.random.uniform(0.9, 1.1)
                competitor_price = base_price * np.random.uniform(0.85, 1.15)
                
                # Buy box share (influenced by price competitiveness)
                price_advantage = (competitor_price - our_price) / competitor_price
                buybox_base = 0.6 + 0.3 * price_advantage
                buybox_share = np.clip(buybox_base + np.random.normal(0, 0.1), 0.1, 0.95)
                
                # Inventory levels
                current_inventory = np.random.randint(10, 500)
                
                # Lead time (days)
                lead_time = np.random.randint(7, 30)
                
                # Competitor inventory estimate
                competitor_inventory = np.random.choice(['Low', 'Medium', 'High'], p=[0.3, 0.5, 0.2])
                
                # Sales (demand * buybox_share)
                sales = int(demand * buybox_share)
                
                data.append({
                    'date': date,
                    'sku_id': f'SKU_{sku_id:03d}',
                    'category': category,
                    'brand': brand,
                    'demand': demand,
                    'sales': sales,
                    'our_price': round(our_price, 2),
                    'competitor_price': round(competitor_price, 2),
                    'buybox_share': round(buybox_share, 3),
                    'current_inventory': current_inventory,
                    'lead_time': lead_time,
                    'competitor_inventory': competitor_inventory,
                    'day_of_week': date.weekday(),
                    'month': date.month,
                    'is_weekend': 1 if date.weekday() >= 5 else 0
                })
        
        return pd.DataFrame(data)
    
    def build_demand_forecasting_model(self, data):
        """Build advanced demand forecasting model with seasonality and competitor effects"""
        st.write("🧠 **Building Advanced Demand Forecasting Model...**")
        
        # Feature engineering
        features = ['our_price', 'competitor_price', 'buybox_share', 'day_of_week', 
                   'month', 'is_weekend', 'lead_time']
        
        # Add price ratio
        data['price_ratio'] = data['our_price'] / data['competitor_price']
        features.append('price_ratio')
        
        # Add moving averages
        data = data.sort_values(['sku_id', 'date'])
        data['demand_ma_7'] = data.groupby('sku_id')['demand'].rolling(window=7, min_periods=1).mean().reset_index(0, drop=True)
        data['demand_ma_30'] = data.groupby('sku_id')['demand'].rolling(window=30, min_periods=1).mean().reset_index(0, drop=True)
        features.extend(['demand_ma_7', 'demand_ma_30'])
        
        # Encode categorical variables
        data['category_encoded'] = self.label_encoder.fit_transform(data['category'])
        features.append('category_encoded')
        
        # Prepare training data
        X = data[features].fillna(0)
        y = data['demand']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build ensemble model
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Ridge Regression': Ridge(alpha=1.0)
        }
        
        best_model = None
        best_score = float('inf')
        model_results = {}
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            model_results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
            
            if mae < best_score:
                best_score = mae
                best_model = model
                
        self.demand_model = best_model
        
        # Display results
        results_df = pd.DataFrame(model_results).T
        st.write("**Model Performance Comparison:**")
        st.dataframe(results_df)
        
        return model_results
    
    def calculate_safety_stock(self, data, service_level=0.95):
        """Calculate optimal safety stock using statistical methods"""
        st.write("📊 **Calculating Optimal Safety Stock Levels...**")
        
        safety_stock_results = []
        
        for sku in data['sku_id'].unique():
            sku_data = data[data['sku_id'] == sku].copy()
            
            # Calculate demand statistics
            mean_demand = sku_data['demand'].mean()
            std_demand = sku_data['demand'].std()
            mean_lead_time = sku_data['lead_time'].mean()
            
            # Z-score for service level
            z_score = stats.norm.ppf(service_level)
            
            # Safety stock calculation
            safety_stock = z_score * std_demand * np.sqrt(mean_lead_time)
            
            # Reorder point
            reorder_point = mean_demand * mean_lead_time + safety_stock
            
            safety_stock_results.append({
                'sku_id': sku,
                'mean_demand': round(mean_demand, 2),
                'std_demand': round(std_demand, 2),
                'mean_lead_time': round(mean_lead_time, 1),
                'safety_stock': round(safety_stock, 0),
                'reorder_point': round(reorder_point, 0),
                'service_level': service_level
            })
        
        return pd.DataFrame(safety_stock_results)
    
    def build_buybox_prediction_model(self, data):
        """Build buy box share prediction model"""
        st.write("🏆 **Building Buy Box Share Prediction Model...**")
        
        # Features for buy box prediction
        features = ['our_price', 'competitor_price', 'price_ratio', 'current_inventory']
        
        X = data[features].fillna(0)
        y = data['buybox_share']
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.buybox_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.buybox_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.buybox_model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        st.write(f"**Buy Box Model Performance:** MAE: {mae:.3f}, R²: {r2:.3f}")
        
        return {'MAE': mae, 'R2': r2}
    
    def detect_stockout_risks(self, data, safety_stock_df):
        """Detect SKUs at risk of stockout"""
        st.write("⚠️ **Detecting Stockout Risks...**")
        
        # Merge with safety stock data
        risk_data = data.merge(safety_stock_df[['sku_id', 'reorder_point']], on='sku_id', how='left')
        
        # Get latest inventory for each SKU
        latest_data = risk_data.groupby('sku_id').last().reset_index()
        
        # Calculate days of inventory remaining
        latest_data['days_remaining'] = latest_data['current_inventory'] / latest_data['demand']
        
        # Identify risks
        latest_data['risk_level'] = 'Low'
        latest_data.loc[latest_data['current_inventory'] <= latest_data['reorder_point'], 'risk_level'] = 'High'
        latest_data.loc[latest_data['days_remaining'] <= 7, 'risk_level'] = 'Critical'
        
        # Sort by risk
        risk_order = {'Critical': 3, 'High': 2, 'Low': 1}
        latest_data['risk_score'] = latest_data['risk_level'].map(risk_order)
        latest_data = latest_data.sort_values('risk_score', ascending=False)
        
        return latest_data[['sku_id', 'category', 'current_inventory', 'reorder_point', 
                          'days_remaining', 'risk_level']].head(20)
    
    def automated_purchase_recommendations(self, data, safety_stock_df):
        """Generate automated purchase recommendations"""
        st.write("🤖 **Generating Automated Purchase Recommendations...**")
        
        recommendations = []
        
        # Get latest data for each SKU
        latest_data = data.groupby('sku_id').last().reset_index()
        latest_data = latest_data.merge(safety_stock_df, on='sku_id', how='left')
        
        for _, row in latest_data.iterrows():
            current_inv = row['current_inventory']
            reorder_point = row['reorder_point']
            mean_demand = row['mean_demand']
            lead_time = row['mean_lead_time']
            
            if current_inv <= reorder_point:
                # Calculate optimal order quantity (EOQ approximation)
                order_qty = int(mean_demand * lead_time * 2)  # 2x lead time demand
                
                # Priority based on how low inventory is
                priority = 'High' if current_inv <= reorder_point * 0.5 else 'Medium'
                
                recommendations.append({
                    'sku_id': row['sku_id'],
                    'category': row['category'],
                    'current_inventory': current_inv,
                    'reorder_point': reorder_point,
                    'recommended_order_qty': order_qty,
                    'priority': priority,
                    'estimated_cost': order_qty * row['our_price'] * 0.7,  # Assume 70% of selling price
                    'days_until_stockout': current_inv / mean_demand if mean_demand > 0 else 999
                })
        
        recommendations_df = pd.DataFrame(recommendations)
        if not recommendations_df.empty:
            recommendations_df = recommendations_df.sort_values('days_until_stockout')
        
        return recommendations_df

# Initialize the AI system
@st.cache_data
def load_data():
    ai_system = AmazonInventoryAI()
    data = ai_system.generate_amazon_data(n_skus=100, days=180)
    return ai_system, data

# Main application
st.title("🚀 Amazon Inventory AI System")
st.markdown("**Advanced ML-Powered Inventory Management for Amazon Sellers**")

# Load data
ai_system, data = load_data()

# Sidebar
st.sidebar.title("🎛️ Control Panel")
module = st.sidebar.selectbox(
    "Select Module",
    ["📊 Dashboard", "🧠 Demand Forecasting", "📦 Safety Stock Calculator", 
     "🏆 Buy Box Prediction", "⚠️ Stockout Alerts", "🤖 Purchase Recommendations"]
)

if module == "📊 Dashboard":
    st.header("📊 Executive Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_skus = data['sku_id'].nunique()
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total SKUs</h3>
            <h2>{total_skus}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_buybox = data['buybox_share'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>Avg Buy Box Share</h3>
            <h2>{avg_buybox:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_sales = data['sales'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Sales (Units)</h3>
            <h2>{total_sales:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        revenue = (data['sales'] * data['our_price']).sum()
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Revenue</h3>
            <h2>${revenue:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sales by Category")
        category_sales = data.groupby('category')['sales'].sum().reset_index()
        fig = px.bar(category_sales, x='category', y='sales', color='sales')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Buy Box Share Distribution")
        fig = px.histogram(data, x='buybox_share', nbins=20)
        st.plotly_chart(fig, use_container_width=True)

elif module == "🧠 Demand Forecasting":
    st.header("🧠 Advanced Demand Forecasting")
    
    if st.button("Train Demand Forecasting Model"):
        with st.spinner("Training advanced ML models..."):
            results = ai_system.build_demand_forecasting_model(data)
            st.success("✅ Demand forecasting model trained successfully!")
    
    # Forecast for specific SKU
    st.subheader("SKU Demand Forecast")
    selected_sku = st.selectbox("Select SKU", data['sku_id'].unique())
    
    if selected_sku:
        sku_data = data[data['sku_id'] == selected_sku].sort_values('date')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sku_data['date'], 
            y=sku_data['demand'],
            mode='lines+markers',
            name='Historical Demand',
            line=dict(color='#00ff88', width=2)
        ))
        
        fig.update_layout(
            title=f"Demand Pattern: {selected_sku}",
            xaxis_title="Date",
            yaxis_title="Demand (Units)",
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)

elif module == "📦 Safety Stock Calculator":
    st.header("📦 Safety Stock Calculator")
    
    service_level = st.slider("Service Level", 0.80, 0.99, 0.95, 0.01)
    
    if st.button("Calculate Safety Stock Levels"):
        with st.spinner("Calculating optimal safety stock..."):
            safety_stock_df = ai_system.calculate_safety_stock(data, service_level)
            st.success("✅ Safety stock levels calculated!")
            
            st.subheader("Safety Stock Recommendations")
            st.dataframe(safety_stock_df.head(20))
            
            # Visualization
            fig = px.scatter(safety_stock_df, x='mean_demand', y='safety_stock', 
                           color='mean_lead_time', size='std_demand',
                           title="Safety Stock vs Mean Demand")
            st.plotly_chart(fig, use_container_width=True)

elif module == "🏆 Buy Box Prediction":
    st.header("🏆 Buy Box Share Prediction")
    
    if st.button("Train Buy Box Model"):
        with st.spinner("Training buy box prediction model..."):
            results = ai_system.build_buybox_prediction_model(data)
            st.success("✅ Buy box prediction model trained!")
    
    # Price optimization
    st.subheader("Price Optimization Analysis")
    selected_sku = st.selectbox("Select SKU for Analysis", data['sku_id'].unique(), key="buybox_sku")
    
    if selected_sku:
        sku_data = data[data['sku_id'] == selected_sku]
        
        fig = px.scatter(sku_data, x='price_ratio', y='buybox_share',
                        title=f"Price Ratio vs Buy Box Share: {selected_sku}",
                        trendline="ols")
        st.plotly_chart(fig, use_container_width=True)

elif module == "⚠️ Stockout Alerts":
    st.header("⚠️ Stockout Risk Alerts")
    
    if st.button("Generate Stockout Alerts"):
        with st.spinner("Analyzing stockout risks..."):
            # First calculate safety stock
            safety_stock_df = ai_system.calculate_safety_stock(data)
            
            # Then detect risks
            risk_alerts = ai_system.detect_stockout_risks(data, safety_stock_df)
            
            st.subheader("High-Risk SKUs")
            
            # Color code by risk level
            def color_risk(val):
                if val == 'Critical':
                    return 'background-color: #ff4444; color: white'
                elif val == 'High':
                    return 'background-color: #ff8800; color: white'
                else:
                    return 'background-color: #44ff44; color: black'
            
            styled_df = risk_alerts.style.applymap(color_risk, subset=['risk_level'])
            st.dataframe(styled_df, use_container_width=True)

elif module == "🤖 Purchase Recommendations":
    st.header("🤖 Automated Purchase Recommendations")
    
    if st.button("Generate Purchase Recommendations"):
        with st.spinner("Generating intelligent purchase recommendations..."):
            # Calculate safety stock first
            safety_stock_df = ai_system.calculate_safety_stock(data)
            
            # Generate recommendations
            recommendations = ai_system.automated_purchase_recommendations(data, safety_stock_df)
            
            if not recommendations.empty:
                st.subheader("Recommended Purchases")
                
                # Priority color coding
                def color_priority(val):
                    if val == 'High':
                        return 'background-color: #ff4444; color: white'
                    elif val == 'Medium':
                        return 'background-color: #ff8800; color: white'
                    else:
                        return 'background-color: #44ff44; color: black'
                
                styled_recommendations = recommendations.style.applymap(color_priority, subset=['priority'])
                st.dataframe(styled_recommendations, use_container_width=True)
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_cost = recommendations['estimated_cost'].sum()
                    st.metric("Total Investment Required", f"${total_cost:,.0f}")
                
                with col2:
                    high_priority = len(recommendations[recommendations['priority'] == 'High'])
                    st.metric("High Priority SKUs", high_priority)
                
                with col3:
                    avg_days = recommendations['days_until_stockout'].mean()
                    st.metric("Avg Days Until Stockout", f"{avg_days:.1f}")
            else:
                st.info("✅ No immediate purchase recommendations. All inventory levels are adequate.")

# Footer
st.markdown("---")
st.markdown("**🚀 Amazon Inventory AI System** | Advanced ML-Powered Inventory Management")