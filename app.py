import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
except ImportError:
    st.error("Required ML libraries not installed. Please run: pip install scikit-learn")
    st.stop()

st.set_page_config(
    page_title="🛒 Amazon Seller Central AI",
    page_icon="🚀",
    layout="wide"
)

# Amazon-themed CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Amazon+Ember:wght@400;500;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #232F3E 0%, #131A22 50%, #FF9900 100%);
        font-family: 'Amazon Ember', 'Helvetica Neue', Arial, sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #232F3E 0%, #131A22 50%, #FF9900 100%);
    }
    
    .amazon-header {
        background: linear-gradient(90deg, #232F3E 0%, #37475A 100%);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        border: 2px solid #FF9900;
        box-shadow: 0 8px 32px rgba(255, 153, 0, 0.3);
    }
    
    .amazon-title {
        color: #FF9900;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .amazon-subtitle {
        color: #FFFFFF;
        font-size: 1.3rem;
        text-align: center;
        margin-top: 10px;
        opacity: 0.9;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8F9FA 100%);
        padding: 25px;
        border-radius: 15px;
        border-left: 5px solid #FF9900;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(255, 153, 0, 0.2);
    }
    
    .metric-title {
        color: #232F3E;
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 10px;
    }
    
    .metric-value {
        color: #FF9900;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .amazon-badge {
        background: linear-gradient(45deg, #FF9900, #FFB84D);
        padding: 8px 20px;
        border-radius: 25px;
        color: #232F3E;
        font-weight: 700;
        display: inline-block;
        margin: 15px auto;
        box-shadow: 0 4px 15px rgba(255, 153, 0, 0.3);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 4px 15px rgba(255, 153, 0, 0.3); }
        50% { box-shadow: 0 8px 25px rgba(255, 153, 0, 0.5); }
        100% { box-shadow: 0 4px 15px rgba(255, 153, 0, 0.3); }
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #232F3E 0%, #37475A 100%);
    }
    
    .chart-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .success-message {
        background: linear-gradient(45deg, #28a745, #20c997);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
        text-align: center;
        font-weight: 600;
    }
    
    h1, h2, h3 {
        color: #FF9900 !important;
        font-weight: 600 !important;
    }
    
    .stSelectbox > div > div {
        background-color: #FFFFFF;
        border: 2px solid #FF9900;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #FF9900, #FFB84D);
        color: #232F3E;
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        font-weight: 700;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 153, 0, 0.4);
    }
    
    .amazon-icon {
        font-size: 2rem;
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

class AmazonListingAI:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def generate_data(self, n_listings=100, days=30):
        """Generate realistic Amazon listing data"""
        np.random.seed(42)
        
        categories = ['📱 Electronics', '🏠 Home & Kitchen', '⚽ Sports & Outdoors', 
                     '💄 Beauty & Personal Care', '👕 Clothing & Fashion', '📚 Books & Media']
        brands = ['TechPro', 'HomeMax', 'SportElite', 'BeautyLux', 'StyleCraft', 'BookWise']
        
        data = []
        
        for listing_id in range(1, n_listings + 1):
            category = np.random.choice(categories)
            brand = np.random.choice(brands)
            base_price = np.random.uniform(15, 299)
            
            # Optimization factors
            title_score = np.random.uniform(35, 98)
            image_score = np.random.uniform(40, 95)
            seo_score = np.random.uniform(30, 90)
            price_score = np.random.uniform(40, 95)
            
            dates = pd.date_range(start='2023-06-01', periods=days, freq='D')
            
            for date in dates:
                # Calculate performance based on optimization scores
                base_conversion = 0.06
                optimization_factor = (title_score + image_score + seo_score + price_score) / 400
                conversion_rate = base_conversion * (1 + optimization_factor)
                
                impressions = np.random.randint(150, 8000)
                clicks = int(impressions * np.random.uniform(0.015, 0.12))
                sales = int(clicks * conversion_rate)
                revenue = sales * base_price
                
                data.append({
                    'date': date,
                    'listing_id': f'B{listing_id:06d}',
                    'category': category,
                    'brand': brand,
                    'price': round(base_price, 2),
                    'title_score': round(title_score, 1),
                    'image_score': round(image_score, 1),
                    'seo_score': round(seo_score, 1),
                    'price_score': round(price_score, 1),
                    'overall_score': round((title_score + image_score + seo_score + price_score) / 4, 1),
                    'impressions': impressions,
                    'clicks': clicks,
                    'conversion_rate': round(conversion_rate, 4),
                    'sales': sales,
                    'revenue': round(revenue, 2),
                    'ctr': round(clicks / impressions if impressions > 0 else 0, 4),
                    'bsr_rank': np.random.randint(1000, 50000)
                })
        
        return pd.DataFrame(data)
    
    def train_optimization_model(self, data):
        """Train ML model for optimization predictions"""
        features = ['title_score', 'image_score', 'seo_score', 'price_score', 'price']
        
        # Encode categorical data
        data_encoded = data.copy()
        data_encoded['category_encoded'] = self.label_encoder.fit_transform(data['category'])
        features.append('category_encoded')
        
        X = data_encoded[features].fillna(0)
        y = data_encoded['conversion_rate']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Ridge Regression': Ridge(alpha=1.0)
        }
        
        results = {}
        best_model = None
        best_score = float('inf')
        
        for name, model in models.items():
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'MAE': round(mae, 5),
                    'R²': round(r2, 4)
                }
                
                if mae < best_score:
                    best_score = mae
                    best_model = model
                    
            except Exception as e:
                st.error(f"Error training {name}: {str(e)}")
                continue
        
        return results, best_model

# Initialize system
@st.cache_data
def load_data():
    ai_system = AmazonListingAI()
    data = ai_system.generate_data(n_listings=75, days=30)
    return ai_system, data

# Main application
st.markdown("""
<div class="amazon-header">
    <h1 class="amazon-title">🛒 Amazon Seller Central AI</h1>
    <p class="amazon-subtitle">Advanced Listing Optimization & Performance Analytics Platform</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="amazon-badge">🚀 Powered by Machine Learning & Big Data Analytics</div>', unsafe_allow_html=True)

try:
    ai_system, data = load_data()
    
    # Sidebar with Amazon theme
    st.sidebar.markdown("## 🎛️ Amazon AI Control Center")
    st.sidebar.markdown("---")
    
    module = st.sidebar.selectbox("🔧 Select AI Module", [
        "📊 Seller Dashboard",
        "🧠 AI Optimization Hub", 
        "📸 Product Image Analyzer",
        "💰 Dynamic Pricing Engine",
        "🔍 Amazon SEO Optimizer",
        "📈 Performance Analytics"
    ])
    
    if module == "📊 Seller Dashboard":
        st.markdown("## 📊 Amazon Seller Performance Dashboard")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_listings = data['listing_id'].nunique()
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-title">🛍️ Active Listings</div>
                <div class="metric-value">{total_listings}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            avg_conversion = data['conversion_rate'].mean()
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-title">📈 Conversion Rate</div>
                <div class="metric-value">{avg_conversion:.2%}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            total_revenue = data['revenue'].sum()
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-title">💵 Total Revenue</div>
                <div class="metric-value">${total_revenue:,.0f}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            avg_score = data['overall_score'].mean()
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-title">⭐ Optimization Score</div>
                <div class="metric-value">{avg_score:.1f}/100</div>
            </div>
            ''', unsafe_allow_html=True)
        
        # Charts with Amazon colors
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("🎯 Performance by Category")
            category_perf = data.groupby('category')['conversion_rate'].mean().reset_index()
            fig = px.bar(category_perf, x='category', y='conversion_rate', 
                        color='conversion_rate', 
                        color_continuous_scale=['#232F3E', '#FF9900', '#FFB84D'],
                        title="Category Performance Analysis")
            fig.update_layout(
                plot_bgcolor='white', 
                paper_bgcolor='white',
                font=dict(color='#232F3E', size=12),
                title_font=dict(color='#232F3E', size=16, family="Arial Black"),
                xaxis=dict(title_font=dict(color='#232F3E', size=14)),
                yaxis=dict(title_font=dict(color='#232F3E', size=14))
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("📈 Revenue Trend Analysis")
            daily_revenue = data.groupby('date')['revenue'].sum().reset_index()
            fig = px.line(daily_revenue, x='date', y='revenue', 
                         color_discrete_sequence=['#FF9900'],
                         title="30-Day Revenue Performance")
            fig.update_layout(
                plot_bgcolor='white', 
                paper_bgcolor='white',
                font=dict(color='#232F3E', size=12),
                title_font=dict(color='#232F3E', size=16, family="Arial Black"),
                xaxis=dict(title_font=dict(color='#232F3E', size=14)),
                yaxis=dict(title_font=dict(color='#232F3E', size=14))
            )
            fig.update_traces(line=dict(width=3))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif module == "🧠 AI Optimization Hub":
        st.markdown("## 🧠 Amazon AI Optimization Hub")
        
        if st.button("🚀 Train Advanced AI Models"):
            with st.spinner("🔄 Training machine learning models for Amazon optimization..."):
                results, best_model = ai_system.train_optimization_model(data)
                
                st.markdown('<div class="success-message">✅ AI models trained successfully! Ready for optimization predictions.</div>', unsafe_allow_html=True)
                
                st.subheader("🎯 Model Performance Metrics")
                results_df = pd.DataFrame(results).T
                st.dataframe(results_df, use_container_width=True)
                
                # Feature importance if available
                if hasattr(best_model, 'feature_importances_'):
                    features = ['Title Score', 'Image Score', 'SEO Score', 'Price Score', 'Price', 'Category']
                    importance_df = pd.DataFrame({
                        'Feature': features,
                        'Importance': best_model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    fig = px.bar(importance_df, x='Importance', y='Feature', 
                                orientation='h', title="🔍 Feature Importance Analysis",
                                color='Importance', color_continuous_scale=['#232F3E', '#FF9900'])
                    fig.update_layout(
                        plot_bgcolor='white', 
                        paper_bgcolor='white',
                        font=dict(color='#232F3E', size=12),
                        title_font=dict(color='#232F3E', size=16, family="Arial Black"),
                        xaxis=dict(title_font=dict(color='#232F3E', size=14)),
                        yaxis=dict(title_font=dict(color='#232F3E', size=14))
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
    
    elif module == "📸 Product Image Analyzer":
        st.markdown("## 📸 Amazon Product Image AI Analyzer")
        
        st.subheader("📊 Image Optimization Performance")
        image_analysis = data.groupby('category').agg({
            'image_score': 'mean',
            'conversion_rate': 'mean',
            'clicks': 'mean'
        }).round(3)
        
        st.dataframe(image_analysis, use_container_width=True)
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig = px.scatter(data, x='image_score', y='conversion_rate', 
                        color='category', size='clicks',
                        title="📸 Image Quality vs Conversion Rate Analysis",
                        color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(
            plot_bgcolor='white', 
            paper_bgcolor='white',
            font=dict(color='#232F3E', size=12),
            title_font=dict(color='#232F3E', size=16, family="Arial Black"),
            xaxis=dict(title_font=dict(color='#232F3E', size=14)),
            yaxis=dict(title_font=dict(color='#232F3E', size=14))
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif module == "💰 Dynamic Pricing Engine":
        st.markdown("## 💰 Amazon Dynamic Pricing AI Engine")
        
        st.subheader("🎯 Smart Pricing Recommendations")
        
        latest_data = data.groupby('listing_id').last().reset_index()
        recommendations = []
        
        for _, row in latest_data.head(12).iterrows():
            if row['price_score'] < 55:
                recommendation = "🔻 Reduce price by 12-18% to boost sales"
                action = "📉 Decrease"
                priority = "🔴 High"
            elif row['price_score'] > 88:
                recommendation = "🔺 Increase price by 8-12% for higher margins"
                action = "📈 Increase"
                priority = "🟢 Medium"
            else:
                recommendation = "✅ Current pricing strategy is optimal"
                action = "⚖️ Maintain"
                priority = "🟡 Low"
            
            recommendations.append({
                'ASIN': row['listing_id'],
                'Category': row['category'],
                'Current Price': f"${row['price']:.2f}",
                'Price Score': f"{row['price_score']:.1f}/100",
                'Action': action,
                'Priority': priority,
                'AI Recommendation': recommendation
            })
        
        rec_df = pd.DataFrame(recommendations)
        st.dataframe(rec_df, use_container_width=True)
    
    elif module == "🔍 Amazon SEO Optimizer":
        st.markdown("## 🔍 Amazon SEO & Keyword Optimization Engine")
        
        st.subheader("📈 SEO Performance Analytics")
        
        seo_analysis = data.groupby('category').agg({
            'seo_score': 'mean',
            'impressions': 'mean',
            'ctr': 'mean',
            'bsr_rank': 'mean'
        }).round(2)
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig = px.bar(seo_analysis.reset_index(), x='category', y='seo_score',
                    title="🔍 SEO Performance by Category", 
                    color='seo_score', 
                    color_continuous_scale=['#232F3E', '#FF9900', '#FFB84D'])
        fig.update_layout(
            plot_bgcolor='white', 
            paper_bgcolor='white',
            font=dict(color='#232F3E', size=12),
            title_font=dict(color='#232F3E', size=16, family="Arial Black"),
            xaxis=dict(title_font=dict(color='#232F3E', size=14)),
            yaxis=dict(title_font=dict(color='#232F3E', size=14))
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.subheader("📊 Detailed SEO Metrics")
        st.dataframe(seo_analysis, use_container_width=True)
    
    elif module == "📈 Performance Analytics":
        st.markdown("## 📈 Advanced Performance Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: #232F3E; text-align: center; font-weight: bold;">🎯 Click-Through Rate Analysis</h3>', unsafe_allow_html=True)
            ctr_data = data.groupby('category')['ctr'].mean().reset_index()
            
            # Create pie chart with custom styling for white text
            fig = go.Figure(data=[go.Pie(
                labels=ctr_data['category'], 
                values=ctr_data['ctr'],
                textfont=dict(size=16, color='white'),
                textposition='inside',
                textinfo='percent+label',
                marker=dict(
                    colors=['#FF9900', '#FFB84D', '#232F3E', '#37475A', '#FF6B35', '#FFA500'],
                    line=dict(color='white', width=2)
                )
            )])
            
            fig.update_layout(
                title=dict(
                    text='<b style="color:white;">CTR Distribution by Category</b>',
                    font=dict(color='white', size=18),
                    x=0.5
                ),
                plot_bgcolor='rgba(0,0,0,0)', 
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=True,
                legend=dict(
                    font=dict(color='white', size=14),
                    bgcolor='rgba(0,0,0,0)',
                    bordercolor='rgba(0,0,0,0)',
                    borderwidth=0
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: #232F3E; text-align: center; font-weight: bold;">💎 Best Sellers Rank Trends</h3>', unsafe_allow_html=True)
            bsr_data = data.groupby('date')['bsr_rank'].mean().reset_index()
            fig = px.line(bsr_data, x='date', y='bsr_rank',
                         title="Average BSR Over Time")
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', 
                paper_bgcolor='rgba(0,0,0,0)',
                title=dict(
                    text='<b style="color:white;">Average BSR Over Time</b>',
                    font=dict(color='white', size=18),
                    x=0.5
                ),
                xaxis=dict(
                    title='<b style="color:white;">Date</b>',
                    title_font=dict(color='white', size=16),
                    tickfont=dict(color='white', size=12),
                    gridcolor='rgba(255,255,255,0.3)'
                ),
                yaxis=dict(
                    title='<b style="color:white;">BSR Rank</b>',
                    title_font=dict(color='white', size=16),
                    tickfont=dict(color='white', size=12),
                    gridcolor='rgba(255,255,255,0.3)'
                )
            )
            fig.update_traces(
                line=dict(color='#FF9900', width=4),
                marker=dict(color='#FF9900', size=6)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

except Exception as e:
    st.error(f"Application Error: {str(e)}")
    st.info("Please ensure all required packages are installed and try refreshing the page.")

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; background: rgba(255, 153, 0, 0.1); border-radius: 10px; margin-top: 30px;">
    <h3 style="color: #FF9900;">🚀 Built for Amazon Sellers by AI Engineers</h3>
    <p style="color: #FFFFFF; opacity: 0.8;">Advanced machine learning algorithms optimizing your Amazon business performance</p>
</div>
""", unsafe_allow_html=True)