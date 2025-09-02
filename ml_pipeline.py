import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class MLPipeline:
    """MLOps pipeline for model training and monitoring"""
    
    def __init__(self):
        self.model_registry = {}
        self.training_history = []
        self.performance_metrics = {}
        self.alerts = []
    
    def train_models(self, df):
        """Train all ML models in the pipeline"""
        
        training_results = {
            'timestamp': datetime.now(),
            'models_trained': 0,
            'average_accuracy': 0,
            'training_time': 0
        }
        
        # Simulate model training
        models = ['ARIMA', 'Random Forest', 'Gradient Boosting', 'Neural Network']
        accuracies = []
        
        for model in models:
            # Simulate training process
            accuracy = np.random.uniform(85, 97)
            accuracies.append(accuracy)
            
            self.model_registry[model] = {
                'accuracy': accuracy,
                'last_trained': datetime.now(),
                'status': 'active',
                'version': '1.0'
            }
            
            training_results['models_trained'] += 1
        
        training_results['average_accuracy'] = np.mean(accuracies)
        training_results['training_time'] = np.random.uniform(120, 300)  # seconds
        
        self.training_history.append(training_results)
        
        return training_results
    
    def monitor_model_drift(self, df):
        """Monitor for model drift and performance degradation"""
        
        drift_results = []
        
        for model_name, model_info in self.model_registry.items():
            # Simulate drift detection
            drift_score = np.random.uniform(0, 1)
            
            if drift_score > 0.7:
                severity = 'high'
                action = 'retrain_immediately'
            elif drift_score > 0.4:
                severity = 'medium'
                action = 'schedule_retrain'
            else:
                severity = 'low'
                action = 'monitor'
            
            drift_results.append({
                'model': model_name,
                'drift_score': drift_score,
                'severity': severity,
                'recommended_action': action,
                'last_check': datetime.now()
            })
        
        return drift_results
    
    def validate_model_performance(self, df):
        """Validate model performance against holdout data"""
        
        validation_results = {}
        
        for model_name in self.model_registry.keys():
            # Simulate validation metrics
            mae = np.random.uniform(2, 8)
            rmse = np.random.uniform(3, 12)
            mape = np.random.uniform(5, 15)
            r2 = np.random.uniform(0.7, 0.95)
            
            validation_results[model_name] = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'r2_score': r2,
                'accuracy': max(0, 100 - mape)
            }
        
        return validation_results
    
    def generate_model_insights(self, df):
        """Generate insights about model performance and recommendations"""
        
        insights = []
        
        # Performance insights
        if self.model_registry:
            best_model = max(self.model_registry.items(), key=lambda x: x[1]['accuracy'])
            insights.append({
                'type': 'performance',
                'message': f"Best performing model: {best_model[0]} with {best_model[1]['accuracy']:.1f}% accuracy",
                'priority': 'info'
            })
        
        # Training frequency insights
        if len(self.training_history) > 1:
            last_training = self.training_history[-1]['timestamp']
            days_since_training = (datetime.now() - last_training).days
            
            if days_since_training > 7:
                insights.append({
                    'type': 'maintenance',
                    'message': f"Models haven't been retrained in {days_since_training} days",
                    'priority': 'warning'
                })
        
        # Data quality insights
        data_quality_score = np.random.uniform(0.8, 1.0)
        if data_quality_score < 0.9:
            insights.append({
                'type': 'data_quality',
                'message': f"Data quality score: {data_quality_score:.2f} - consider data cleaning",
                'priority': 'warning'
            })
        
        return insights
    
    def get_pipeline_status(self):
        """Get overall pipeline health status"""
        
        if not self.model_registry:
            return {
                'status': 'not_initialized',
                'health_score': 0,
                'active_models': 0,
                'last_training': None
            }
        
        # Calculate health score
        active_models = sum(1 for model in self.model_registry.values() if model['status'] == 'active')
        avg_accuracy = np.mean([model['accuracy'] for model in self.model_registry.values()])
        
        health_score = min(100, (active_models / 4) * 50 + (avg_accuracy / 100) * 50)
        
        last_training = None
        if self.training_history:
            last_training = self.training_history[-1]['timestamp']
        
        status = 'healthy' if health_score > 80 else 'warning' if health_score > 60 else 'critical'
        
        return {
            'status': status,
            'health_score': health_score,
            'active_models': active_models,
            'last_training': last_training,
            'total_models': len(self.model_registry)
        }
    
    def schedule_retraining(self, model_name=None):
        """Schedule model retraining"""
        
        if model_name:
            models_to_retrain = [model_name]
        else:
            models_to_retrain = list(self.model_registry.keys())
        
        scheduled_jobs = []
        
        for model in models_to_retrain:
            job = {
                'model': model,
                'scheduled_time': datetime.now() + timedelta(hours=1),
                'priority': 'normal',
                'estimated_duration': np.random.randint(30, 180)  # minutes
            }
            scheduled_jobs.append(job)
        
        return scheduled_jobs
    
    def export_model_metrics(self):
        """Export model metrics for reporting"""
        
        metrics_export = {
            'export_timestamp': datetime.now(),
            'model_registry': self.model_registry,
            'training_history': self.training_history,
            'pipeline_status': self.get_pipeline_status()
        }
        
        return metrics_export

# Global instance
ml_pipeline = MLPipeline()