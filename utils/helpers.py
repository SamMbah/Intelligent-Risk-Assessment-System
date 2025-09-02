import json
import streamlit as st
from typing import Dict, Any, List
import pandas as pd
from datetime import datetime

def load_sample_data() -> Dict[str, Any]:
    """Load sample risk assessment data."""
    return {
        "risk_scenarios": [
            {
                "title": "Cybersecurity Data Breach",
                "description": "Unauthorized access to customer data through system vulnerability",
                "impact": "High financial loss, regulatory penalties, reputation damage",
                "likelihood": "Medium",
                "category": "cybersecurity",
                "mitigation": "Implement multi-factor authentication, regular security audits, employee training"
            },
            {
                "title": "Regulatory Compliance Violation",
                "description": "Failure to comply with new GDPR requirements for data processing",
                "impact": "Significant fines, legal action, operational restrictions",
                "likelihood": "Low",
                "category": "regulatory",
                "mitigation": "Regular compliance reviews, legal consultation, staff training"
            },
            {
                "title": "Supply Chain Disruption",
                "description": "Major supplier bankruptcy affecting critical business operations",
                "impact": "Production delays, increased costs, customer dissatisfaction",
                "likelihood": "Medium",
                "category": "operational",
                "mitigation": "Diversify supplier base, establish backup suppliers, inventory buffers"
            },
            {
                "title": "Market Volatility Impact",
                "description": "Economic downturn affecting customer demand and revenue",
                "impact": "Revenue decline, cash flow issues, potential layoffs",
                "likelihood": "High",
                "category": "financial",
                "mitigation": "Diversify revenue streams, maintain cash reserves, flexible cost structure"
            },
            {
                "title": "Key Personnel Loss",
                "description": "Departure of critical executives or specialized staff",
                "impact": "Knowledge loss, project delays, team disruption",
                "likelihood": "Medium",
                "category": "operational",
                "mitigation": "Succession planning, knowledge documentation, competitive retention packages"
            },
            {
                "title": "Technology Infrastructure Failure",
                "description": "Critical system outage affecting business operations",
                "impact": "Business interruption, data loss, customer impact",
                "likelihood": "Low",
                "category": "operational",
                "mitigation": "Redundant systems, regular backups, disaster recovery plan"
            },
            {
                "title": "Reputation Crisis",
                "description": "Negative media coverage affecting brand perception",
                "impact": "Customer loss, revenue decline, stakeholder concerns",
                "likelihood": "Low",
                "category": "reputational",
                "mitigation": "Crisis communication plan, social media monitoring, stakeholder engagement"
            },
            {
                "title": "Foreign Exchange Risk",
                "description": "Currency fluctuations affecting international operations",
                "impact": "Revenue variability, cost increases, margin compression",
                "likelihood": "High",
                "category": "financial",
                "mitigation": "Currency hedging, natural hedging, contract adjustments"
            }
        ],
        "regulatory_requirements": [
            {
                "name": "GDPR Data Protection",
                "description": "General Data Protection Regulation requirements for EU data processing",
                "jurisdiction": "European Union",
                "compliance_level": "Mandatory"
            },
            {
                "name": "SOX Financial Reporting",
                "description": "Sarbanes-Oxley Act requirements for financial reporting controls",
                "jurisdiction": "United States",
                "compliance_level": "Mandatory for public companies"
            },
            {
                "name": "ISO 27001 Security",
                "description": "Information security management system requirements",
                "jurisdiction": "International",
                "compliance_level": "Optional but recommended"
            },
            {
                "name": "Basel III Capital Requirements",
                "description": "International regulatory framework for bank capital adequacy",
                "jurisdiction": "International",
                "compliance_level": "Mandatory for banks"
            }
        ]
    }

def format_risk_score(score: float) -> str:
    """Format risk score with appropriate color and description."""
    if score < 0.3:
        return f"{score:.2f} (Low Risk)"
    elif score < 0.6:
        return f"{score:.2f} (Medium Risk)"
    elif score < 0.8:
        return f"{score:.2f} (High Risk)"
    else:
        return f"{score:.2f} (Critical Risk)"

def export_risk_data(data: Dict[str, Any], filename: str = None) -> str:
    """Export risk data to JSON format."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"risk_assessment_{timestamp}.json"
    
    try:
        json_str = json.dumps(data, indent=2, default=str)
        return json_str
    except Exception as e:
        st.error(f"Error exporting data: {str(e)}")
        return ""

def import_risk_data(json_str: str) -> Dict[str, Any]:
    """Import risk data from JSON format."""
    try:
        return json.loads(json_str)
    except Exception as e:
        st.error(f"Error importing data: {str(e)}")
        return {}

def calculate_risk_trend(scores: List[float], window: int = 5) -> List[float]:
    """Calculate moving average for risk trend analysis."""
    if len(scores) < window:
        return scores
    
    trends = []
    for i in range(len(scores) - window + 1):
        window_scores = scores[i:i + window]
        trend = sum(window_scores) / window
        trends.append(trend)
    
    return trends

def validate_risk_data(data: Dict[str, Any]) -> List[str]:
    """Validate risk data structure and return list of issues."""
    issues = []
    
    # Check required fields
    required_fields = ['risk_scenarios', 'regulatory_requirements']
    for field in required_fields:
        if field not in data:
            issues.append(f"Missing required field: {field}")
    
    # Validate risk scenarios
    if 'risk_scenarios' in data:
        for i, scenario in enumerate(data['risk_scenarios']):
            required_scenario_fields = ['title', 'description', 'impact', 'likelihood', 'category']
            for field in required_scenario_fields:
                if field not in scenario:
                    issues.append(f"Risk scenario {i}: Missing field '{field}'")
    
    # Validate regulatory requirements
    if 'regulatory_requirements' in data:
        for i, req in enumerate(data['regulatory_requirements']):
            required_req_fields = ['name', 'description', 'jurisdiction', 'compliance_level']
            for field in required_req_fields:
                if field not in req:
                    issues.append(f"Regulatory requirement {i}: Missing field '{field}'")
    
    return issues

def create_sample_csv_data() -> pd.DataFrame:
    """Create sample CSV data for testing."""
    sample_data = {
        'Risk_ID': ['R001', 'R002', 'R003', 'R004', 'R005'],
        'Risk_Title': [
            'Server Hardware Failure',
            'Customer Data Breach',
            'Regulatory Compliance Gap',
            'Key Supplier Bankruptcy',
            'Currency Exchange Volatility'
        ],
        'Category': ['Operational', 'Cybersecurity', 'Regulatory', 'Operational', 'Financial'],
        'Likelihood': ['Medium', 'Low', 'High', 'Low', 'High'],
        'Impact': ['High', 'Critical', 'Medium', 'High', 'Medium'],
        'Risk_Score': [6.5, 8.2, 5.8, 7.1, 6.0],
        'Owner': ['IT Manager', 'CISO', 'Compliance Officer', 'Procurement Manager', 'CFO'],
        'Status': ['Active', 'Mitigated', 'Under Review', 'Active', 'Monitored']
    }
    
    return pd.DataFrame(sample_data)

def generate_risk_heatmap_data(entities: List[str], categories: List[str]) -> pd.DataFrame:
    """Generate sample data for risk heatmap visualization."""
    import numpy as np
    
    # Generate random risk scores for demonstration
    np.random.seed(42)  # For reproducible results
    data = []
    
    for entity in entities:
        for category in categories:
            score = np.random.uniform(0, 10)
            data.append({
                'Entity': entity,
                'Category': category,
                'Risk_Score': score
            })
    
    return pd.DataFrame(data)

def get_risk_color_scale():
    """Get color scale for risk visualization."""
    return {
        'low': '#2ECC71',      # Green
        'medium': '#F39C12',   # Orange
        'high': '#E74C3C',     # Red
        'critical': '#8E44AD'  # Purple
    }

def parse_uploaded_file(file, file_type: str) -> List[Dict[str, Any]]:
    """Parse uploaded file and return structured data."""
    try:
        if file_type == 'csv':
            df = pd.read_csv(file)
            return df.to_dict('records')
        elif file_type == 'json':
            data = json.load(file)
            return data if isinstance(data, list) else [data]
        else:
            return []
    except Exception as e:
        st.error(f"Error parsing file: {str(e)}")
        return []

@st.cache_data
def get_cached_sample_data():
    """Get cached sample data for performance."""
    return load_sample_data()

def format_percentage(value: float) -> str:
    """Format a decimal value as percentage."""
    return f"{value * 100:.1f}%"

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    return numerator / denominator if denominator != 0 else default

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to specified length with ellipsis."""
    return text[:max_length] + "..." if len(text) > max_length else text
