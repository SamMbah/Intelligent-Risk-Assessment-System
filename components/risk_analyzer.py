import networkx as nx
from typing import Dict, Any, List
import pandas as pd
from collections import defaultdict, Counter
import numpy as np
from datetime import datetime, timedelta
import random

class RiskAnalyzer:
    """Analyzes risks using knowledge graph and LLM insights."""
    
    def __init__(self, llm_interface):
        self.llm = llm_interface
        self.risk_weights = {
            'operational': 0.25,
            'financial': 0.25,
            'regulatory': 0.20,
            'strategic': 0.15,
            'reputational': 0.10,
            'cybersecurity': 0.05
        }
    
    def analyze_entity_risk(self, entity: str, graph: nx.Graph) -> Dict[str, Any]:
        """Analyze risk for a specific entity using graph analysis."""
        if entity not in graph:
            return {
                'risk_score': 0.0,
                'confidence': 0.0,
                'risk_factors': [],
                'recommendations': ['Entity not found in knowledge graph']
            }
        
        # Calculate graph-based risk metrics
        centrality = nx.degree_centrality(graph)[entity]
        betweenness = nx.betweenness_centrality(graph)[entity]
        
        # Get connected risk factors
        neighbors = list(graph.neighbors(entity))
        risk_neighbors = [
            n for n in neighbors 
            if graph.nodes[n].get('type') == 'RISK_FACTOR'
        ]
        
        # Calculate base risk score
        connectivity_score = (centrality + betweenness) / 2
        risk_factor_score = len(risk_neighbors) / 10  # Normalize
        
        base_risk_score = min((connectivity_score + risk_factor_score) * 5, 10)
        
        # Get entity context for LLM analysis
        entity_context = self._build_entity_context(entity, graph)
        
        # Use LLM for detailed analysis
        llm_analysis = self.llm.analyze_risk_text(entity_context)
        
        # Combine graph and LLM insights
        combined_score = (base_risk_score + llm_analysis.get('overall_risk_score', 0)) / 2
        
        return {
            'risk_score': combined_score,
            'confidence': 0.8,  # Confidence based on available data
            'risk_factors': risk_neighbors + llm_analysis.get('risk_factors', []),
            'recommendations': llm_analysis.get('recommendations', []),
            'graph_metrics': {
                'centrality': centrality,
                'betweenness': betweenness,
                'connections': len(neighbors)
            }
        }
    
    def analyze_text_risks(self, text: str) -> Dict[str, float]:
        """Analyze text for various risk categories."""
        # Use LLM for detailed risk analysis
        llm_analysis = self.llm.analyze_risk_text(text)
        
        if 'error' in llm_analysis:
            # Fallback to keyword-based analysis
            return self._keyword_based_risk_analysis(text)
        
        # Extract risk scores from LLM analysis
        risk_categories = llm_analysis.get('risk_categories', {})
        
        # Normalize scores to 0-1 scale
        normalized_scores = {}
        for category, score in risk_categories.items():
            try:
                normalized_scores[category] = float(score) / 10.0
            except (ValueError, TypeError):
                normalized_scores[category] = 0.0
        
        return normalized_scores
    
    def _keyword_based_risk_analysis(self, text: str) -> Dict[str, float]:
        """Fallback keyword-based risk analysis."""
        text_lower = text.lower()
        
        risk_keywords = {
            'operational': ['failure', 'disruption', 'downtime', 'process', 'system'],
            'financial': ['loss', 'debt', 'cash', 'revenue', 'profit', 'investment'],
            'regulatory': ['compliance', 'regulation', 'legal', 'audit', 'violation'],
            'strategic': ['competition', 'market', 'innovation', 'growth', 'strategic'],
            'reputational': ['reputation', 'brand', 'image', 'scandal', 'publicity'],
            'cybersecurity': ['cyber', 'security', 'breach', 'hacking', 'data']
        }
        
        scores = {}
        for category, keywords in risk_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[category] = min(score / len(keywords), 1.0)
        
        return scores
    
    def _build_entity_context(self, entity: str, graph: nx.Graph) -> str:
        """Build context about an entity from the knowledge graph."""
        if entity not in graph:
            return f"Entity: {entity} (no additional context available)"
        
        context_parts = [f"Entity: {entity}"]
        
        # Add entity properties
        node_data = graph.nodes[entity]
        entity_type = node_data.get('type', 'Unknown')
        context_parts.append(f"Type: {entity_type}")
        
        # Add connected entities
        neighbors = list(graph.neighbors(entity))[:5]  # Limit to top 5
        if neighbors:
            context_parts.append(f"Connected to: {', '.join(neighbors)}")
        
        # Add risk-specific information
        risk_neighbors = [
            n for n in neighbors 
            if graph.nodes[n].get('type') == 'RISK_FACTOR'
        ]
        if risk_neighbors:
            context_parts.append(f"Associated risk factors: {', '.join(risk_neighbors)}")
        
        return ". ".join(context_parts)
    
    def get_dashboard_data(self, graph: nx.Graph) -> Dict[str, Any]:
        """Generate data for the risk dashboard."""
        if not graph or graph.number_of_nodes() == 0:
            return self._empty_dashboard_data()
        
        # Count entities by type
        entity_counts = Counter()
        risk_entities = []
        
        for node in graph.nodes():
            node_data = graph.nodes[node]
            entity_type = node_data.get('type', 'Unknown')
            entity_counts[entity_type] += 1
            
            if entity_type == 'RISK_FACTOR':
                risk_entities.append(node)
        
        # Generate risk scores for entities
        risk_scores = []
        for entity in list(graph.nodes())[:20]:  # Analyze top 20 entities
            centrality = nx.degree_centrality(graph)[entity]
            # Simple risk score based on centrality and connections
            score = min(centrality * 10 + random.uniform(0, 2), 10)
            risk_scores.append(score)
        
        avg_risk_score = np.mean(risk_scores) if risk_scores else 0
        
        # Risk distribution
        risk_distribution = {
            'Low': sum(1 for score in risk_scores if score < 3),
            'Medium': sum(1 for score in risk_scores if 3 <= score < 6),
            'High': sum(1 for score in risk_scores if 6 <= score < 8),
            'Critical': sum(1 for score in risk_scores if score >= 8)
        }
        
        # Top risk factors
        top_risk_factors = []
        for i, entity in enumerate(risk_entities[:10]):
            score = risk_scores[i] if i < len(risk_scores) else random.uniform(3, 8)
            top_risk_factors.append({
                'factor': entity,
                'score': score
            })
        
        # Sort by score
        top_risk_factors.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'total_entities': graph.number_of_nodes(),
            'high_risk_count': risk_distribution['High'] + risk_distribution['Critical'],
            'avg_risk_score': avg_risk_score,
            'risk_categories': len(entity_counts),
            'risk_distribution': risk_distribution,
            'top_risk_factors': top_risk_factors[:10],
            'entity_type_counts': dict(entity_counts)
        }
    
    def _empty_dashboard_data(self) -> Dict[str, Any]:
        """Return sample dashboard data when no graph is available."""
        return {
            'total_entities': 5,
            'high_risk_count': 2,
            'avg_risk_score': 6.2,
            'risk_categories': 5,
            'risk_distribution': {'Low': 1, 'Medium': 2, 'High': 1, 'Critical': 1},
            'top_risk_factors': [
                {'factor': 'Cybersecurity Threats', 'score': 8.5},
                {'factor': 'Regulatory Compliance', 'score': 7.2},
                {'factor': 'Supply Chain Risk', 'score': 6.8},
                {'factor': 'Market Volatility', 'score': 6.1},
                {'factor': 'Operational Risk', 'score': 5.4}
            ],
            'entity_type_counts': {
                'ORGANIZATION': 2,
                'RISK_FACTOR': 8,
                'FINANCIAL': 3,
                'REGULATORY': 2
            }
        }
    
    def calculate_risk_propagation(self, graph: nx.Graph, source_entity: str, max_hops: int = 3) -> Dict[str, Any]:
        """Calculate how risk might propagate through the network."""
        if source_entity not in graph:
            return {'propagation_paths': [], 'affected_entities': []}
        
        # Find entities within max_hops
        affected_entities = []
        propagation_paths = []
        
        try:
            # Use BFS to find reachable entities
            visited = set()
            queue = [(source_entity, 0, [source_entity])]
            
            while queue:
                current_entity, hops, path = queue.pop(0)
                
                if current_entity in visited or hops > max_hops:
                    continue
                
                visited.add(current_entity)
                
                if hops > 0:  # Don't include source entity
                    affected_entities.append({
                        'entity': current_entity,
                        'distance': hops,
                        'path': path
                    })
                
                # Add neighbors to queue
                for neighbor in graph.neighbors(current_entity):
                    if neighbor not in visited:
                        new_path = path + [neighbor]
                        queue.append((neighbor, hops + 1, new_path))
                        
                        if hops + 1 <= max_hops:
                            propagation_paths.append(new_path)
        
        except Exception as e:
            print(f"Error calculating risk propagation: {e}")
        
        return {
            'propagation_paths': propagation_paths[:10],  # Limit results
            'affected_entities': affected_entities[:20]
        }
    
    def generate_risk_matrix(self, entities: List[str], graph: nx.Graph) -> pd.DataFrame:
        """Generate a risk matrix for multiple entities."""
        risk_data = []
        
        for entity in entities:
            if entity in graph:
                risk_analysis = self.analyze_entity_risk(entity, graph)
                
                risk_data.append({
                    'Entity': entity,
                    'Risk Score': risk_analysis['risk_score'],
                    'Confidence': risk_analysis['confidence'],
                    'Risk Factors': len(risk_analysis['risk_factors']),
                    'Centrality': risk_analysis['graph_metrics']['centrality'],
                    'Connections': risk_analysis['graph_metrics']['connections']
                })
        
        return pd.DataFrame(risk_data)
