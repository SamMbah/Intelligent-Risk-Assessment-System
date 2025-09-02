import networkx as nx
import spacy
from typing import List, Dict, Any, Tuple, Optional
import re
import streamlit as st
from collections import defaultdict, Counter
import json

class KnowledgeGraphBuilder:
    """Builds and manages knowledge graphs from processed documents using NetworkX."""
    
    def __init__(self):
        self.nlp = self._load_spacy_model()
        
        self.risk_keywords = {
            'cybersecurity': ['cyber', 'security', 'breach', 'hacking', 'malware', 'phishing', 'ransomware'],
            'financial': ['financial', 'credit', 'market', 'liquidity', 'currency', 'investment'],
            'operational': ['operational', 'process', 'system', 'failure', 'disruption', 'downtime'],
            'regulatory': ['regulatory', 'compliance', 'legal', 'legislation', 'law', 'regulation'],
            'reputational': ['reputation', 'brand', 'image', 'public', 'media', 'scandal'],
            'strategic': ['strategic', 'business', 'market', 'competition', 'innovation']
        }
        
    def _load_spacy_model(self):
        """Load spaCy model for NLP processing."""
        try:
            # Try to load the English model
            return spacy.load("en_core_web_sm")
        except OSError:
            st.error(
                "spaCy English model not found. "
                "Please install it using: python -m spacy download en_core_web_sm"
            )
            # Return a mock object for basic functionality
            return None
    
    def extract_entities_and_relationships(self, chunks: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
        """Extract entities and relationships from document chunks."""
        entities = []
        relationships = []
        entity_counter = Counter()
        
        for chunk in chunks:
            text = chunk['content']
            
            # Extract named entities using spaCy
            if self.nlp:
                doc = self.nlp(text)
                
                chunk_entities = []
                for ent in doc.ents:
                    entity_type = self._map_entity_type(ent.label_)
                    entity_text = ent.text.strip()
                    
                    if len(entity_text) > 2 and entity_type:  # Filter out short entities
                        entity = {
                            'text': entity_text,
                            'type': entity_type,
                            'source': chunk['metadata']['source'],
                            'confidence': 1.0
                        }
                        entities.append(entity)
                        chunk_entities.append(entity_text)
                        entity_counter[entity_text] += 1
                
                # Extract relationships between entities in the same chunk
                for i in range(len(chunk_entities)):
                    for j in range(i + 1, len(chunk_entities)):
                        relationship = {
                            'source': chunk_entities[i],
                            'target': chunk_entities[j],
                            'type': 'co_occurrence',
                            'weight': 1.0,
                            'context': text[:200] + "..." if len(text) > 200 else text
                        }
                        relationships.append(relationship)
            
            # Extract risk-specific entities using keyword matching
            risk_entities = self._extract_risk_entities(text, chunk['metadata']['source'])
            entities.extend(risk_entities)
        
        # Filter entities by frequency (keep entities that appear multiple times)
        frequent_entities = [
            entity for entity in entities 
            if entity_counter[entity['text']] >= 1  # Adjust threshold as needed
        ]
        
        return frequent_entities, relationships
    
    def _extract_risk_entities(self, text: str, source: str) -> List[Dict]:
        """Extract risk-specific entities using keyword matching."""
        entities = []
        text_lower = text.lower()
        
        for category, keywords in self.risk_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Find the actual occurrence in the text
                    pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
                    matches = pattern.findall(text)
                    
                    for match in matches:
                        entity = {
                            'text': match,
                            'type': 'RISK_FACTOR',
                            'category': category,
                            'source': source,
                            'confidence': 0.8
                        }
                        entities.append(entity)
        
        return entities
    
    def _map_entity_type(self, spacy_label: str) -> str:
        """Map spaCy entity labels to our risk assessment entity types."""
        mapping = {
            'ORG': 'ORGANIZATION',
            'PERSON': 'PERSON',
            'GPE': 'LOCATION',
            'LOC': 'LOCATION',
            'MONEY': 'FINANCIAL',
            'LAW': 'REGULATORY',
            'EVENT': 'RISK_EVENT',
            'PRODUCT': 'ASSET',
            'WORK_OF_ART': 'ASSET'
        }
        return mapping.get(spacy_label, "")
    
    def build_graph(self, entities: List[Dict], relationships: List[Dict]) -> nx.Graph:
        """Build NetworkX graph from entities and relationships."""
        G = nx.Graph()
        
        # Add nodes (entities)
        entity_dict = {}
        for entity in entities:
            entity_id = entity['text']
            if entity_id not in entity_dict:
                entity_dict[entity_id] = entity
                G.add_node(
                    entity_id,
                    type=entity['type'],
                    category=entity.get('category', ''),
                    confidence=entity['confidence'],
                    sources=[entity['source']]
                )
            else:
                # Merge information for duplicate entities
                if entity['source'] not in G.nodes[entity_id]['sources']:
                    G.nodes[entity_id]['sources'].append(entity['source'])
        
        # Add edges (relationships)
        relationship_weights = defaultdict(float)
        
        for rel in relationships:
            source = rel['source']
            target = rel['target']
            
            if source in G.nodes and target in G.nodes and source != target:
                edge_key = (source, target) if source < target else (target, source)
                relationship_weights[edge_key] += rel['weight']
        
        # Add edges to graph
        for (source, target), weight in relationship_weights.items():
            G.add_edge(source, target, weight=weight, type='co_occurrence')
        
        return G
    
    def get_advanced_risk_analysis(self, entity_name: str, graph: nx.Graph) -> Dict[str, Any]:
        """Get enhanced risk analysis using NetworkX graph algorithms."""
        try:
            analysis = {}
            
            # Find connected components (risk clusters)
            if entity_name in graph:
                # Get subgraph of connected entities
                connected_entities = nx.node_connected_component(graph, entity_name)
                subgraph = graph.subgraph(connected_entities)
                
                analysis['connected_entities'] = list(connected_entities)
                analysis['cluster_size'] = len(connected_entities)
                
                # Calculate centrality measures
                analysis['centrality_metrics'] = {
                    'degree': nx.degree_centrality(subgraph).get(entity_name, 0),
                    'betweenness': nx.betweenness_centrality(subgraph).get(entity_name, 0),
                    'closeness': nx.closeness_centrality(subgraph).get(entity_name, 0)
                }
                
                # Find shortest paths to other entities
                analysis['risk_pathways'] = []
                for target in list(connected_entities)[:5]:  # Limit to 5 for performance
                    if target != entity_name:
                        try:
                            path = nx.shortest_path(subgraph, entity_name, target)
                            analysis['risk_pathways'].append({
                                'target': target,
                                'path': path,
                                'distance': len(path) - 1
                            })
                        except nx.NetworkXNoPath:
                            continue
            else:
                analysis['error'] = f"Entity '{entity_name}' not found in knowledge graph"
            
            return analysis
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def find_risk_clusters(self, graph: nx.Graph, min_cluster_size: int = 3) -> List[Dict[str, Any]]:
        """Find clusters of related entities using NetworkX community detection."""
        try:
            # Find connected components
            components = list(nx.connected_components(graph))
            clusters = []
            
            for i, component in enumerate(components):
                if len(component) >= min_cluster_size:
                    subgraph = graph.subgraph(component)
                    
                    # Calculate cluster metrics
                    cluster_info = {
                        'cluster_id': f"cluster_{i}",
                        'entities': list(component),
                        'size': len(component),
                        'density': nx.density(subgraph),
                        'avg_clustering': nx.average_clustering(subgraph)
                    }
                    
                    # Find most central entity in cluster
                    centrality = nx.degree_centrality(subgraph)
                    most_central = max(centrality.items(), key=lambda x: x[1])
                    cluster_info['central_entity'] = most_central[0]
                    cluster_info['central_score'] = most_central[1]
                    
                    clusters.append(cluster_info)
            
            return sorted(clusters, key=lambda x: x['size'], reverse=True)
            
        except Exception as e:
            st.error(f"Cluster analysis failed: {str(e)}")
            return []
    
    def get_graph_visualization_data(self, graph: nx.Graph, node_limit: int = 50, entity_types: Optional[List[str]] = None) -> Dict:
        """Prepare graph data for visualization."""
        if entity_types:
            # Filter nodes by entity type
            filtered_nodes = [
                node for node in graph.nodes()
                if graph.nodes[node].get('type', '') in entity_types
            ]
        else:
            filtered_nodes = list(graph.nodes())
        
        # Limit number of nodes
        if len(filtered_nodes) > node_limit:
            # Select nodes with highest degree (most connected)
            node_degrees = [(node, graph.degree(node)) for node in filtered_nodes]
            node_degrees.sort(key=lambda x: int(x[1]), reverse=True)
            filtered_nodes = [node for node, _ in node_degrees[:node_limit]]
        
        # Create subgraph
        subgraph = graph.subgraph(filtered_nodes)
        
        # Prepare node data
        nodes = []
        for node in subgraph.nodes():
            node_data = {
                'id': node,
                'label': node[:30] + "..." if len(node) > 30 else node,
                'type': subgraph.nodes[node].get('type', 'UNKNOWN'),
                'size': min(10 + int(subgraph.degree(node)) * 2, 30)
            }
            nodes.append(node_data)
        
        # Prepare edge data
        edges = []
        for source, target in subgraph.edges():
            edge_data = {
                'source': source,
                'target': target,
                'weight': subgraph.edges[source, target].get('weight', 1.0)
            }
            edges.append(edge_data)
        
        return {'nodes': nodes, 'edges': edges}
    
    def get_node_color(self, node_type: str) -> str:
        """Get color for node based on its type."""
        color_map = {
            'ORGANIZATION': '#FF6B6B',
            'PERSON': '#4ECDC4',
            'LOCATION': '#45B7D1',
            'RISK_FACTOR': '#FFA07A',
            'REGULATORY': '#98D8C8',
            'FINANCIAL': '#F7DC6F',
            'RISK_EVENT': '#BB8FCE',
            'ASSET': '#85C1E9',
            'UNKNOWN': '#BDC3C7'
        }
        return color_map.get(node_type, '#BDC3C7')
    
    def get_graph_statistics(self, graph: nx.Graph) -> Dict[str, Any]:
        """Get basic statistics about the knowledge graph."""
        if not graph:
            return {
                'nodes': 0,
                'edges': 0,
                'components': 0,
                'density': 0.0,
                'avg_degree': 0.0
            }
        
        return {
            'nodes': graph.number_of_nodes(),
            'edges': graph.number_of_edges(),
            'components': nx.number_connected_components(graph),
            'density': nx.density(graph),
            'avg_degree': sum(dict(graph.degree()).values()) / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0
        }
    
    def query_graph(self, graph: nx.Graph, query: str) -> List[str]:
        """Query the knowledge graph for relevant entities and relationships."""
        if not graph:
            return ["No knowledge graph available"]
        
        query_lower = query.lower()
        results = []
        
        # Find nodes that match the query
        matching_nodes = []
        for node in graph.nodes():
            if any(word in node.lower() for word in query_lower.split()):
                matching_nodes.append(node)
        
        if not matching_nodes:
            return ["No matching entities found in the knowledge graph"]
        
        # For each matching node, find its neighbors and relationships
        for node in matching_nodes[:5]:  # Limit to top 5 matches
            neighbors = list(graph.neighbors(node))
            if neighbors:
                neighbor_list = ", ".join(neighbors[:3])  # Show top 3 neighbors
                results.append(f"{node} is connected to: {neighbor_list}")
            else:
                results.append(f"{node} (isolated entity)")
        
        return results
    
    def find_risk_paths(self, graph: nx.Graph, source_entity: str, target_entity: str) -> List[List[str]]:
        """Find paths between two entities that might represent risk propagation."""
        try:
            if source_entity in graph and target_entity in graph:
                # Find shortest paths (up to 3 hops)
                paths = list(nx.all_simple_paths(graph, source_entity, target_entity, cutoff=3))
                return paths[:5]  # Return top 5 paths
            return []
        except nx.NetworkXNoPath:
            return []
