import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_agraph import agraph, Node, Edge, Config
import json
import os
from typing import List, Dict, Any

# Import custom components
from components.document_processor import DocumentProcessor
from components.knowledge_graph import KnowledgeGraphBuilder
from components.vector_store import VectorStore
from components.llm_interface import LLMInterface
from components.risk_analyzer import RiskAnalyzer
from utils.helpers import load_sample_data, format_risk_score
from dotenv import load_dotenv
from openai import OpenAI
import spacy

def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        # Fallback to a blank English pipeline (works without downloading a model)
        return spacy.blank("en")

nlp = load_spacy_model()

load_dotenv()  # load values from .env
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Page configuration
st.set_page_config(
    page_title="IRAS - Intelligent Risk Assessment System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'knowledge_graph' not in st.session_state:
    st.session_state.knowledge_graph = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'processed_documents' not in st.session_state:
    st.session_state.processed_documents = False

# Initialize components
@st.cache_resource
def initialize_components():
    doc_processor = DocumentProcessor()
    kg_builder = KnowledgeGraphBuilder()
    vector_store = VectorStore()
    llm_interface = LLMInterface()
    risk_analyzer = RiskAnalyzer(llm_interface)
    return doc_processor, kg_builder, vector_store, llm_interface, risk_analyzer

doc_processor, kg_builder, vector_store, llm_interface, risk_analyzer = initialize_components()

# Main header
st.title("🛡️ IRAS - Intelligent Risk Assessment System")
st.markdown("**Knowledge Graph-Powered SaaS Platform for Automated Business Risk Identification**")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    ["Document Upload", "Knowledge Graph", "Risk Analysis", "Query Interface", "Risk Dashboard"]
)

# Document Upload Page
if page == "Document Upload":
    st.header("📄 Document Ingestion")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload risk assessment documents",
        type=['pdf', 'txt', 'csv'],
        accept_multiple_files=True,
        help="Supported formats: PDF, TXT, CSV"
    )
    
    if uploaded_files:
        st.subheader("Uploaded Documents")
        for file in uploaded_files:
            st.write(f"📁 {file.name} ({file.size} bytes)")
        
        if st.button("Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                progress_bar = st.progress(0)
                
                # Process each document
                all_chunks = []
                for i, file in enumerate(uploaded_files):
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    # Process document based on type
                    if file.type == "application/pdf":
                        chunks = doc_processor.process_pdf(file)
                    elif file.type == "text/plain":
                        chunks = doc_processor.process_text(file)
                    elif file.type == "text/csv":
                        chunks = doc_processor.process_csv(file)
                    else:
                        st.warning(f"Unsupported file type: {file.type}")
                        continue
                    
                    all_chunks.extend(chunks)
                    st.session_state.documents.append({
                        'name': file.name,
                        'type': file.type,
                        'chunks': len(chunks)
                    })
                
                # Build knowledge graph
                st.info("Building knowledge graph...")
                entities, relationships = kg_builder.extract_entities_and_relationships(all_chunks)
                graph = kg_builder.build_graph(entities, relationships)
                st.session_state.knowledge_graph = graph
                
                # Create vector embeddings
                st.info("Creating vector embeddings...")
                vector_store.add_documents(all_chunks)
                st.session_state.vector_store = vector_store
                st.session_state.processed_documents = True
                
                progress_bar.progress(1.0)
                st.success(f"Successfully processed {len(uploaded_files)} documents with {len(all_chunks)} chunks!")
    
    # Load sample data option
    if st.button("Load Sample Risk Data"):
        with st.spinner("Loading sample data..."):
            sample_data = load_sample_data()
            chunks = doc_processor.process_sample_data(sample_data)
            
            # Build knowledge graph
            entities, relationships = kg_builder.extract_entities_and_relationships(chunks)
            graph = kg_builder.build_graph(entities, relationships)
            st.session_state.knowledge_graph = graph
            
            # Create vector embeddings
            vector_store.add_documents(chunks)
            st.session_state.vector_store = vector_store
            st.session_state.processed_documents = True
            
            st.success("Sample risk data loaded successfully!")

# Knowledge Graph Page
elif page == "Knowledge Graph":
    st.header("🕸️ Risk Knowledge Graph")
    
    if not st.session_state.processed_documents:
        st.warning("Please upload and process documents first.")
    else:
        # Graph visualization options
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Graph Controls")
            
            # Filter options
            node_limit = st.slider("Number of nodes to display", 10, 100, 50)
            
            # Layout options
            layout = st.selectbox(
                "Graph Layout",
                ["spring", "circular", "kamada_kawai", "random"]
            )
            
            # Node types to show
            show_entities = st.multiselect(
                "Entity Types to Show",
                ["ORGANIZATION", "PERSON", "LOCATION", "RISK_FACTOR", "REGULATORY", "FINANCIAL"],
                default=["RISK_FACTOR", "ORGANIZATION", "REGULATORY"]
            )
        
        with col2:
            if st.session_state.knowledge_graph:
                # Get graph data
                graph_data = kg_builder.get_graph_visualization_data(
                    st.session_state.knowledge_graph,
                    node_limit=node_limit,
                    entity_types=show_entities
                )
                
                # Create nodes and edges for agraph
                nodes = []
                edges = []
                
                for node_data in graph_data['nodes']:
                    color = kg_builder.get_node_color(node_data['type'])
                    nodes.append(Node(
                        id=node_data['id'],
                        label=node_data['label'],
                        size=node_data.get('size', 20),
                        color=color
                    ))
                
                for edge_data in graph_data['edges']:
                    edges.append(Edge(
                        source=edge_data['source'],
                        target=edge_data['target'],
                        label=edge_data.get('label', ''),
                        color="#666666"
                    ))
                
                # Graph configuration
                config = Config(
                    width=700,
                    height=500,
                    directed=True,
                    physics=True,
                    hierarchical=False
                )
                
                # Display graph
                if nodes and edges:
                    agraph(nodes=nodes, edges=edges, config=config)
                else:
                    st.info("No graph data available for the selected filters.")
                
                # Graph statistics
                st.subheader("Graph Statistics")
                stats = kg_builder.get_graph_statistics(st.session_state.knowledge_graph)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Nodes", stats['nodes'])
                with col2:
                    st.metric("Total Edges", stats['edges'])
                with col3:
                    st.metric("Connected Components", stats['components'])

# Risk Analysis Page
elif page == "Risk Analysis":
    st.header("📊 Risk Analysis")
    
    if not st.session_state.processed_documents:
        st.warning("Please upload and process documents first.")
    else:
        # Risk analysis options
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Entity-based Risk Assessment", "Document Risk Scoring", "Relationship Analysis"]
        )
        
        if analysis_type == "Entity-based Risk Assessment":
            st.subheader("Entity Risk Assessment")
            
            # Get entities from knowledge graph
            if st.session_state.knowledge_graph:
                entities = list(st.session_state.knowledge_graph.nodes())
                selected_entity = st.selectbox("Select entity to analyze", entities)
                
                if st.button("Analyze Risk") and selected_entity:
                    with st.spinner("Analyzing risk factors..."):
                        risk_assessment = risk_analyzer.analyze_entity_risk(
                            selected_entity,
                            st.session_state.knowledge_graph
                        )
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric(
                                "Risk Score",
                                f"{risk_assessment['risk_score']:.2f}",
                                delta=f"{risk_assessment['confidence']:.2%} confidence"
                            )
                            
                            st.subheader("Risk Factors")
                            for factor in risk_assessment['risk_factors']:
                                st.write(f"• {factor}")
                        
                        with col2:
                            st.subheader("Mitigation Recommendations")
                            for rec in risk_assessment['recommendations']:
                                st.write(f"• {rec}")
        
        elif analysis_type == "Document Risk Scoring":
            st.subheader("Document Risk Scoring")
            
            # Input text for analysis
            text_input = st.text_area(
                "Enter text for risk analysis",
                placeholder="Enter business description, process, or scenario to analyze for risks..."
            )
            
            if st.button("Analyze Text") and text_input:
                with st.spinner("Analyzing text for risks..."):
                    risk_scores = risk_analyzer.analyze_text_risks(text_input)
                    
                    # Display risk scores
                    st.subheader("Risk Categories")
                    
                    # Create visualization
                    risk_df = pd.DataFrame([
                        {"Category": category, "Score": score}
                        for category, score in risk_scores.items()
                    ])
                    
                    fig = px.bar(
                        risk_df,
                        x="Category",
                        y="Score",
                        title="Risk Scores by Category",
                        color="Score",
                        color_continuous_scale="Reds"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display detailed scores
                    for category, score in risk_scores.items():
                        st.metric(category, format_risk_score(score))

# Query Interface Page
elif page == "Query Interface":
    st.header("🔍 Risk Query Interface")
    
    if not st.session_state.processed_documents:
        st.warning("Please upload and process documents first.")
    else:
        # Query input
        query = st.text_input(
            "Enter your risk-related question",
            placeholder="e.g., What are the main cybersecurity risks for financial institutions?"
        )
        
        # Query type selection
        query_type = st.selectbox(
            "Query Type",
            ["Semantic Search", "Knowledge Graph Query", "Hybrid (Graph + Vector)"]
        )
        
        if st.button("Search") and query:
            with st.spinner("Searching..."):
                if query_type == "Semantic Search":
                    results = vector_store.similarity_search(query, k=5)
                    
                    st.subheader("Search Results")
                    for i, result in enumerate(results, 1):
                        with st.expander(f"Result {i}"):
                            st.write(result['content'])
                            st.caption(f"Similarity: {result['score']:.3f}")
                
                elif query_type == "Knowledge Graph Query":
                    # Graph-based query
                    if st.session_state.knowledge_graph:
                        graph_results = kg_builder.query_graph(st.session_state.knowledge_graph, query)
                    else:
                        graph_results = ["No knowledge graph available"]
                    
                    st.subheader("Knowledge Graph Results")
                    for result in graph_results:
                        st.write(f"• {result}")
                
                elif query_type == "Hybrid (Graph + Vector)":
                    # Combine both approaches
                    vector_results = vector_store.similarity_search(query, k=3)
                    if st.session_state.knowledge_graph:
                        graph_results = kg_builder.query_graph(st.session_state.knowledge_graph, query)
                    else:
                        graph_results = ["No knowledge graph available"]
                    
                    # Generate comprehensive answer
                    combined_answer = llm_interface.generate_answer(
                        query, vector_results, graph_results
                    )
                    
                    st.subheader("AI-Generated Answer")
                    st.write(combined_answer)
                    
                    # Show supporting evidence
                    with st.expander("Supporting Evidence"):
                        st.write("**Vector Search Results:**")
                        for result in vector_results:
                            st.write(f"• {result['content'][:200]}...")
                        
                        st.write("**Knowledge Graph Results:**")
                        for result in graph_results:
                            st.write(f"• {result}")

# Risk Dashboard Page
elif page == "Risk Dashboard":
    st.header("📈 Risk Dashboard")
    
    if not st.session_state.processed_documents:
        st.warning("Please upload and process documents first.")
    else:
        # Dashboard metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Get dashboard data
        if st.session_state.knowledge_graph:
            dashboard_data = risk_analyzer.get_dashboard_data(st.session_state.knowledge_graph)
        else:
            dashboard_data = risk_analyzer.get_dashboard_data(None)
        
        with col1:
            st.metric("Total Risk Entities", dashboard_data['total_entities'])
        with col2:
            st.metric("High Risk Items", dashboard_data['high_risk_count'])
        with col3:
            st.metric("Avg Risk Score", f"{dashboard_data['avg_risk_score']:.2f}")
        with col4:
            st.metric("Risk Categories", dashboard_data['risk_categories'])
        
        # Risk distribution chart
        st.subheader("Risk Distribution")
        
        risk_dist_df = pd.DataFrame([
            {"Risk Level": level, "Count": count}
            for level, count in dashboard_data['risk_distribution'].items()
        ])
        
        fig_pie = px.pie(
            risk_dist_df,
            values="Count",
            names="Risk Level",
            title="Risk Level Distribution",
            color_discrete_map={
                "Low": "green",
                "Medium": "orange",
                "High": "red",
                "Critical": "darkred"
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Top risk factors
        st.subheader("Top Risk Factors")
        
        # Ensure we have risk factors to display
        top_risk_factors = dashboard_data.get('top_risk_factors', [])
        
        if not top_risk_factors:
            # Fallback sample data
            top_risk_factors = [
                {'factor': 'Cybersecurity Threats', 'score': 8.5},
                {'factor': 'Regulatory Compliance', 'score': 7.2},
                {'factor': 'Supply Chain Risk', 'score': 6.8},
                {'factor': 'Market Volatility', 'score': 6.1},
                {'factor': 'Operational Risk', 'score': 5.4}
            ]
        
        top_risks_df = pd.DataFrame(top_risk_factors)
        
        fig_bar = px.bar(
            top_risks_df,
            x="score",
            y="factor",
            orientation="h",
            title="Top Risk Factors by Score",
            color="score",
            color_continuous_scale="Reds"
        )
        fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Risk trends (if time series data available)
        if 'risk_trends' in dashboard_data:
            st.subheader("Risk Trends")
            
            trends_df = pd.DataFrame(dashboard_data['risk_trends'])
            
            fig_line = px.line(
                trends_df,
                x="date",
                y="risk_score",
                color="category",
                title="Risk Score Trends Over Time"
            )
            st.plotly_chart(fig_line, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "**IRAS** - Intelligent Risk Assessment System | "
    "Built with Streamlit, OpenAI, and Knowledge Graphs  \n"
    "© 2025 Developed by **Samuel Mbah**"
)
