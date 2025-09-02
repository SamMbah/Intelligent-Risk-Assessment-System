from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.http import models
import numpy as np
from typing import List, Dict, Any, Optional
import streamlit as st
import uuid
import hashlib
import re
from collections import Counter

# Try to import sentence-transformers, fall back to simple embeddings if not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    
# Simple TF-IDF fallback for embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class VectorStore:
    """Manages vector embeddings and similarity search using Qdrant."""
    
    def __init__(self):
        self.client = None
        self.collection_name = "risk_documents"
        self.embedding_model = None
        self.vector_size = 384  # Default for all-MiniLM-L6-v2
        self._initialize()
    
    def _initialize(self):
        """Initialize Qdrant client and embedding model."""
        try:
            # Initialize Qdrant client (in-memory mode for simplicity)
            self.client = QdrantClient(":memory:")
            
            # Initialize embedding model
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.vector_size = self.embedding_model.get_sentence_embedding_dimension()
            else:
                st.warning("Sentence transformers not available. Using fallback TF-IDF embeddings.")
                self.embedding_model = None
                self.vector_size = 384  # Fixed size for fallback
            
            # Create collection if it doesn't exist
            try:
                self.client.get_collection(self.collection_name)
            except:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
            
        except Exception as e:
            st.error(f"Failed to initialize Qdrant vector store: {str(e)}")
            self.client = None
            self.embedding_model = None
    
    def add_documents(self, chunks: List[Dict[str, Any]]):
        """Add document chunks to the vector store."""
        if not self.client:
            st.error("Qdrant client not properly initialized")
            return
        
        if not chunks:
            return
        
        try:
            # Prepare data for Qdrant
            documents = []
            points = []
            
            for chunk in chunks:
                content = chunk['content']
                metadata = chunk['metadata']
                documents.append(content)
                
                # Generate unique ID - Qdrant requires either int or valid UUID
                chunk_id = metadata.get('chunk_id', None)
                if chunk_id is None or not self._is_valid_uuid(chunk_id):
                    chunk_id = str(uuid.uuid4())
                
                # Prepare point for later insertion
                points.append({
                    'id': chunk_id,
                    'content': content,
                    'metadata': metadata
                })
            
            # Generate embeddings
            with st.spinner("Generating embeddings..."):
                if self.embedding_model and SENTENCE_TRANSFORMERS_AVAILABLE:
                    embeddings = self.embedding_model.encode(documents).tolist()
                else:
                    # Fallback to simple TF-IDF-like embeddings
                    embeddings = self._generate_simple_embeddings(documents)
            
            # Create PointStruct objects for Qdrant
            qdrant_points = []
            for i, point in enumerate(points):
                qdrant_points.append(
                    PointStruct(
                        id=point['id'],
                        vector=embeddings[i],
                        payload={
                            'content': point['content'],
                            'metadata': point['metadata']
                        }
                    )
                )
            
            # Add to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=qdrant_points
            )
            
            st.success(f"Added {len(chunks)} documents to Qdrant vector store")
            
        except Exception as e:
            st.error(f"Error adding documents to vector store: {str(e)}")
    
    def similarity_search(self, query: str, k: int = 5, filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Perform similarity search on the vector store."""
        if not self.client:
            return []
        
        try:
            # Generate query embedding
            if self.embedding_model and SENTENCE_TRANSFORMERS_AVAILABLE:
                query_embedding = self.embedding_model.encode([query]).tolist()[0]
            else:
                # Fallback to simple embedding
                query_embedding = self._generate_simple_embeddings([query])[0]
            
            # Prepare filter if provided
            search_filter = None
            if filter_metadata:
                conditions = []
                for key, value in filter_metadata.items():
                    conditions.append(
                        FieldCondition(
                            key=f"metadata.{key}",
                            match=MatchValue(value=value)
                        )
                    )
                if conditions:
                    search_filter = Filter(must=conditions)
            
            # Perform search with Qdrant
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k,
                query_filter=search_filter
            )
            
            # Format results
            formatted_results = []
            for hit in search_result:
                result = {
                    'content': hit.payload.get('content', ''),
                    'metadata': hit.payload.get('metadata', {}),
                    'score': hit.score,
                    'id': str(hit.id)
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            st.error(f"Error performing similarity search: {str(e)}")
            return []
    
    def search_by_category(self, category: str, k: int = 10) -> List[Dict[str, Any]]:
        """Search for documents by risk category."""
        return self.similarity_search(
            query=f"risk {category}",
            k=k,
            filter_metadata={"type": "risk_scenario"}
        )
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection."""
        if not self.client:
            return {
                'total_documents': 0,
                'collection_name': 'Not initialized'
            }
        
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                'total_documents': collection_info.points_count,
                'collection_name': self.collection_name,
                'vectors_count': collection_info.vectors_count,
                'status': collection_info.status
            }
        except Exception as e:
            return {
                'total_documents': 0,
                'collection_name': f'Error: {str(e)}'
            }
    
    def delete_collection(self):
        """Delete the current collection (useful for testing)."""
        if self.client:
            try:
                self.client.delete_collection(self.collection_name)
                
                # Recreate collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                st.success("Qdrant collection reset successfully")
            except Exception as e:
                st.error(f"Error resetting collection: {str(e)}")
    
    def hybrid_search(self, query: str, graph_results: List[str], k: int = 3) -> List[Dict[str, Any]]:
        """Combine vector search with knowledge graph results."""
        # Get vector search results
        vector_results = self.similarity_search(query, k=k)
        
        # Enhance with graph context
        enhanced_results = []
        for result in vector_results:
            # Check if any graph results are related to this document
            content = result['content'].lower()
            related_graph_info = []
            
            for graph_result in graph_results:
                # Simple keyword matching - can be improved
                if any(word in content for word in graph_result.lower().split()):
                    related_graph_info.append(graph_result)
            
            enhanced_result = result.copy()
            enhanced_result['graph_context'] = related_graph_info
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def get_similar_documents(self, document_id: str, k: int = 5) -> List[Dict[str, Any]]:
        """Find documents similar to a given document."""
        if not self.client:
            return []
        
        try:
            # Get the document by ID
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[document_id],
                with_payload=True,
                with_vectors=True
            )
            
            if not points:
                return []
            
            # Use the document's vector for similarity search
            document_vector = points[0].vector
            
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=document_vector,
                limit=k + 1  # +1 to account for the original document
            )
            
            # Format results and exclude the original document
            formatted_results = []
            for hit in search_result:
                if str(hit.id) != document_id:  # Exclude original document
                    result = {
                        'content': hit.payload.get('content', ''),
                        'metadata': hit.payload.get('metadata', {}),
                        'score': hit.score,
                        'id': str(hit.id)
                    }
                    formatted_results.append(result)
            
            return formatted_results[:k]  # Return only k results
            
        except Exception as e:
            st.error(f"Error finding similar documents: {str(e)}")
            return []
    
    def _generate_simple_embeddings(self, documents: List[str]) -> List[List[float]]:
        """Generate simple TF-IDF-like embeddings as fallback."""
        # Simple word frequency-based embeddings
        all_words = set()
        doc_words = []
        
        for doc in documents:
            words = self._tokenize(doc)
            doc_words.append(words)
            all_words.update(words)
        
        vocab = sorted(list(all_words))[:1000]  # Limit vocabulary size
        embeddings = []
        
        for words in doc_words:
            word_counts = Counter(words)
            embedding = []
            
            for word in vocab:
                # Simple TF-IDF approximation
                tf = word_counts.get(word, 0) / max(len(words), 1)
                embedding.append(tf)
            
            # Pad to ensure consistent dimensionality
            while len(embedding) < 384:  # Match sentence transformer dimensions
                embedding.append(0.0)
            
            embeddings.append(embedding[:384])  # Truncate to 384 dimensions
        
        return embeddings
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for fallback embeddings."""
        # Convert to lowercase and extract words
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return words
    
    def _is_valid_uuid(self, uuid_string: str) -> bool:
        """Check if a string is a valid UUID."""
        try:
            uuid.UUID(uuid_string)
            return True
        except (ValueError, TypeError):
            return False
