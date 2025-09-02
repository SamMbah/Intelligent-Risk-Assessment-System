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
from sklearn.feature_extraction.text import TfidfVectorizer

# Try to import sentence-transformers, fall back to TF-IDF if not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

class VectorStore:
    """Manages vector embeddings and similarity search using Qdrant with robust fallbacks."""
    
    def __init__(self):
        self.client = None
        self.collection_name = "risk_documents"
        self.embedding_model = None
        self.tfidf_vectorizer = None
        self.vector_size = 384  # Default size
        self._initialize()
    
    def _initialize(self):
        """Initialize Qdrant client and embedding model with fallbacks."""
        try:
            # Initialize Qdrant client (in-memory mode)
            self.client = QdrantClient(":memory:")
            
            # Try to initialize sentence transformers
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                    self.vector_size = self.embedding_model.get_sentence_embedding_dimension()
                    st.success("Using SentenceTransformers for high-quality embeddings")
                except Exception as e:
                    st.warning(f"SentenceTransformers failed: {e}. Using TF-IDF fallback.")
                    self.embedding_model = None
            else:
                st.info("Using TF-IDF embeddings (fallback mode)")
                self.embedding_model = None
            
            # If no sentence transformers, use TF-IDF
            if not self.embedding_model:
                self.vector_size = 300  # Reasonable size for TF-IDF
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=self.vector_size,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
            
            # Create collection
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
            st.error(f"Failed to initialize vector store: {str(e)}")
            self.client = None
    
    def add_documents(self, chunks: List[Dict[str, Any]]):
        """Add document chunks to the vector store."""
        if not self.client:
            st.error("Vector store not properly initialized")
            return
        
        if not chunks:
            return
        
        try:
            # Extract texts for embedding
            texts = []
            for chunk in chunks:
                content = chunk.get('content', chunk.get('text', ''))
                if content.strip():
                    texts.append(content)
            
            if not texts:
                st.warning("No valid text content found in chunks")
                return
            
            # Generate embeddings
            embeddings = self._generate_embeddings(texts)
            
            # Create Qdrant points
            points = []
            for i, chunk in enumerate(chunks):
                content = chunk.get('content', chunk.get('text', ''))
                if not content.strip():
                    continue
                    
                metadata = chunk.get('metadata', {})
                
                # Generate valid UUID from content hash
                text_hash = hashlib.md5(content.encode()).hexdigest()
                # Convert hash to valid UUID format
                doc_id = f"{text_hash[:8]}-{text_hash[8:12]}-{text_hash[12:16]}-{text_hash[16:20]}-{text_hash[20:32]}"
                
                point = PointStruct(
                    id=doc_id,
                    vector=embeddings[i],
                    payload={
                        'content': content,
                        'source': metadata.get('source', 'unknown'),
                        'chunk_index': metadata.get('chunk_index', i),
                        'document_id': metadata.get('document_id', 'unknown')
                    }
                )
                points.append(point)
            
            # Add to Qdrant
            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                st.success(f"Added {len(points)} document chunks to vector store")
            
        except Exception as e:
            st.error(f"Error adding documents: {str(e)}")
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using best available method."""
        try:
            if self.embedding_model:
                # Use SentenceTransformers
                embeddings = self.embedding_model.encode(texts)
                return embeddings.tolist()
            else:
                # Use TF-IDF fallback
                if not hasattr(self, '_fitted_tfidf') or not self._fitted_tfidf:
                    # Fit TF-IDF on provided texts
                    self.tfidf_vectorizer.fit(texts)
                    self._fitted_tfidf = True
                
                tfidf_matrix = self.tfidf_vectorizer.transform(texts)
                
                # Convert to dense array and pad/truncate to vector_size
                embeddings = []
                for i in range(tfidf_matrix.shape[0]):
                    dense_vec = tfidf_matrix[i].toarray().flatten()
                    
                    # Pad or truncate to vector_size
                    if len(dense_vec) < self.vector_size:
                        padded = np.zeros(self.vector_size)
                        padded[:len(dense_vec)] = dense_vec
                        embeddings.append(padded.tolist())
                    else:
                        embeddings.append(dense_vec[:self.vector_size].tolist())
                
                return embeddings
        except Exception as e:
            st.error(f"Embedding generation failed: {e}")
            # Return random embeddings as last resort
            return [np.random.normal(0, 0.1, self.vector_size).tolist() for _ in texts]
    
    def similarity_search(self, query: str, k: int = 5, filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Perform similarity search."""
        if not self.client:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self._generate_embeddings([query])[0]
            
            # Prepare filter
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
            
            # Search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k,
                query_filter=search_filter
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'content': result.payload.get('content', ''),
                    'metadata': {
                        'source': result.payload.get('source', 'unknown'),
                        'chunk_index': result.payload.get('chunk_index', 0),
                        'document_id': result.payload.get('document_id', 'unknown')
                    },
                    'score': float(result.score)
                })
            
            return formatted_results
            
        except Exception as e:
            st.error(f"Search failed: {str(e)}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        if not self.client:
            return {}
        
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                'total_documents': info.points_count,
                'vector_size': info.config.params.vectors.size,
                'distance_metric': info.config.params.vectors.distance
            }
        except Exception as e:
            return {'error': str(e)}
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        if self.client:
            try:
                self.client.delete_collection(self.collection_name)
                self._initialize()  # Recreate collection
                st.success("Vector store cleared")
            except Exception as e:
                st.error(f"Failed to clear collection: {str(e)}")