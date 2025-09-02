import streamlit as st
import PyPDF2
import pandas as pd
import io
import re
from typing import List, Dict, Any
import json

class DocumentProcessor:
    """Handles document ingestion and preprocessing for the RAG system."""
    
    def __init__(self):
        self.chunk_size = 1000
        self.chunk_overlap = 200
    
    def process_pdf(self, file) -> List[Dict[str, Any]]:
        """Process PDF file and extract text chunks."""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return self._create_chunks(text, file.name, "pdf")
        
        except Exception as e:
            st.error(f"Error processing PDF {file.name}: {str(e)}")
            return []
    
    def process_text(self, file) -> List[Dict[str, Any]]:
        """Process text file and extract chunks."""
        try:
            text = str(file.read(), "utf-8")
            return self._create_chunks(text, file.name, "text")
        
        except Exception as e:
            st.error(f"Error processing text file {file.name}: {str(e)}")
            return []
    
    def process_csv(self, file) -> List[Dict[str, Any]]:
        """Process CSV file and convert to text chunks."""
        try:
            df = pd.read_csv(file)
            
            chunks = []
            for index, row in df.iterrows():
                # Convert each row to a text representation
                row_text = self._row_to_text(row, list(df.columns))
                chunk = {
                    'content': row_text,
                    'metadata': {
                        'source': file.name,
                        'type': 'csv',
                        'row_index': index,
                        'chunk_id': f"{file.name}_row_{index}"
                    }
                }
                chunks.append(chunk)
            
            return chunks
        
        except Exception as e:
            st.error(f"Error processing CSV file {file.name}: {str(e)}")
            return []
    
    def process_sample_data(self, sample_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process sample risk data."""
        chunks = []
        
        # Process risk scenarios
        for i, scenario in enumerate(sample_data.get('risk_scenarios', [])):
            text = f"Risk Scenario: {scenario['title']}\n"
            text += f"Description: {scenario['description']}\n"
            text += f"Impact: {scenario['impact']}\n"
            text += f"Likelihood: {scenario['likelihood']}\n"
            text += f"Category: {scenario['category']}\n"
            if 'mitigation' in scenario:
                text += f"Mitigation: {scenario['mitigation']}\n"
            
            chunk = {
                'content': text,
                'metadata': {
                    'source': 'sample_data',
                    'type': 'risk_scenario',
                    'scenario_id': i,
                    'chunk_id': f"sample_scenario_{i}"
                }
            }
            chunks.append(chunk)
        
        # Process regulatory requirements
        for i, req in enumerate(sample_data.get('regulatory_requirements', [])):
            text = f"Regulatory Requirement: {req['name']}\n"
            text += f"Description: {req['description']}\n"
            text += f"Jurisdiction: {req['jurisdiction']}\n"
            text += f"Compliance Level: {req['compliance_level']}\n"
            
            chunk = {
                'content': text,
                'metadata': {
                    'source': 'sample_data',
                    'type': 'regulatory_requirement',
                    'requirement_id': i,
                    'chunk_id': f"sample_reg_{i}"
                }
            }
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunks(self, text: str, source: str, doc_type: str) -> List[Dict[str, Any]]:
        """Create overlapping text chunks from document text."""
        # Clean the text
        text = self._clean_text(text)
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # If not the last chunk, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings within the overlap region
                sentence_end = text.rfind('.', start, end)
                if sentence_end != -1 and sentence_end > start + self.chunk_size // 2:
                    end = sentence_end + 1
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:  # Only add non-empty chunks
                chunk = {
                    'content': chunk_text,
                    'metadata': {
                        'source': source,
                        'type': doc_type,
                        'chunk_id': f"{source}_{chunk_id}",
                        'start_pos': start,
                        'end_pos': end
                    }
                }
                chunks.append(chunk)
                chunk_id += 1
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere with processing
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _row_to_text(self, row: pd.Series, columns: List[str]) -> str:
        """Convert a DataFrame row to natural language text."""
        text_parts = []
        
        for col in columns:
            value = row[col]
            if pd.notna(value) and value != "":
                # Format the column-value pair as natural language
                text_parts.append(f"{col}: {value}")
        
        return ". ".join(text_parts) + "."
    
    def get_processing_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about processed chunks."""
        if not chunks:
            return {
                'total_chunks': 0,
                'total_characters': 0,
                'avg_chunk_size': 0,
                'sources': []
            }
        
        total_chars = sum(len(chunk['content']) for chunk in chunks)
        sources = list(set(chunk['metadata']['source'] for chunk in chunks))
        
        return {
            'total_chunks': len(chunks),
            'total_characters': total_chars,
            'avg_chunk_size': total_chars // len(chunks),
            'sources': sources
        }
