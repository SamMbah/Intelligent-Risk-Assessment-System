import os
import json
from typing import List, Dict, Any, Optional
import streamlit as st
from openai import OpenAI

class LLMInterface:
    """Interface for LLM operations using OpenAI API."""
    
    def __init__(self):
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        self.model = "gpt-4o"
        self.client = self._initialize_client()
    
    def _initialize_client(self) -> Optional[OpenAI]:
        """Initialize OpenAI client."""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.warning("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
                return None
            
            return OpenAI(api_key=api_key)
        except Exception as e:
            st.error(f"Failed to initialize OpenAI client: {str(e)}")
            return None
    
    def generate_answer(self, query: str, vector_results: List[Dict], graph_results: List[str]) -> str:
        """Generate comprehensive answer using vector and graph search results."""
        if not self.client:
            return "LLM service not available. Please check your OpenAI API key."
        
        # Prepare context from search results
        context_parts = []
        
        # Add vector search context
        if vector_results:
            context_parts.append("Relevant document excerpts:")
            for i, result in enumerate(vector_results[:3], 1):
                context_parts.append(f"{i}. {result['content'][:500]}...")
        
        # Add knowledge graph context
        if graph_results:
            context_parts.append("\nKnowledge graph relationships:")
            for i, result in enumerate(graph_results[:3], 1):
                context_parts.append(f"{i}. {result}")
        
        context = "\n".join(context_parts)
        
        # Create prompt
        prompt = f"""You are an expert risk assessment analyst. Based on the provided context, answer the following question about business risks.

Question: {query}

Context:
{context}

Please provide a comprehensive answer that:
1. Directly addresses the question
2. References specific information from the context
3. Identifies key risk factors and relationships
4. Provides actionable insights where appropriate
5. Maintains a professional, analytical tone

Answer:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional risk assessment expert with deep knowledge of business risks, regulatory compliance, and risk management strategies."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.2
            )
            
            return response.choices[0].message.content or "No response generated"
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def analyze_risk_text(self, text: str) -> Dict[str, Any]:
        """Analyze text for risk factors and provide risk assessment."""
        if not self.client:
            return {"error": "LLM service not available"}
        
        prompt = f"""Analyze the following text for business risks. Provide a JSON response with risk assessment.

Text: {text}

Please analyze and return a JSON object with the following structure:
{{
    "risk_factors": ["list of identified risk factors"],
    "risk_categories": {{
        "operational": "score from 0-10",
        "financial": "score from 0-10", 
        "regulatory": "score from 0-10",
        "strategic": "score from 0-10",
        "reputational": "score from 0-10"
    }},
    "overall_risk_score": "score from 0-10",
    "key_concerns": ["list of main concerns"],
    "recommendations": ["list of risk mitigation recommendations"]
}}

Provide realistic scores based on the content analysis."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a risk assessment expert. Analyze text for business risks and respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            if content:
                result = json.loads(content)
                return result
            else:
                return {"error": "No response from LLM"}
            
        except Exception as e:
            return {
                "error": f"Analysis failed: {str(e)}",
                "risk_factors": [],
                "risk_categories": {
                    "operational": 0,
                    "financial": 0,
                    "regulatory": 0,
                    "strategic": 0,
                    "reputational": 0
                },
                "overall_risk_score": 0,
                "key_concerns": [],
                "recommendations": []
            }
    
    def generate_risk_report(self, entity: str, risk_data: Dict[str, Any]) -> str:
        """Generate a detailed risk assessment report."""
        if not self.client:
            return "LLM service not available. Please check your OpenAI API key."
        
        prompt = f"""Generate a professional risk assessment report for the entity: {entity}

Risk Data:
{json.dumps(risk_data, indent=2)}

Please create a comprehensive report with the following sections:
1. Executive Summary
2. Risk Profile Overview
3. Key Risk Factors
4. Risk Quantification
5. Mitigation Recommendations
6. Monitoring and Review Suggestions

The report should be professional, actionable, and suitable for business stakeholders."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional risk management consultant creating detailed risk assessment reports."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            return response.choices[0].message.content or "No report generated"
            
        except Exception as e:
            return f"Error generating report: {str(e)}"
    
    def categorize_risk(self, risk_description: str) -> Dict[str, Any]:
        """Categorize a risk description into standard risk categories."""
        if not self.client:
            return {"category": "unknown", "confidence": 0.0}
        
        prompt = f"""Categorize the following risk description into one of these categories:
- operational
- financial  
- regulatory
- strategic
- reputational
- cybersecurity

Risk Description: {risk_description}

Respond with JSON containing the category and confidence score (0-1):
{{
    "category": "category_name",
    "confidence": 0.85,
    "reasoning": "brief explanation"
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a risk categorization expert. Classify risks accurately and respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            if content:
                result = json.loads(content)
                return result
            else:
                return {"category": "unknown", "confidence": 0.0, "reasoning": "No response"}
            
        except Exception as e:
            return {
                "category": "unknown",
                "confidence": 0.0,
                "reasoning": f"Classification failed: {str(e)}"
            }
    
    def extract_entities_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract risk-related entities from text using LLM."""
        if not self.client:
            return []
        
        prompt = f"""Extract risk-related entities from the following text. Focus on:
- Organizations
- Risk factors
- Regulatory terms
- Financial instruments
- Geographic locations
- Key people/roles

Text: {text}

Return a JSON array of entities with this structure:
[
    {{
        "entity": "entity name",
        "type": "ORGANIZATION|RISK_FACTOR|REGULATORY|FINANCIAL|LOCATION|PERSON",
        "context": "brief context from text"
    }}
]"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at extracting business and risk-related entities from text. Respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            if content:
                result = json.loads(content)
                return result.get("entities", [])
            else:
                return []
            
        except Exception as e:
            st.error(f"Entity extraction failed: {str(e)}")
            return []
