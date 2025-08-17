#!/usr/bin/env python3
"""
AI-Powered Reference Validation Service.

This service uses LLM models to perform semantic validation of reference matches,
determining whether a found ArXiv paper truly corresponds to the original citation
by analyzing content similarity, authorship, and academic context.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class AIValidationResult:
    """Result of AI-powered reference validation."""
    is_valid: bool
    confidence_score: float
    semantic_match: bool
    author_analysis: Dict[str, Any]
    temporal_analysis: Dict[str, Any]
    content_analysis: Dict[str, Any]
    reasoning: str
    model_used: str


class AIReferenceValidator:
    """AI-powered semantic reference validation service."""
    
    def __init__(self):
        self.validation_prompt_template = self._create_validation_prompt()
    
    def _create_validation_prompt(self) -> str:
        """Create the prompt template for batch AI validation."""
        return """You are an expert academic librarian matching citations to papers.

TASK: Find the BEST MATCH for the original citation from the candidate papers below.

ORIGINAL CITATION:
- Title: {ref_title}
- Authors: {ref_authors}
- Year: {ref_year}
- Context: {ref_context}

CANDIDATE ARXIV PAPERS:
{candidates_list}

FLEXIBLE MATCHING RULES:
1. TITLE SIMILARITY: Primary matching criterion - look for semantically identical or very similar research work
2. AUTHOR OVERLAP: At least ONE author should match (allow minor variations: "J. Smith"="John Smith", "García"="Garcia")
3. RESEARCH IDENTITY: Must represent the same fundamental research work, not just related topics
4. PREFERENCE ORDER: Title similarity > Author match > Context relevance

DECISION LOGIC:
- ACCEPT if titles represent the same research work AND at least one author matches
- ACCEPT if titles are nearly identical (author match preferred but not required)
- ACCEPT if strong author overlap exists with reasonably similar title
- REJECT only if clearly different research or no meaningful overlap

NOTE: Year filtering has already been applied - all candidates are within 6-year range.

INSTRUCTIONS:
- Focus on finding the best semantic match for the same research work
- Be generous with matching if core research appears identical  
- Select the candidate number that BEST matches (1, 2, 3, etc.)
- Select 0 (REJECT ALL) only if no candidate represents the same research
- Explain your reasoning focusing on title and author similarity

Respond with JSON: {{"selected_candidate": number, "reasoning": "explanation"}}"""
    
    async def validate_reference_batch(
        self,
        reference: Dict[str, Any],
        arxiv_candidates: List[Dict[str, Any]],
        model_preference: str = "openai"
    ) -> Dict[str, Any]:
        """
        Use AI to find the best match from multiple ArXiv paper candidates.
        
        Args:
            reference: Original reference data
            arxiv_candidates: List of ArXiv paper metadata dictionaries
            model_preference: Preferred model ("openai" or "gemini")
            
        Returns:
            Dictionary with selected candidate index and reasoning
        """
        try:
            # Prepare candidates list for the prompt
            candidates_list = ""
            for i, candidate in enumerate(arxiv_candidates, 1):
                candidates_list += f"{i}. Title: {candidate.get('title', 'Unknown')}\n"
                candidates_list += f"   Authors: {candidate.get('authors', 'Unknown')}\n"
                candidates_list += f"   Year: {candidate.get('year', 'Unknown')}\n"
                candidates_list += f"   Abstract: {candidate.get('abstract', 'No abstract')[:200]}...\n\n"
            
            # Prepare data for the prompt
            prompt_data = {
                'ref_title': reference.get('cited_title', 'Unknown'),
                'ref_authors': reference.get('cited_authors', 'Unknown'),
                'ref_year': reference.get('cited_year', 'Unknown'),
                'ref_context': reference.get('reference_context', 'No context available')[:200],
                'candidates_list': candidates_list.strip()
            }
            
            # Create the full prompt
            prompt = self.validation_prompt_template.format(**prompt_data)
            
            # Get AI structured response
            ai_response = await self._query_ai_batch_model(prompt, model_preference)
            
            if ai_response and 'selected_candidate' in ai_response:
                return {
                    'selected_candidate': ai_response['selected_candidate'],
                    'reasoning': ai_response.get('reasoning', 'No reasoning provided'),
                    'model_used': model_preference,
                    'total_candidates': len(arxiv_candidates)
                }
            else:
                logger.warning("No valid structured response from AI model")
                return {
                    'selected_candidate': 0,
                    'reasoning': 'AI validation failed - rejecting all candidates',
                    'model_used': f"{model_preference}_fallback",
                    'total_candidates': len(arxiv_candidates)
                }
                
        except Exception as e:
            logger.error(f"AI batch validation failed: {e}")
            return {
                'selected_candidate': 0,
                'reasoning': f'AI validation error: {str(e)}',
                'model_used': f"{model_preference}_error",
                'total_candidates': len(arxiv_candidates)
            }

    async def validate_reference_match(
        self,
        reference: Dict[str, Any],
        arxiv_paper: Dict[str, Any],
        model_preference: str = "openai"
    ) -> AIValidationResult:
        """
        Use AI to validate if an ArXiv paper matches a reference citation.
        
        Args:
            reference: Original reference data
            arxiv_paper: ArXiv paper metadata
            model_preference: Preferred model ("gemini" or "openai")
            
        Returns:
            AIValidationResult with detailed analysis
        """
        try:
            # Prepare data for the prompt
            prompt_data = {
                'ref_title': reference.get('cited_title', 'Unknown'),
                'ref_authors': reference.get('cited_authors', 'Unknown'),
                'ref_year': reference.get('cited_year', 'Unknown'),
                'ref_context': reference.get('reference_context', 'No context available')[:200],
                'arxiv_title': arxiv_paper.get('title', 'Unknown'),
                'arxiv_authors': arxiv_paper.get('authors', 'Unknown'),
                'arxiv_year': arxiv_paper.get('year', 'Unknown'),
                'arxiv_abstract': arxiv_paper.get('abstract', 'No abstract available')[:500]
            }
            
            # Create the full prompt
            prompt = self.validation_prompt_template.format(**prompt_data)
            
            # Get AI structured response
            ai_response = await self._query_ai_model(prompt, model_preference)
            
            if ai_response and 'answer' in ai_response:
                return self._create_validation_result(ai_response, model_preference)
            else:
                logger.warning("No valid structured response from AI model")
                return self._create_fallback_result(model_preference)
                
        except Exception as e:
            logger.error(f"AI validation failed: {e}")
            return self._create_fallback_result(model_preference)
    
    async def _query_ai_model(self, prompt: str, model_preference: str) -> Optional[Dict[str, Any]]:
        """Query the preferred AI model with structured output for validation."""
        try:
            # Define the structured output schema
            validation_schema = {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "boolean",
                        "description": "True if the ArXiv paper matches the original citation, False otherwise"
                    },
                    "description": {
                        "type": "string",
                        "description": "Brief explanation of the validation decision and reasoning"
                    }
                },
                "required": ["answer", "description"],
                "additionalProperties": False
            }
            
            # Use a simple direct API call for validation
            # This is more reliable than trying to integrate with complex existing services
            json_prompt = f"{prompt}\n\nRespond with only JSON in this format: {{\"answer\": boolean, \"description\": \"explanation\"}}"
            
            # Log the validation request for debugging
            logger.info("🤖 AI VALIDATION REQUEST")
            logger.info(f"Model preference: {model_preference}")
            logger.info(f"Prompt length: {len(prompt)} characters")
            logger.info("📝 FULL AI PROMPT:")
            logger.info(f"--- PROMPT START ---\n{prompt}\n--- PROMPT END ---")
            
            # Try OpenAI first (recommended default)
            if model_preference == "openai":
                try:
                    import openai
                    import os
                    
                    api_key = os.getenv('OPENAI_API_KEY')
                    if api_key:
                        logger.debug("🟢 Attempting OpenAI API call...")
                        client = openai.AsyncOpenAI(api_key=api_key)
                        
                        response = await client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": json_prompt}],
                            response_format={"type": "json_object"}
                        )
                        
                        response_text = response.choices[0].message.content.strip()
                        
                        # Log raw response for debugging
                        logger.info("📥 OPENAI RAW RESPONSE")
                        logger.info(f"Response length: {len(response_text)} characters")
                        logger.info(f"Raw response: {response_text}")
                        
                        if response_text:
                            try:
                                response_data = json.loads(response_text)
                                logger.info("✅ OPENAI PARSED RESPONSE")
                                logger.info(f"Parsed JSON: {response_data}")
                                
                                if response_data and 'answer' in response_data:
                                    logger.info(f"🎯 Validation result: {response_data['answer']}")
                                    logger.info(f"💭 AI reasoning: {response_data.get('description', 'No description')}")
                                    return response_data
                                else:
                                    logger.warning("❌ OpenAI response missing required 'answer' field")
                            except json.JSONDecodeError as e:
                                logger.error(f"❌ OpenAI JSON parsing failed: {e}")
                                logger.error(f"Failed to parse: {response_text}")
                    else:
                        logger.warning("❌ No OpenAI API key available")
                        
                except ImportError:
                    logger.warning("❌ OpenAI library not available")
                except Exception as e:
                    logger.warning(f"❌ OpenAI validation failed: {e}")
                    logger.debug(f"OpenAI error details: {str(e)}")
            
            # Fallback to Gemini
            logger.debug("🔄 Falling back to Gemini...")
            try:
                import google.generativeai as genai
                import os
                
                # Configure Gemini
                api_key = os.getenv('GEMINI_API_KEY')
                if api_key:
                    logger.debug("🟢 Attempting Gemini API call...")
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-pro')
                    
                    # Generate response
                    response_obj = await asyncio.to_thread(model.generate_content, json_prompt)
                    response_text = response_obj.text.strip()
                    
                    # Log raw response for debugging
                    logger.info("📥 GEMINI RAW RESPONSE")
                    logger.info(f"Response length: {len(response_text)} characters")
                    logger.info(f"Raw response: {response_text}")
                    
                    # Parse JSON response
                    if response_text:
                        # Clean up response (remove markdown formatting if present)
                        cleaned_response = response_text
                        if response_text.startswith('```'):
                            cleaned_response = response_text.split('\\n', 1)[1]
                            logger.debug("Removed markdown start from response")
                        if cleaned_response.endswith('```'):
                            cleaned_response = cleaned_response.rsplit('\\n', 1)[0]
                            logger.debug("Removed markdown end from response")
                        
                        if cleaned_response != response_text:
                            logger.info(f"Cleaned response: {cleaned_response}")
                        
                        try:
                            response = json.loads(cleaned_response)
                            logger.info("✅ GEMINI PARSED RESPONSE")
                            logger.info(f"Parsed JSON: {response}")
                            
                            if response and 'answer' in response:
                                logger.info(f"🎯 Validation result: {response['answer']}")
                                logger.info(f"💭 AI reasoning: {response.get('description', 'No description')}")
                                return response
                            else:
                                logger.warning("❌ Response missing required 'answer' field")
                        except json.JSONDecodeError as e:
                            logger.error(f"❌ JSON parsing failed: {e}")
                            logger.error(f"Failed to parse: {cleaned_response}")
                else:
                    logger.warning("❌ No Gemini API key available")
                
            except ImportError:
                logger.warning("❌ Gemini library not available")
            except Exception as e:
                logger.warning(f"❌ Gemini validation failed: {e}")
                logger.debug(f"Gemini error details: {str(e)}")
            
            logger.error("❌ ALL AI VALIDATION METHODS FAILED")
            return None
            
        except Exception as e:
            logger.error(f"❌ AI model query failed: {e}")
            logger.debug(f"Query error details: {str(e)}")
            return None
    
    async def _query_ai_batch_model(self, prompt: str, model_preference: str) -> Optional[Dict[str, Any]]:
        """Query the AI model for batch validation with structured output."""
        try:
            # JSON prompt for batch validation
            json_prompt = f"{prompt}\n\nRespond with only JSON in this format: {{\"selected_candidate\": number, \"reasoning\": \"explanation\"}}"
            
            # Log the validation request for debugging
            logger.info("🤖 AI BATCH VALIDATION REQUEST")
            logger.info(f"Model preference: {model_preference}")
            logger.info(f"Prompt length: {len(prompt)} characters")
            logger.info("📝 FULL AI PROMPT:")
            logger.info(f"--- PROMPT START ---\n{prompt}\n--- PROMPT END ---")
            
            # Try OpenAI first (recommended default)
            if model_preference == "openai":
                try:
                    import openai
                    import os
                    
                    api_key = os.getenv('OPENAI_API_KEY')
                    if api_key:
                        logger.debug("🟢 Attempting OpenAI batch API call...")
                        client = openai.AsyncOpenAI(api_key=api_key)
                        
                        response = await client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": json_prompt}],
                            response_format={"type": "json_object"}
                        )
                        
                        response_text = response.choices[0].message.content.strip()
                        
                        # Log raw response for debugging
                        logger.info("📥 OPENAI BATCH RAW RESPONSE")
                        logger.info(f"Response length: {len(response_text)} characters")
                        logger.info(f"Raw response: {response_text}")
                        
                        if response_text:
                            try:
                                response_data = json.loads(response_text)
                                logger.info("✅ OPENAI BATCH PARSED RESPONSE")
                                logger.info(f"Parsed JSON: {response_data}")
                                
                                if response_data and 'selected_candidate' in response_data:
                                    candidate_num = response_data['selected_candidate']
                                    reasoning = response_data.get('reasoning', 'No reasoning provided')
                                    logger.info(f"🎯 Selected candidate: {candidate_num}")
                                    logger.info(f"💭 AI reasoning: {reasoning}")
                                    return response_data
                                else:
                                    logger.warning("❌ OpenAI response missing required 'selected_candidate' field")
                            except json.JSONDecodeError as e:
                                logger.error(f"❌ OpenAI JSON parsing failed: {e}")
                                logger.error(f"Failed to parse: {response_text}")
                    else:
                        logger.warning("❌ No OpenAI API key available")
                        
                except ImportError:
                    logger.warning("❌ OpenAI library not available")
                except Exception as e:
                    logger.warning(f"❌ OpenAI batch validation failed: {e}")
                    logger.debug(f"OpenAI error details: {str(e)}")
            
            # Fallback to Gemini
            logger.debug("🔄 Falling back to Gemini...")
            try:
                import google.generativeai as genai
                import os
                
                # Configure Gemini
                api_key = os.getenv('GEMINI_API_KEY')
                if api_key:
                    logger.debug("🟢 Attempting Gemini batch API call...")
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-pro')
                    
                    # Generate response
                    response_obj = await asyncio.to_thread(model.generate_content, json_prompt)
                    response_text = response_obj.text.strip()
                    
                    # Log raw response for debugging
                    logger.info("📥 GEMINI BATCH RAW RESPONSE")
                    logger.info(f"Response length: {len(response_text)} characters")
                    logger.info(f"Raw response: {response_text}")
                    
                    # Parse JSON response
                    if response_text:
                        # Clean up response (remove markdown formatting if present)
                        cleaned_response = response_text
                        if response_text.startswith('```'):
                            cleaned_response = response_text.split('\\n', 1)[1]
                            logger.debug("Removed markdown start from response")
                        if cleaned_response.endswith('```'):
                            cleaned_response = cleaned_response.rsplit('\\n', 1)[0]
                            logger.debug("Removed markdown end from response")
                        
                        if cleaned_response != response_text:
                            logger.info(f"Cleaned response: {cleaned_response}")
                        
                        try:
                            response = json.loads(cleaned_response)
                            logger.info("✅ GEMINI BATCH PARSED RESPONSE")
                            logger.info(f"Parsed JSON: {response}")
                            
                            if response and 'selected_candidate' in response:
                                candidate_num = response['selected_candidate']
                                reasoning = response.get('reasoning', 'No reasoning provided')
                                logger.info(f"🎯 Selected candidate: {candidate_num}")
                                logger.info(f"💭 AI reasoning: {reasoning}")
                                return response
                            else:
                                logger.warning("❌ Response missing required 'selected_candidate' field")
                        except json.JSONDecodeError as e:
                            logger.error(f"❌ JSON parsing failed: {e}")
                            logger.error(f"Failed to parse: {cleaned_response}")
                else:
                    logger.warning("❌ No Gemini API key available")
                
            except ImportError:
                logger.warning("❌ Gemini library not available")
            except Exception as e:
                logger.warning(f"❌ Gemini batch validation failed: {e}")
                logger.debug(f"Gemini error details: {str(e)}")
            
            logger.error("❌ ALL AI BATCH VALIDATION METHODS FAILED")
            return None
            
        except Exception as e:
            logger.error(f"❌ AI batch model query failed: {e}")
            logger.debug(f"Batch query error details: {str(e)}")
            return None
    
    def _create_validation_result(self, data: Dict[str, Any], model_used: str) -> AIValidationResult:
        """Create AIValidationResult from structured AI response."""
        try:
            # Extract structured response fields
            is_valid = data.get('answer', False)
            description = data.get('description', 'AI validation completed')
            
            # Set confidence based on AI decision
            confidence_score = 0.9 if is_valid else 0.1
            
            return AIValidationResult(
                is_valid=is_valid,
                confidence_score=confidence_score,
                semantic_match=is_valid,  # Simplified: if valid, then semantic match
                author_analysis={'ai_decision': is_valid, 'description': description},
                temporal_analysis={'ai_decision': is_valid, 'description': description},
                content_analysis={'ai_decision': is_valid, 'description': description},
                reasoning=description,
                model_used=model_used
            )
        except Exception as e:
            logger.warning(f"Failed to create validation result: {e}")
            return self._create_fallback_result(model_used)
    
    def _create_fallback_result(self, model_used: str) -> AIValidationResult:
        """Create a conservative fallback result when AI fails."""
        return AIValidationResult(
            is_valid=False,
            confidence_score=0.0,
            semantic_match=False,
            author_analysis={'error': 'AI validation unavailable'},
            temporal_analysis={'error': 'AI validation unavailable'},
            content_analysis={'error': 'AI validation unavailable'},
            reasoning='AI validation service unavailable - conservative rejection',
            model_used=f"{model_used}_fallback"
        )


class HybridReferenceValidator:
    """Combines rule-based and AI-powered validation for best results."""
    
    def __init__(self):
        self.ai_validator = AIReferenceValidator()
    
    async def validate_reference_match(
        self,
        reference: Dict[str, Any],
        arxiv_paper: Dict[str, Any],
        use_ai: bool = True,
        ai_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        Perform hybrid validation using both rule-based and AI approaches.
        
        Args:
            reference: Original reference data
            arxiv_paper: ArXiv paper metadata
            use_ai: Whether to use AI validation
            ai_threshold: Minimum AI confidence for acceptance
            
        Returns:
            Combined validation result
        """
        results = {
            'final_decision': False,
            'confidence_score': 0.0,
            'validation_methods': [],
            'reasoning': []
        }
        
        # 1. Rule-based validation (fast, conservative)
        try:
            from services.reference_validation_service import validate_arxiv_reference_match
            
            rule_based = await validate_arxiv_reference_match(reference, arxiv_paper)
            
            results['rule_based'] = {
                'is_valid': rule_based.is_valid,
                'confidence': rule_based.confidence_score,
                'author_match': rule_based.author_match,
                'year_match': rule_based.year_match,
                'title_similarity': rule_based.title_similarity
            }
            
            results['validation_methods'].append('rule_based')
            results['reasoning'].extend(rule_based.reasons)
            
            # If rule-based gives high confidence, we can potentially skip AI
            if rule_based.is_valid and rule_based.confidence_score >= 0.9:
                results['final_decision'] = True
                results['confidence_score'] = rule_based.confidence_score
                results['reasoning'].append('High-confidence rule-based match')
                return results
            
            # If rule-based completely rejects (no author/year match), skip AI
            if not rule_based.author_match and not rule_based.year_match:
                results['final_decision'] = False
                results['confidence_score'] = rule_based.confidence_score
                results['reasoning'].append('Rule-based rejection: no author/year match')
                return results
                
        except Exception as e:
            logger.warning(f"Rule-based validation failed: {e}")
            results['rule_based'] = {'error': str(e)}
        
        # 2. AI validation (for edge cases and semantic understanding)
        if use_ai:
            try:
                ai_result = await self.ai_validator.validate_reference_match(
                    reference, arxiv_paper
                )
                
                results['ai_validation'] = {
                    'is_valid': ai_result.is_valid,
                    'confidence': ai_result.confidence_score,
                    'semantic_match': ai_result.semantic_match,
                    'author_analysis': ai_result.author_analysis,
                    'reasoning': ai_result.reasoning,
                    'model_used': ai_result.model_used
                }
                
                results['validation_methods'].append('ai_powered')
                results['reasoning'].append(f"AI analysis: {ai_result.reasoning}")
                
                # Combine rule-based and AI results
                rule_confidence = results.get('rule_based', {}).get('confidence', 0.0)
                ai_confidence = ai_result.confidence_score
                
                # Weighted combination: rule-based 40%, AI 60%
                combined_confidence = (rule_confidence * 0.4) + (ai_confidence * 0.6)
                
                # Final decision: both must agree OR AI must be very confident
                rule_valid = results.get('rule_based', {}).get('is_valid', False)
                ai_valid = ai_result.is_valid and ai_result.confidence_score >= ai_threshold
                
                if rule_valid and ai_valid:
                    results['final_decision'] = True
                    results['confidence_score'] = combined_confidence
                    results['reasoning'].append('Both rule-based and AI validation passed')
                elif ai_valid and ai_result.confidence_score >= 0.9:
                    results['final_decision'] = True
                    results['confidence_score'] = ai_confidence
                    results['reasoning'].append('High-confidence AI override')
                else:
                    results['final_decision'] = False
                    results['confidence_score'] = combined_confidence
                    results['reasoning'].append('Insufficient confidence for match')
                
            except Exception as e:
                logger.warning(f"AI validation failed: {e}")
                results['ai_validation'] = {'error': str(e)}
                
                # Fall back to rule-based only
                rule_result = results.get('rule_based', {})
                results['final_decision'] = rule_result.get('is_valid', False)
                results['confidence_score'] = rule_result.get('confidence', 0.0)
                results['reasoning'].append('AI validation failed, using rule-based only')
        else:
            # Use rule-based only
            rule_result = results.get('rule_based', {})
            results['final_decision'] = rule_result.get('is_valid', False)
            results['confidence_score'] = rule_result.get('confidence', 0.0)
        
        return results


# Convenience functions for integration
async def validate_batch_with_ai(
    reference: Dict[str, Any],
    arxiv_candidates: List[Dict[str, Any]],
    model_preference: str = "openai"
) -> Dict[str, Any]:
    """
    Validate reference match against multiple ArXiv candidates using AI.
    
    Args:
        reference: Original reference data
        arxiv_candidates: List of ArXiv paper metadata dictionaries
        model_preference: Preferred AI model ("openai" or "gemini")
        
    Returns:
        Validation result with selected candidate index
    """
    validator = AIReferenceValidator()
    return await validator.validate_reference_batch(reference, arxiv_candidates, model_preference)

async def validate_with_ai(
    reference: Dict[str, Any],
    arxiv_paper: Dict[str, Any],
    use_hybrid: bool = True
) -> Dict[str, Any]:
    """
    Validate reference match using AI-powered analysis.
    
    Args:
        reference: Original reference data
        arxiv_paper: ArXiv paper metadata
        use_hybrid: Whether to use hybrid (rule-based + AI) validation
        
    Returns:
        Validation result with AI analysis
    """
    if use_hybrid:
        validator = HybridReferenceValidator()
        return await validator.validate_reference_match(reference, arxiv_paper)
    else:
        validator = AIReferenceValidator()
        result = await validator.validate_reference_match(reference, arxiv_paper)
        
        return {
            'final_decision': result.is_valid,
            'confidence_score': result.confidence_score,
            'ai_validation': {
                'is_valid': result.is_valid,
                'confidence': result.confidence_score,
                'semantic_match': result.semantic_match,
                'reasoning': result.reasoning,
                'model_used': result.model_used
            },
            'validation_methods': ['ai_powered'],
            'reasoning': [result.reasoning]
        }