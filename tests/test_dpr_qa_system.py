#!/usr/bin/env python3
"""
Unit tests for the DPR Question Answering System
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import torch

# Import the system to test
from dpr_qa_system import DPRQuestionAnsweringSystem


class TestDPRQuestionAnsweringSystem:
    """Test cases for DPRQuestionAnsweringSystem class"""
    
    @pytest.fixture
    def mock_qa_system(self):
        """Create a mock QA system for testing"""
        with patch('dpr_qa_system.DPRContextEncoder') as mock_context_encoder, \
             patch('dpr_qa_system.DPRContextEncoderTokenizer') as mock_context_tokenizer, \
             patch('dpr_qa_system.DPRQuestionEncoder') as mock_question_encoder, \
             patch('dpr_qa_system.DPRQuestionEncoderTokenizer') as mock_question_tokenizer:
            
            # Mock the encoders and tokenizers
            mock_context_encoder.from_pretrained.return_value = Mock()
            mock_context_tokenizer.from_pretrained.return_value = Mock()
            mock_question_encoder.from_pretrained.return_value = Mock()
            mock_question_tokenizer.from_pretrained.return_value = Mock()
            
            qa_system = DPRQuestionAnsweringSystem()
            return qa_system
    
    def test_initialization(self, mock_qa_system):
        """Test that the QA system initializes correctly"""
        assert mock_qa_system.documents == []
        assert len(mock_qa_system.document_embeddings) == 0
        assert mock_qa_system.context_encoder is not None
        assert mock_qa_system.question_encoder is not None
    
    def test_embed_documents(self, mock_qa_system):
        """Test document embedding functionality"""
        # Mock the tokenizer and encoder outputs
        mock_inputs = {'input_ids': torch.tensor([[1, 2, 3]])}
        mock_qa_system.context_tokenizer.return_value = mock_inputs
        
        # Mock encoder output
        mock_output = Mock()
        mock_output.pooler_output = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        mock_qa_system.context_encoder.return_value = mock_output
        
        documents = ["Test document 1", "Test document 2"]
        embeddings = mock_qa_system.embed_documents(documents)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 2  # Two documents
        assert embeddings.shape[1] == 4  # Four dimensions in mock embedding
    
    def test_embed_question(self, mock_qa_system):
        """Test question embedding functionality"""
        # Mock the tokenizer and encoder outputs
        mock_inputs = {'input_ids': torch.tensor([[1, 2, 3]])}
        mock_qa_system.question_tokenizer.return_value = mock_inputs
        
        # Mock encoder output
        mock_output = Mock()
        mock_output.pooler_output = torch.tensor([[0.5, 0.6, 0.7, 0.8]])
        mock_qa_system.question_encoder.return_value = mock_output
        
        question = "What is the test question?"
        embedding = mock_qa_system.embed_question(question)
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 4  # Four dimensions in mock embedding
    
    def test_add_documents(self, mock_qa_system):
        """Test adding documents to the knowledge base"""
        # Mock the embed_documents method
        mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_qa_system.embed_documents = Mock(return_value=mock_embeddings)
        
        documents = ["Document 1", "Document 2"]
        mock_qa_system.add_documents(documents)
        
        assert len(mock_qa_system.documents) == 2
        assert mock_qa_system.documents == documents
        np.testing.assert_array_equal(mock_qa_system.document_embeddings, mock_embeddings)
    
    def test_retrieve_relevant_documents(self, mock_qa_system):
        """Test document retrieval functionality"""
        # Set up mock data
        mock_qa_system.documents = ["Doc 1", "Doc 2", "Doc 3"]
        mock_qa_system.document_embeddings = np.array([
            [0.1, 0.2],
            [0.3, 0.4], 
            [0.5, 0.6]
        ])
        
        # Mock question embedding
        mock_qa_system.embed_question = Mock(return_value=np.array([0.3, 0.4]))
        
        results = mock_qa_system.retrieve_relevant_documents("Test question", top_k=2)
        
        assert len(results) == 2
        assert all(isinstance(result, tuple) and len(result) == 2 for result in results)
        assert all(isinstance(result[0], str) and isinstance(result[1], float) for result in results)
    
    def test_answer_question_with_documents(self, mock_qa_system):
        """Test answering questions when documents are available"""
        # Set up mock data
        mock_qa_system.documents = ["This is a test document about machine learning."]
        mock_qa_system.document_embeddings = np.array([[0.1, 0.2, 0.3]])
        
        # Mock the retrieve_relevant_documents method
        mock_qa_system.retrieve_relevant_documents = Mock(
            return_value=[("This is a test document about machine learning.", 0.8)]
        )
        
        result = mock_qa_system.answer_question("What is machine learning?")
        
        assert "question" in result
        assert "answer" in result
        assert "relevant_documents" in result
        assert "confidence_scores" in result
        assert "best_match_score" in result
        assert result["question"] == "What is machine learning?"
    
    def test_answer_question_no_documents(self, mock_qa_system):
        """Test answering questions when no documents are available"""
        result = mock_qa_system.answer_question("What is machine learning?")
        
        assert result["answer"] == "No relevant documents found in the knowledge base."
        assert result["relevant_documents"] == []
        assert result["confidence_scores"] == []
    
    def test_extract_answer_from_context(self, mock_qa_system):
        """Test answer extraction from context"""
        question = "What is AI?"
        context = "Artificial Intelligence is amazing. It can solve complex problems. AI is the future."
        
        answer = mock_qa_system._extract_answer_from_context(question, context)
        
        # Should return first two sentences
        expected = "Artificial Intelligence is amazing. It can solve complex problems."
        assert answer == expected
    
    def test_extract_answer_short_context(self, mock_qa_system):
        """Test answer extraction from short context"""
        question = "What is AI?"
        context = "AI is artificial intelligence."
        
        answer = mock_qa_system._extract_answer_from_context(question, context)
        
        # Should return the entire context for short texts
        assert answer == context
    
    @patch('builtins.open')
    @patch('json.dump')
    def test_save_knowledge_base(self, mock_json_dump, mock_open, mock_qa_system):
        """Test saving knowledge base to file"""
        mock_qa_system.documents = ["Doc 1", "Doc 2"]
        mock_qa_system.document_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        
        mock_qa_system.save_knowledge_base("test.json")
        
        mock_open.assert_called_once_with("test.json", 'w', encoding='utf-8')
        mock_json_dump.assert_called_once()
    
    @patch('builtins.open')
    @patch('json.load')
    def test_load_knowledge_base(self, mock_json_load, mock_open, mock_qa_system):
        """Test loading knowledge base from file"""
        mock_data = {
            "documents": ["Doc 1", "Doc 2"],
            "embeddings": [[0.1, 0.2], [0.3, 0.4]]
        }
        mock_json_load.return_value = mock_data
        
        mock_qa_system.load_knowledge_base("test.json")
        
        mock_open.assert_called_once_with("test.json", 'r', encoding='utf-8')
        assert mock_qa_system.documents == ["Doc 1", "Doc 2"]
        np.testing.assert_array_equal(
            mock_qa_system.document_embeddings, 
            np.array([[0.1, 0.2], [0.3, 0.4]])
        )


class TestIntegration:
    """Integration tests that test the system end-to-end"""
    
    @pytest.mark.slow
    def test_full_pipeline_mock(self):
        """Test the full pipeline with mocked models"""
        with patch('dpr_qa_system.DPRContextEncoder') as mock_context_encoder, \
             patch('dpr_qa_system.DPRContextEncoderTokenizer') as mock_context_tokenizer, \
             patch('dpr_qa_system.DPRQuestionEncoder') as mock_question_encoder, \
             patch('dpr_qa_system.DPRQuestionEncoderTokenizer') as mock_question_tokenizer:
            
            # Setup mocks
            mock_context_encoder.from_pretrained.return_value = Mock()
            mock_context_tokenizer.from_pretrained.return_value = Mock()
            mock_question_encoder.from_pretrained.return_value = Mock()
            mock_question_tokenizer.from_pretrained.return_value = Mock()
            
            qa_system = DPRQuestionAnsweringSystem()
            
            # Mock tokenizer returns
            qa_system.context_tokenizer.return_value = {'input_ids': torch.tensor([[1, 2, 3]])}
            qa_system.question_tokenizer.return_value = {'input_ids': torch.tensor([[1, 2, 3]])}
            
            # Mock encoder outputs
            context_output = Mock()
            context_output.pooler_output = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
            qa_system.context_encoder.return_value = context_output
            
            question_output = Mock()
            question_output.pooler_output = torch.tensor([[0.2, 0.3, 0.4, 0.5]])
            qa_system.question_encoder.return_value = question_output
            
            # Test the full pipeline
            documents = ["Python is a programming language.", "Machine learning is AI."]
            qa_system.add_documents(documents)
            
            result = qa_system.answer_question("What is Python?")
            
            assert "answer" in result
            assert "confidence_scores" in result
            assert len(result["relevant_documents"]) > 0


def test_cosine_similarity_calculation():
    """Test cosine similarity calculation works correctly"""
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Test vectors
    vec1 = np.array([[1, 0, 0]])
    vec2 = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]])
    
    similarities = cosine_similarity(vec1, vec2).flatten()
    
    # Should be [1.0, 0.0, 0.707...]
    assert abs(similarities[0] - 1.0) < 1e-6
    assert abs(similarities[1] - 0.0) < 1e-6
    assert abs(similarities[2] - 0.7071067811865476) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__]) 
