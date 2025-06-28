import torch
import numpy as np
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from typing import List, Dict, Tuple
import argparse


class DPRQuestionAnsweringSystem:
    """
    A Question Answering system using Dense Passage Retrieval (DPR) encoders.
    Embeds documents using DPRContextEncoder and questions using DPRQuestionEncoder.
    """
    
    def __init__(self, context_model_name: str = "facebook/dpr-ctx_encoder-single-nq-base",
                 question_model_name: str = "facebook/dpr-question_encoder-single-nq-base"):
        """
        Initialize the DPR QA system with pre-trained encoders.
        
        Args:
            context_model_name: Name of the DPR context encoder model
            question_model_name: Name of the DPR question encoder model
        """
        print("Loading DPR models...")
        
        # Load context encoder
        self.context_encoder = DPRContextEncoder.from_pretrained(context_model_name)
        self.context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(context_model_name)
        
        # Load question encoder
        self.question_encoder = DPRQuestionEncoder.from_pretrained(question_model_name)
        self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(question_model_name)
        
        # Set models to evaluation mode
        self.context_encoder.eval()
        self.question_encoder.eval()
        
        # Storage for document embeddings and metadata
        self.document_embeddings = []
        self.documents = []
        
        print("DPR models loaded successfully!")
    
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of documents using DPRContextEncoder.
        
        Args:
            documents: List of document texts to embed
            
        Returns:
            numpy array of document embeddings
        """
        print(f"Embedding {len(documents)} documents...")
        
        embeddings = []
        from pdb import set_trace as st
        with torch.no_grad():
            for i, doc in enumerate(documents):
                # Tokenize the document
                st()
                inputs = self.context_tokenizer(
                    doc,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True
                )
                
                # Get embeddings
                outputs = self.context_encoder(**inputs)
                embedding = outputs.pooler_output.cpu().numpy()
                embeddings.append(embedding.flatten())
                
                if (i + 1) % 10 == 0:
                    print(f"Embedded {i + 1}/{len(documents)} documents")
        
        embeddings_array = np.array(embeddings)
        print(f"Document embedding shape: {embeddings_array.shape}")
        
        return embeddings_array
    
    def embed_question(self, question: str) -> np.ndarray:
        """
        Create embedding for a question using DPRQuestionEncoder.
        
        Args:
            question: Question text to embed
            
        Returns:
            numpy array of question embedding
        """
        with torch.no_grad():
            # Tokenize the question
            inputs = self.question_tokenizer(
                question,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            # Get embedding
            outputs = self.question_encoder(**inputs)
            embedding = outputs.pooler_output.cpu().numpy()
            
        return embedding.flatten()
    
    def add_documents(self, documents: List[str]):
        """
        Add documents to the knowledge base by creating their embeddings.
        
        Args:
            documents: List of document texts to add
        """
        # Create embeddings for the documents
        doc_embeddings = self.embed_documents(documents)
        
        # Store documents and their embeddings
        self.documents.extend(documents)
        if len(self.document_embeddings) == 0:
            self.document_embeddings = doc_embeddings
        else:
            self.document_embeddings = np.vstack([self.document_embeddings, doc_embeddings])
        
        print(f"Added {len(documents)} documents. Total documents: {len(self.documents)}")
    
    def retrieve_relevant_documents(self, question: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve the most relevant documents for a given question.
        
        Args:
            question: Question to find relevant documents for
            top_k: Number of top documents to retrieve
            
        Returns:
            List of tuples (document, similarity_score)
        """
        if len(self.documents) == 0:
            print("No documents in the knowledge base!")
            return []
        
        # Embed the question
        question_embedding = self.embed_question(question)
        
        # Calculate cosine similarity between question and all documents
        similarities = cosine_similarity(
            question_embedding.reshape(1, -1),
            self.document_embeddings
        ).flatten()
        
        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            score = similarities[idx]
            results.append((doc, score))
        
        return results
    
    def answer_question(self, question: str, top_k: int = 3) -> Dict:
        """
        Answer a question by retrieving relevant documents and providing context.
        
        Args:
            question: Question to answer
            top_k: Number of top documents to consider for answering
            
        Returns:
            Dictionary containing the answer, relevant documents, and scores
        """
        print(f"\nAnswering question: '{question}'")
        
        # Retrieve relevant documents
        relevant_docs = self.retrieve_relevant_documents(question, top_k)
        
        if not relevant_docs:
            return {
                "question": question,
                "answer": "No relevant documents found in the knowledge base.",
                "relevant_documents": [],
                "confidence_scores": []
            }
        
        # Prepare the answer based on the most relevant document
        best_doc, best_score = relevant_docs[0]
        
        # Simple answer extraction (in a real system, you might use a reader model)
        answer = self._extract_answer_from_context(question, best_doc)
        
        result = {
            "question": question,
            "answer": answer,
            "relevant_documents": [doc for doc, _ in relevant_docs],
            "confidence_scores": [float(score) for _, score in relevant_docs],
            "best_match_score": float(best_score)
        }
        
        return result
    
    def _extract_answer_from_context(self, question: str, context: str) -> str:
        """
        Simple answer extraction from context.
        In a production system, you would use a more sophisticated reader model.
        
        Args:
            question: The question being asked
            context: The relevant context document
            
        Returns:
            Extracted answer or context snippet
        """
        # For this example, we'll return the most relevant context
        # In practice, you would use a reader model like BERT for span extraction
        
        # Simple heuristic: return first few sentences of the context
        sentences = context.split('. ')
        if len(sentences) > 2:
            return '. '.join(sentences[:2]) + '.'
        else:
            return context
    
    def save_knowledge_base(self, filepath: str):
        """Save the current knowledge base to a file."""
        data = {
            "documents": self.documents,
            "embeddings": self.document_embeddings.tolist() if len(self.document_embeddings) > 0 else []
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Knowledge base saved to {filepath}")
    
    def load_knowledge_base(self, filepath: str):
        """Load a knowledge base from a file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.documents = data["documents"]
        self.document_embeddings = np.array(data["embeddings"]) if data["embeddings"] else np.array([])
        
        print(f"Knowledge base loaded from {filepath}")
        print(f"Loaded {len(self.documents)} documents")


def main():
    """
    Main function to demonstrate the DPR Question Answering system.
    """
    parser = argparse.ArgumentParser(description="DPR Question Answering System")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--question", type=str, help="Single question to answer")
    parser.add_argument("--documents", nargs="+", help="Documents to add to knowledge base")
    
    args = parser.parse_args()
    
    # Initialize the DPR QA system
    qa_system = DPRQuestionAnsweringSystem()
    
    # Sample documents for demonstration
    sample_documents = [
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower. The tower is 324 metres tall, and was the world's tallest man-made structure until the Chrysler Building was built in New York in 1930.",
        
        "Python is a high-level, interpreted programming language with dynamic semantics. Its high-level built-in data structures, combined with dynamic typing and dynamic binding, make it very attractive for Rapid Application Development, as well as for use as a scripting or glue language to connect existing components together.",
        
        "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention. Machine learning algorithms build a model based on training data in order to make predictions or decisions without being explicitly programmed to do so.",
        
        "The Great Wall of China is a series of fortifications that were built across the historical northern borders of ancient Chinese states and Imperial China as protection against various nomadic groups from the Eurasian Steppe. The wall stretches over 13,000 miles and was built over many centuries by millions of workers.",
        
        "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving. AI can be categorized as either weak or strong AI."
    ]
    
    # Add documents from command line or use samples
    if args.documents:
        qa_system.add_documents(args.documents)
    else:
        print("Adding sample documents to knowledge base...")
        qa_system.add_documents(sample_documents)
    
    # Handle single question
    if args.question:
        result = qa_system.answer_question(args.question)
        print("\n" + "="*50)
        print("QUESTION ANSWERING RESULT")
        print("="*50)
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['best_match_score']:.4f}")
        print("\nRelevant Documents:")
        for i, (doc, score) in enumerate(zip(result['relevant_documents'], result['confidence_scores'])):
            print(f"{i+1}. (Score: {score:.4f}) {doc[:100]}...")
        return
    
    # Interactive mode or demo questions
    if args.interactive:
        print("\n" + "="*50)
        print("INTERACTIVE DPR QUESTION ANSWERING")
        print("="*50)
        print("Enter questions (type 'quit' to exit):")
        
        while True:
            question = input("\nQuestion: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if question:
                result = qa_system.answer_question(question)
                print(f"\nAnswer: {result['answer']}")
                print(f"Confidence: {result['best_match_score']:.4f}")
    else:
        # Demo with sample questions
        sample_questions = [
            "How tall is the Eiffel Tower?",
            "What is Python programming language?",
            "What is machine learning?",
            "How long is the Great Wall of China?",
            "What is artificial intelligence?"
        ]
        
        print("\n" + "="*50)
        print("DEMO: ANSWERING SAMPLE QUESTIONS")
        print("="*50)
        
        for question in sample_questions:
            result = qa_system.answer_question(question)
            print(f"\nQ: {result['question']}")
            print(f"A: {result['answer']}")
            print(f"Confidence: {result['best_match_score']:.4f}")
            print("-" * 30)


if __name__ == "__main__":
    main() 