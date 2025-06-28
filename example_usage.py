#!/usr/bin/env python3
"""
Example usage of the DPR Question Answering System
"""

from dpr_qa_system import DPRQuestionAnsweringSystem


def main():
    # Initialize the DPR QA system
    print("Initializing DPR Question Answering System...")
    qa_system = DPRQuestionAnsweringSystem()
    
    # Example documents about different topics
    documents = [
        "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics. His work is also known for its influence on the philosophy of science. He is best known to the general public for his mass–energy equivalence formula E = mc², which has been dubbed 'the world's most famous equation'.",
        
        "The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 square kilometers, of which 5,500,000 square kilometers are covered by the rainforest.",
        
        "Bitcoin is a decentralized digital currency, without a central bank or single administrator, that can be sent from user to user on the peer-to-peer bitcoin network without the need for intermediaries. Transactions are verified by network nodes through cryptography and recorded in a public distributed ledger called a blockchain.",
        
        "The human brain contains approximately 86 billion neurons, which communicate with each other through synapses. The brain is responsible for controlling thoughts, memory, emotion, touch, motor skills, vision, breathing, temperature, hunger and every process that regulates our body.",
        
        "Climate change refers to long-term shifts in global or regional climate patterns. Since the mid-20th century, scientists have observed that the primary cause of climate change is human activities, particularly the burning of fossil fuels, which increases heat-trapping greenhouse gas levels in Earth's atmosphere."
    ]
    
    # Add documents to the knowledge base
    print("\nAdding documents to knowledge base...")
    qa_system.add_documents(documents)
    
    # Example questions
    questions = [
        "What is Einstein famous for?",
        "How large is the Amazon rainforest?",
        "What is Bitcoin?",
        "How many neurons are in the human brain?",
        "What causes climate change?"
    ]
    
    # Answer each question
    print("\n" + "="*60)
    print("ANSWERING QUESTIONS USING DPR")
    print("="*60)
    
    for i, question in enumerate(questions, 1):
        print(f"\n{i}. Question: {question}")
        print("-" * 40)
        
        # Get answer
        result = qa_system.answer_question(question, top_k=2)
        
        print(f"Answer: {result['answer']}")
        print(f"Confidence Score: {result['best_match_score']:.4f}")
        
        # Show top relevant documents
        print("Most relevant documents:")
        for j, (doc, score) in enumerate(zip(result['relevant_documents'], result['confidence_scores'])):
            print(f"  {j+1}. (Score: {score:.4f}) {doc[:80]}...")
    
    # Interactive example
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("You can now ask questions about the documents.")
    print("Type 'quit' to exit.")
    
    while True:
        user_question = input("\nYour question: ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if user_question:
            result = qa_system.answer_question(user_question)
            print(f"\nAnswer: {result['answer']}")
            print(f"Confidence: {result['best_match_score']:.4f}")


if __name__ == "__main__":
    main() 