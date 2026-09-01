"""
Project 3: Agentic RAG - Evaluation Script
Tests router accuracy, tool execution, retrieval, and memory rewriting
"""

import json
import sys
from main import AgenticRAG
from router import route_query


def evaluate_router():
    """Test 1: Does the router correctly classify query types?"""
    print("="*60)
    print("TEST 1: ROUTER ACCURACY")
    print("="*60)

    test_cases = [
        ("What is 25 + 17?", "calculator"),
        ("Calculate 100 divided by 4", "calculator"),
        ("What's today's date?", "datetime"),
        ("What time is it?", "datetime"),
        ("What is Kimi K3?", "retrieve"),          # regression test - used to false-positive
        ("Explain hybrid search", "retrieve"),
        ("What did I learn about chunk sizing?", "retrieve"),
    ]

    correct = 0
    results = []

    for query, expected in test_cases:
        actual = route_query(query)
        is_correct = actual == expected
        correct += is_correct

        status = "✅" if is_correct else "❌"
        print(f"{status} '{query}' → expected: {expected}, got: {actual}")

        results.append({
            "query": query,
            "expected": expected,
            "actual": actual,
            "correct": is_correct
        })

    accuracy = correct / len(test_cases)
    print(f"\nRouter Accuracy: {correct}/{len(test_cases)} ({accuracy*100:.1f}%)\n")

    return {"accuracy": accuracy, "results": results}


def evaluate_end_to_end():
    """Test 2: Full pipeline - does it produce sensible answers?"""
    print("="*60)
    print("TEST 2: END-TO-END PIPELINE")
    print("="*60)

    agent = AgenticRAG()

    test_questions = [
        "What is 25 + 17?",
        "What's today's date?",
        "What is the main topic discussed across these papers?",
        "Can you tell me more about that?",  # tests memory rewriting
    ]

    results = []

    for i, question in enumerate(test_questions, 1):
        print(f"\nQ{i}: {question}")
        print("-" * 60)

        try:
            answer = agent.answer(question)
            print(f"A{i}: {answer[:200]}...")

            results.append({
                "question_number": i,
                "question": question,
                "answer": answer,
                "status": "success"
            })
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            results.append({
                "question_number": i,
                "question": question,
                "status": "error",
                "error": str(e)
            })

    successful = len([r for r in results if r["status"] == "success"])
    success_rate = successful / len(test_questions)

    print(f"\n{'='*60}")
    print(f"End-to-End Success Rate: {successful}/{len(test_questions)} ({success_rate*100:.1f}%)")
    print("="*60 + "\n")

    return {"success_rate": success_rate, "results": results}


def evaluate_web_loading_safety():
    """Test 3: Does the system correctly reject bad web sources?"""
    print("="*60)
    print("TEST 3: WEB LOADING SAFETY (blocklist + thin-content detection)")
    print("="*60)

    from retrieval import HybridRetriever

    # Note: this reuses an already-initialized retriever from evaluate_end_to_end
    # if running standalone, this test is informational only
    test_urls = [
        ("https://www.amazon.in/Clothing-accesories/b?ie", False, "blocklisted domain"),
        ("https://en.wikipedia.org/wiki/Retrieval-augmented_generation", True, "known working site"),
    ]

    print("(Informational - see manual testing log in CASE_STUDY.md for full results)\n")
    for url, expected_success, note in test_urls:
        print(f"  {url}")
        print(f"  Expected: {'load successfully' if expected_success else 'reject safely'} ({note})\n")

    return {"note": "Manually verified - see CASE_STUDY.md for session logs"}


if __name__ == "__main__":
    print("\n🚀 PROJECT 3 EVALUATION\n")

    router_results = evaluate_router()
    e2e_results = evaluate_end_to_end()
    web_safety_results = evaluate_web_loading_safety()

    final_results = {
        "project": "Project 3 - Agentic RAG (Hybrid Search + Memory + Tools)",
        "router_accuracy": router_results["accuracy"],
        "router_test_results": router_results["results"],
        "end_to_end_success_rate": e2e_results["success_rate"],
        "end_to_end_results": e2e_results["results"],
        "web_loading_safety": web_safety_results,
    }

    with open("evaluation-results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    print("✅ Full results saved to evaluation-results.json")
    print(f"✅ Router Accuracy: {router_results['accuracy']*100:.1f}%")
    print(f"✅ End-to-End Success: {e2e_results['success_rate']*100:.1f}%")