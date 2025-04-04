import httpx
import csv
import os
from datetime import datetime
import asyncio
from config.settings import SIMILARITY_THRESHOLD

async def test_plagiarism(method: str, code: str, language: str = "python"):
    """Test plagiarism detection using different methods"""
    async with httpx.AsyncClient() as client:
        if method == "rag":
            # Direct vector DB query
            embed_resp = await client.post(
                "http://embedding-service:8001/embed",
                json={"text": code, "language": language}
            )
            embedding = embed_resp.json()["embedding"]
            
            search_resp = await client.post(
                "http://vector-db:8002/search",
                json={"embedding": embedding, "k": 5}
            )
            matches = search_resp.json().get("matches", [])
            return any(m['similarity'] >= SIMILARITY_THRESHOLD for m in matches)
            
        elif method == "llm":
            # Direct LLM analysis (simplified)
            return False  # Implement proper LLM call
            
        else:  # combined
            resp = await client.post(
                "http://frontend:8000/check",
                data={"code": code, "language": language}
            )
            return "Potential Plagiarism Detected" in resp.text

async def run_evaluation():
    """Run evaluation on test cases"""
    test_cases = [
        # Exact copies (should detect plagiarism)
        {
            "code": "def add(a, b):\n    return a + b",
            "language": "python",
            "expected": True,
            "description": "Simple function - exact copy"
        },
        
        # Renamed variables (should detect)
        {
            "code": "def sum(x, y):\n    return x + y",
            "language": "python", 
            "expected": True,
            "description": "Renamed variables"
        },
        
        # Different logic (should not detect)
        {
            "code": "def multiply(a, b):\n    return a * b",
            "language": "python",
            "expected": False,
            "description": "Different functionality"
        },
        
        # Obfuscated example (should detect)
        {
            "code": "def calculate(a, b):\n    result = a + b\n    return result",
            "language": "python",
            "expected": True,
            "description": "Rewritten but same logic"
        },
        
        # Empty code
        {
            "code": "",
            "language": "python",
            "expected": False,
            "description": "Empty input"
        }
    ]   
    
    methods = ["rag", "llm", "combined"]
    results = []
    
    for case in test_cases:
        for method in methods:
            try:
                actual = await test_plagiarism(method, case["code"], case["language"])
                results.append({
                    "method": method,
                    "code_snippet": case["description"],
                    "expected": case["expected"],
                    "actual": actual,
                    "correct": case["expected"] == actual,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                print(f"Error testing case: {str(e)}")
                continue
    
    # Save results
    os.makedirs("/results", exist_ok=True)
    with open("/results/evaluation.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Evaluation complete. Saved {len(results)} results.")

if __name__ == "__main__":
    asyncio.run(run_evaluation())