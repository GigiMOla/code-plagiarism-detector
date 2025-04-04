### **6. evaluation/README.md**
```markdown
# Evaluation Service

Automated testing framework for plagiarism detection.

## Test Cases 🧪
1. Exact code copies
2. Renamed variables
3. Logic restructuring
4. Empty input
5. Obfuscated code

## Metrics Tracked 📊
- True/False positives
- Method comparison (RAG vs LLM vs Combined)
- Similarity score distribution
- Error rates

## Running Tests
```bash
docker-compose run evaluation
Results Output 📁
CSV format with columns:

csv

method,code_snippet,expected,actual,correct,timestamp
Custom Test Cases
Add new cases to test_cases/ directory in format:

python

{
  "code": "function test() {}", 
  "language": "javascript",
  "expected": True,
  "description": "New test case"
}