Here's a comprehensive README.md for your project:

```markdown
# Multi-Language Code Plagiarism Detector

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A sophisticated plagiarism detection system that combines vector similarity search with LLM analysis to identify code similarities across multiple programming languages.

## Features

- **Multi-Language Support**: Python, Java, and JavaScript
- **Dual Detection Methods**:
  - Vector similarity search using CodeBERT embeddings
  - LLM semantic analysis (GPT-3.5 Turbo)
- **Advanced Code Normalization**:
  - Identifier normalization
  - Comment/type hint removal
  - Language-specific keyword preservation
- **Web Interface**: Easy-to-use UI for code submission and results visualization
- **Scalable Architecture**: FAISS-based indexing for efficient similarity search

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/code-plagiarism-detector.git
   cd code-plagiarism-detector
   cd vanilla/code-plagiarism-vanilla-02
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment**:
   - Create `.env` file with your OpenAI API key:
     ```env
     OPENAI_API_KEY=your_api_key_here
     ```

4. **Initialize repositories and index**:
   ```bash
   python index.py
   ```

## Usage

1. **Start the server**:
   ```bash
   uvicorn api:app --reload

   or

   python -m uvicorn api:app --reload
   ```

2. **Access the web interface**:
   ```
   http://localhost:8000
   ```

3. **Submit code**:
   - Select programming language
   - Paste code snippet (minimum 3 lines)
   - Choose detection method (recommended: Combined)

4. **Interpret results**:
   - Similarity scores (0-1 scale)
   - Side-by-side code comparisons
   - LLM-powered plagiarism verdict

## Project Structure

```plaintext
code-plagiarism-detector/
├── config/              # Configuration files
│   ├── repos.txt       # Repository URLs to index
│   └── settings.py     # Application settings
├── repositories/       # Cloned code repositories
├── index/              # FAISS index and metadata
├── static/             # CSS stylesheets
├── templates/          # HTML templates
├── api.py              # FastAPI application
├── index.py            # Indexing script
└── requirements.txt    # Dependencies
```

## Configuration

Key settings in `config/settings.py`:
- `SIMILARITY_THRESHOLD`: Minimum similarity score (0.6)
- `TOP_K`: Number of matches to retrieve (10)
- `EMBEDDING_MODEL`: CodeBERT model for embeddings
- `GPT_MODEL`: LLM for semantic analysis
- `LANGUAGE_CONFIG`: Language-specific processing rules

## API Documentation

**POST /api/check**
```json
{
  "code": "your_code_here",
  "language": "python"
}
```

Response:
```json
{
  "plagiarized": boolean,
  "matches": number,
  "top_similarity": float
}
```

## Troubleshooting

Common Issues:
1. **Missing Index Files**:
   - Run `python index.py` before starting server

2. **API Key Errors**:
   - Verify `.env` file contains valid OpenAI API key

3. **Code Processing Issues**:
   - Ensure submitted code meets minimum line requirements
   - Check language configuration matches code syntax

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This tool is designed to assist in code similarity detection. Final plagiarism determinations should always involve human review.
```

```
Unfortunately, I initially misunderstood the task requirements, which is why I spent quite a lot of time creating this version. I thought it would be easy to dockerize from here, but I was wrong... Considering I don't have a complete project, I want to share this with you too... Thanks
```
