# Contributing Guidelines

Thank you for your interest in contributing!

This project is structured to be modular and easy to extend.  
Contributions of all kinds are welcome — bug fixes, tests, documentation improvements, new features, or design discussions.

---

## Development Setup

1. Clone the repository
2. Create a Python virtual environment
3. Install dependencies using `pip install -r requirements.txt`
4. Configure environment variables in `app/.env`
5. Run the API locally using `uvicorn app.main:app --reload`

---

## Contribution Types

### Code Improvements
- Add or improve services under `app/services/`
- Enhance chunking, embedding, reranking, or generation modules
- Improve search accuracy or performance

### Documentation
- Update `README.md`
- Improve architecture diagrams or explanations in `docs/`

### Tests
- Add unit or integration tests under `tests/`

---

## Pull Requests

Please include:

- Clear description of changes  
- Associated issue or motivation  
- Tests if applicable  
- Documentation updates if needed  

Thank you for helping improve this project!