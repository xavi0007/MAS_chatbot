# A foundation model based multi agent system.

Enhanced a standard LLM chatbot by integrating an AI evaluator that assesses the chatbot's responses in real-time and provides suggestions for improvements. This evaluator also identifies potential hallucinations. The chatbot can accept basic user feedback, and if the feedback indicates issues, the LLM may require additional prompts and follow-ups to generate a better response. This approach aims to reduce the impact of the garbage in, garbage out problem. The entire system is deployed on a Flask platform, with the LLM model currently running locally.

## To get started

1. Simply install all requirement packages
```
pip install -r ~/requirements.txt
```
2. Launch the flask web app
```
python PrivateLLM/main.py
```

### Future work
- Distributed and private data training for the chatbot
- Further integration of a RAG system
- Optimization of resources usage
- Progressive Online Fine-tuning for the AI agents
- Multiple Agent working together synchronously
