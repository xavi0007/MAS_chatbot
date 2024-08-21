# A foundation model based multi agent system.

On top of a normal LLM chatbot, I added an AI evaluator to evaluate the chatbot's response on the fly and offer 'advice' for improvements. 
The evaluator should also detect any possible hallucinations. The LLM chatbot is also capable of recieving simple feedback from user. If feedback is bad, perhaps LLM requires
more prompting and follow ups from the user to output better response. This hopes to mitigate some parts of the garbage in, garbage out problem. 

The entire system is deployed on a flask system. Currently, the LLM model is ran locally. 

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
- Distributed and prviate data training for the chatbot
- Futher integration of a RAG system
- Optimization of resources usage
- Progressive Online Fine-tuning for the AI agents
- Multiple Agent working together synchronously
