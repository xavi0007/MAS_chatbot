from flask import Flask, request, jsonify, render_template, redirect, session
from runllm import LLM
from utils import calculate_rep
import os

app = Flask(__name__)

users = {
    'id1': '1234',
    'id2': 'password2'
}    
messages = []
llm = LLM()

@app.route('/')
def chat_bot():
    return render_template('chat.html')

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    feedback = data.get('feedback', '')
    # print(feedback)
    # Here you can handle the feedback (e.g., log it, etc.)
    user_score = int(feedback)
    followup_response = llm.follow_up_llm(user_score, messages[-1])
    # followup_response = 'thanks'
    # print(followup_response)
    return jsonify({'followup': followup_response})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    messages.append(message)
    response = llm.infer_llm(message)
    # print(response)
    evaluate_response = llm.evaluate_llm_resp(response)
    
    ## place holder
    # response = get_bot_response(message)
    # evaluate_response = f'{COUNT}'
    
    return jsonify({'response': response, 'evaluation': evaluate_response})

#sample test
def get_bot_response(message):
    responses = {
        "hello": "Hello! How can I help you today?",
        "how are you": "I'm a bot, so I don't have feelings, but thanks for asking!",
        "bye": "Goodbye! Have a great day!",
        'hi': 'bye'
    }
    return responses.get(message.lower(), "I'm sorry, I don't understand that!")



def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    # app.config.from_mapping(
    #     SECRET_KEY='dev',
    #     DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    # )

    # if test_config is None:
    #     # load the instance config, if it exists, when not testing
    #     app.config.from_pyfile('config.py', silent=True)
    # else:
    #     # load the test config if passed in
    #     app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    return app

if __name__ =='__main__':
    app.run(debug=True, host='172.21.47.117', port=5000)