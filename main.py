from flask import Flask, request, jsonify, render_template, redirect, session
from run_llm import LLM

app = Flask(__name__)
users = {
    'id1': '1234',
    'id2': 'password2'
}    
messages = []
llm = LLM()

cur_reputation = 0.5
COUNT = 0

@app.route('/')
def chat_bot():
    return render_template('chat.html')

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    feedback = data.get('feedback', '')
    # Here you can handle the feedback (e.g., log it, store it in a database, etc.)
    print(f'Received feedback: {feedback}')
    if feedback == "good":
        rep = calculate_rep(1)
    else:
        rep = calculate_rep(0)
    followup_response = llm.follow_up_llm(rep, messages[-1])
    # followup_response = 'thanks'
    # print(followup_response)
    return jsonify({'followup': followup_response})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    messages.append(message)
    response = llm.infer_llm(message, 'user')
    evaluate_response = llm.infer_llm(response, 'evaluator')
    
    global COUNT 
    COUNT += 1
    # place holder
    response = get_bot_response(message)
    evaluate_response = f'{COUNT}'
    
    return jsonify({'response': response, 'evaluation': evaluate_response})

def get_bot_response(message):
    responses = {
        "hello": "Hello! How can I help you today?",
        "how are you": "I'm a bot, so I don't have feelings, but thanks for asking!",
        "bye": "Goodbye! Have a great day!",
        'hi': 'bye'
    }
    return responses.get(message.lower(), "I'm sorry, I don't understand that!")

def calculate_rep(feedback):
        numerator = 1
        denominator = 2
        total_count = 5
        #let freshness k = 0.9
        if feedback == 1:
           alpha_i, beta_i = 1, 0 
        else:
            alpha_i, beta_i = 0, 1
        numerator += (alpha_i*0.9**(total_count - COUNT)) + 1
        denominator += ((alpha_i*0.9**(total_count - COUNT)) + (0.9**(total_count - COUNT)*beta_i)) + 2
        cur_reputation = numerator/denominator
        if COUNT > total_count:
            total_count += 5
        print(cur_reputation)
        return cur_reputation

# @app.route('/')
# def view_form():
#     return render_template('login.html')

# @app.route("/test-create", methods=['POST'])
# def test_create():
#     data = request.get_json()
#     return jsonify(data), 201

# @app.route('/handle_post', methods=['POST'])
# def handle_post():
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']
#         print(username, password)
#         if username in users and users[username] == password:
#             return '<h1>Welcome!!!</h1>'
#         else:
#             return '<h1>invalid credentials!</h1>'
#     else:
#         return render_template('login.html')

if __name__ =='__main__':
    app.run(debug=True, host='172.21.47.117', port=5000)