import json
import torch
from flask import Flask, request, jsonify
from transformers import pipeline
import random
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Constants
TEST_DATA_PATH = "/data/changye/hospital/data/test_data/formal_2_test_dataset.json"
PATIENT_SYSTEM_PROMPT = "假设你是一个病人，你的过去病史是{input1}，你的主诉是{input2},现在你正在一位全科医生面前接受问诊,你需要根据医生的问题回答,输出时直接输出对话内容即可，请尽量避免不输出任何东西！请尽量避免不输出任何东西！请仔细了解病史，不要说你没有哪里不舒服的！"

def load_data():
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    data = random.choice(dataset)
    patient_system_prompt = PATIENT_SYSTEM_PROMPT.format(input1=data['past_history'], input2=data['chief_complaint'])
    questions = [
        "请从以下科室中选择一个你认为患者最有可能进入的科室," + str(data["department_single_question"]) + ",请仅输出该科室前的大写字母(如A、B、C等)。",
        "请从以下疾病中选择至多三个你认为患者最有可能患有的疾病" + str(data["disease_multi_question"]) + ",请仅输出这些科室前的大写字母(如ABD、BE、CDEF等)。",
        "请输出你对患者记录的现病史",
        "请输出你对患者记录的既往史"
    ]
    return patient_system_prompt, questions, data["chief_complaint"], data["past_history"]

# Initialize pipeline
pipeline = pipeline("text-generation", model="/data/changye/model/PM-14B-10k", model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")

# Load initial data
chat_data = {}

@app.route('/')
def index():
    return "Flask server is running."

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    chat_id = data.get('chatId')
    messages = data['messages']
    prompt = data['prompt']

    app.logger.info(f"Received messages for chat {chat_id}: {messages}")
    app.logger.info(f"Received prompt: {prompt}")

    if chat_id not in chat_data:
        patient_system_prompt, questions, chief_complaint, past_history = load_data()
        chat_data[chat_id] = {
            "patient_system_prompt": patient_system_prompt,
            "questions": questions,
            "chief_complaint": chief_complaint,
            "past_history": past_history
        }
    else:
        patient_system_prompt = chat_data[chat_id]["patient_system_prompt"]

    message = [
        {"role": "system", "content": patient_system_prompt},
        {"role": "user", "content": prompt}
    ]
    text = pipeline.tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True
    )
    app.logger.info(f"Generated text for pipeline: {text}")

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("")
    ]
    outputs = pipeline(
        text,
        max_new_tokens=4096,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        pad_token_id=128001,
    )
    response = outputs[0]["generated_text"][len(text):]

    app.logger.info(f"Generated response: {response}")

    messages.append({"role": "user", "content": prompt})
    messages.append({"role": "assistant", "content": response})

    return jsonify({'reply': response, 'messages': messages})

@app.route('/api/questions', methods=['POST'])
def get_questions():
    data = request.get_json()
    chat_id = data.get('chatId')
    messages = data.get('messages')

    app.logger.info(f"Questions endpoint accessed for chat {chat_id}.")

    if chat_id not in chat_data:
        patient_system_prompt, questions, chief_complaint, past_history = load_data()
        chat_data[chat_id] = {
            "patient_system_prompt": patient_system_prompt,
            "questions": questions,
            "chief_complaint": chief_complaint,
            "past_history": past_history
        }
    else:
        questions = chat_data[chat_id]["questions"]
        chief_complaint = chat_data[chat_id]["chief_complaint"]
        past_history = chat_data[chat_id]["past_history"]

    return jsonify({
        "questions": questions,
        "chief_complaint": chief_complaint,
        "past_history": past_history
    })

@app.route('/api/refresh', methods=['POST'])
def refresh_data():
    data = request.get_json()
    chat_id = data.get('chatId')

    app.logger.info(f"Data refresh requested for chat {chat_id}.")

    patient_system_prompt, questions, chief_complaint, past_history = load_data()
    chat_data[chat_id] = {
        "patient_system_prompt": patient_system_prompt,
        "questions": questions,
        "chief_complaint": chief_complaint,
        "past_history": past_history
    }

    return jsonify({
        "status": "Data refreshed",
        "questions": questions,
        "chief_complaint": chief_complaint,
        "past_history": past_history
    })

@app.route('/api/clear', methods=['POST'])
def clear_chat_history():
    app.logger.info("Chat history cleared.")
    return jsonify({"status": "Chat history cleared"})

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
