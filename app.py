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
PATIENT_SYSTEM_PROMPT = ("假设你是一个病人，你的过去病史是{past_history}，你的现病史是{chief_complaint}，"
                         "你的性别是{gender}，年龄是{age}岁，个人史是{personal_history}，身高是{height}厘米，"
                         "体重是{weight}公斤，体温是{temperature}度，心率是{heart_rate}次/分钟，"
                         "呼吸频次是{respiratory_rate}次/分钟，收缩压是{systolic_blood_pressure}毫米汞柱，"
                         "舒张压是{diastolic_blood_pressure}毫米汞柱。现在你正在一位全科医生面前接受问诊，"
                         "你需要根据医生的问题回答，输出时直接输出对话内容即可，不要输出“患者：”！"
                         "请尽量避免不输出任何东西！请仔细了解病史，不要说你没有哪里不舒服的！"
                         "当你觉得医生的问询应该结束时，请输出[END]！")

def load_data():
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    data = random.choice(dataset)
    patient_system_prompt = PATIENT_SYSTEM_PROMPT.format(
        past_history=data['past_history'],
        chief_complaint=data['chief_complaint'],
        gender=data['gender'],
        age=data['age'],
        personal_history=data['personal history'],
        height=data['height'],
        weight=data['weight'],
        temperature=data['temperature'],
        heart_rate=data['heart_rate'],
        respiratory_rate=data['respiratory_rate'],
        systolic_blood_pressure=data['systolic blood pressure'],
        diastolic_blood_pressure=data['diastolic blood pressure']
    )
    questions = [
        "请从以下科室中选择一个你认为患者最有可能进入的科室," + str(data["department_single_question"]) + ",请仅输出该科室前的大写字母(如A、B、C等)。",
        "请从以下疾病中选择至多三个你认为患者最有可能患有的疾病" + str(data["disease_multi_question"]) + ",请仅输出这些科室前的大写字母(如ABD、BE、CDEF等)。",
        "请输出你对患者记录的现病史",
        "请输出你对患者记录的既往史"
    ]
    return {
        "patient_system_prompt": patient_system_prompt,
        "questions": questions,
        "chief_complaint": data["chief_complaint"],
        "past_history": data["past_history"],
        "index": data['index'],
        "gender": data['gender'],
        "age": data['age'],
        "personal_history": data['personal history'],
        "height": data['height'],
        "weight": data['weight'],
        "temperature": data['temperature'],
        "heart_rate": data['heart_rate'],
        "respiratory_rate": data['respiratory_rate'],
        "systolic_blood_pressure": data['systolic blood pressure'],
        "diastolic_blood_pressure": data['diastolic blood pressure']
    }

def initialize_chat_data(chat_id):
    data = load_data()
    chat_data[chat_id] = {
        "patient_system_prompt": data["patient_system_prompt"],
        "questions": data["questions"],
        "chief_complaint": data["chief_complaint"],
        "past_history": data["past_history"],
        "index": data["index"],
        "gender": data["gender"],
        "age": data["age"],
        "personal_history": data["personal_history"],
        "height": data["height"],
        "weight": data["weight"],
        "temperature": data["temperature"],
        "heart_rate": data["heart_rate"],
        "respiratory_rate": data["respiratory_rate"],
        "systolic_blood_pressure": data["systolic_blood_pressure"],
        "diastolic_blood_pressure": data["diastolic_blood_pressure"],
        "messages": []
    }
    return chat_data[chat_id]

# Initialize pipeline
pipeline = pipeline("text-generation", model="/data/changye/model/PM-14B_10k_comprehensive", model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")

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
        chat_info = initialize_chat_data(chat_id)
    else:
        chat_info = chat_data[chat_id]

    message = [
        {"role": "system", "content": chat_info["patient_system_prompt"]},
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

    chat_data[chat_id]["messages"].append({"role": "user", "content": prompt})
    chat_data[chat_id]["messages"].append({"role": "assistant", "content": response})

    return jsonify({'reply': response, 'messages': chat_data[chat_id]["messages"], 'patient_info': chat_data[chat_id]})

@app.route('/api/questions', methods=['POST'])
def get_questions():
    data = request.get_json()
    chat_id = data.get('chatId')
    messages = data.get('messages')

    app.logger.info(f"Questions endpoint accessed for chat {chat_id}.")

    if chat_id not in chat_data:
        chat_info = initialize_chat_data(chat_id)
    else:
        chat_info = chat_data[chat_id]

    return jsonify({
        "questions": chat_info["questions"],
        "chief_complaint": chat_info["chief_complaint"],
        "past_history": chat_info["past_history"],
        "index": chat_info["index"],
        "gender": chat_info["gender"],
        "age": chat_info["age"],
        "personal_history": chat_info["personal_history"],
        "height": chat_info["height"],
        "weight": chat_info["weight"],
        "temperature": chat_info["temperature"],
        "heart_rate": chat_info["heart_rate"],
        "respiratory_rate": chat_info["respiratory_rate"],
        "systolic_blood_pressure": chat_info["systolic_blood_pressure"],
        "diastolic_blood_pressure": chat_info["diastolic_blood_pressure"]
    })

@app.route('/api/refresh', methods=['POST'])
def refresh_data():
    data = request.get_json()
    chat_id = data.get('chatId')

    app.logger.info(f"Data refresh requested for chat {chat_id}.")

    chat_info = initialize_chat_data(chat_id)

    return jsonify({
        "status": "Data refreshed",
        "questions": chat_info["questions"],
        "chief_complaint": chat_info["chief_complaint"],
        "past_history": chat_info["past_history"],
        "index": chat_info["index"],
        "gender": chat_info["gender"],
        "age": chat_info["age"],
        "personal_history": chat_info["personal_history"],
        "height": chat_info["height"],
        "weight": chat_info["weight"],
        "temperature": chat_info["temperature"],
        "heart_rate": chat_info["heart_rate"],
        "respiratory_rate": chat_info["respiratory_rate"],
        "systolic_blood_pressure": chat_info["systolic_blood_pressure"],
        "diastolic_blood_pressure": chat_info["diastolic_blood_pressure"]
    })

@app.route('/api/clear', methods=['POST'])
def clear_chat_history():
    app.logger.info("Chat history cleared.")
    return jsonify({"status": "Chat history cleared"})

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
