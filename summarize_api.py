from flask import Flask, request, jsonify
from transformers import pipeline
import time
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

app = Flask(__name__)
# CORS(app)

def setup_server():
    # Put your setup code here, for example:
    print("Flask app starting...")

    global question_answerer
    global question
    global device
    global tokenizer
    global model

    question = "What is the main topic of the abstract?"
    model_name = "deepset/roberta-base-squad2"
    question_answerer = pipeline("question-answering", model=model_name, tokenizer=model_name, device=0)

    model_name = "google/pegasus-cnn_dailymail"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

    
    # setup_server()

@app.before_first_request
def setup():
    # Put your setup code here, for example:
    print("Flask app starting...")
    setup_server()

@app.route('/answer', methods=['GET'])
def answer():
    abstract = request.args.get('abstract')
    start = time.time()
    answer = question_answerer(question=question, context=abstract)
    end = time.time()
    print("Time to answer the pipeline: ", end - start)
    answer = answer['answer']
    print(answer)

    # Summarize
    start = time.time()
    batch = tokenizer(abstract, truncation=True, padding="longest", return_tensors="pt").to(device)
    translated = model.generate(**batch)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    end = time.time()
    print("Time to summarize pegasus: ", end - start)
    print(tgt_text)
    summary = tgt_text[0]

    return jsonify(answer, summary)

if __name__ == '__main__':
    app.run(debug=True)  # run our Flask app
