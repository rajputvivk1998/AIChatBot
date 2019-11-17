from flask import Flask, request, render_template, jsonify
import os
from chatbot import ChatBot
from sentiment import Smileys
from hard_coded import hard_reply
from imagenet import ImageNet
from wiki import get_summary

chatbot = ChatBot(layers=4, maxlen=20, embedding_size=256, batch_size=128, is_train=True, lr=0.001)
smileys = Smileys()

UPLOAD_FOLDER = 'uploads/'


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat',methods= ['POST'])
def chat():
    response = request.form['msg']
    response = response.lower()

    emoji_text = smileys.to_text(response)
    hard_reply_text = hard_reply(response)

    text_split = response.split(" ")

    if hard_reply_text != None:
        reply = hard_reply_text
    elif emoji_text != None:
        reply = emoji_text
    elif text_split[0] == "what" and text_split[1] == "is":
        reply = get_summary(text_split[2])
    else:
        reply = chatbot.infer(response)

    return jsonify({'reply': reply})

@app.route('/image',methods= ['POST'])
def image():
    file_val = request.files['file']
    filename = file_val.filename
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    file_val.save(image_path)

    pred = "It can be a " + ImageNet(image_path)

    return jsonify({'image_path': image_path, 'reply': pred})

if __name__ == '__main__':
    app.run()
