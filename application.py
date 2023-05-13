import speech_recognition as sr
from pydub import AudioSegment

from flask import Flask, render_template, request, redirect

application = Flask(__name__)


@application.route("/", methods=["GET", "POST"])
def index():
    transcript = ""
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "":
            return redirect(request.url)
        if file:
            recognizer = sr.Recognizer()
            audioFile = sr.AudioFile(file)
            with audioFile as source:
                data = recognizer.record(source)
            transcript = recognizer.recognize_google(data)

    return render_template('index.html', transcript=transcript)


@application.route('/recognition/audio/mp3', methods=['POST'])
def recognise_mp3():
    if request.method == 'POST':
        f = request.files['file']

    x = AudioSegment.from_file(f.stream)
    file = x.export(format="wav")

    mp3 = sr.AudioFile(file)

    r = sr.Recognizer()

    with mp3 as source:
        audio = r.record(source)
    rec = r.recognize_google(audio)
    return rec


if __name__ == "__main__":
    application.run()