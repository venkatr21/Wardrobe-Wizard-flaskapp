from flask import Flask, render_template, Response, jsonify, request
import cv2
import requests
from PIL import Image
from script import predict
from evaluate import execute
from pose_parser import pose_parse
cascPath = "./models/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
app = Flask(__name__)
flag = True


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()
        frame1 = frame[48:433, 176:465]
        img = cv2.resize(frame1, (192, 256))
        cv2.imwrite('temp.jpg', img)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        frame1 = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imwrite('face.jpg', frame1[y-10:y+h+10, x:x+w])
        cv2.rectangle(frame, (176, 48), (464, 432), (0, 255, 0), 2)
        return frame


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/feature', methods=['GET', 'POST'])
def feature():
    img1 = open('face.jpg', 'rb').read()
    headers = {'Ocp-Apim-Subscription-Key': "9467429b1aff4c28be9580ab86b684d7",
               'Content-Type': 'application/octet-stream'}
    response = requests.post(
        "https://cv21.cognitiveservices.azure.com/vision/v3.1/analyze?visualFeatures=faces,color", headers=headers, data=img1)
    response.raise_for_status()
    analysis = response.json()
    age = analysis["faces"][0]["age"]
    gender = analysis["faces"][0]["gender"]
    ethnicity = analysis["color"]["dominantColors"][0]
    return render_template('projection.html', age=age, gender=gender, eth=ethnicity)


@app.route('/project', methods=['POST'])
def project():
    render_template('index.html')
    name = request.form["name"]
    selection = request.form["selection"]
    person = Image.open('temp.jpg')
    person.save("./Database/val/person/"+name+".jpg")
    pose_parse(name)
    execute()
    f = open("./Database/val_pairs.txt", "w")
    f.write(name+".jpg "+selection+"_1.jpg")
    f.close()
    predict()
    im = Image.open("./output/second/TOM/val/" + selection + "_1.jpg")
    width, height = im.size
    left = width / 3
    top = 2 * height / 3
    right = width
    bottom = height
    im1 = im.crop((left, top, right, bottom))
    newsize = (600, 450)
    im1 = im1.resize(newsize)
    im1.save("./output/second/TOM/val/" + selection + "_1.jpg")
    result = Image.open("./output/second/TOM/val/" + selection + "_1.jpg")
    result.save("temp.jpg")
    return render_template('projection.html')


def gen(camera):
    timer = 100
    global video_stream
    while timer > 0:
        frame = camera.get_frame()
        frame = cv2.putText(frame, str(timer//10), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 0, 0), 4, cv2.LINE_AA)
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        timer -= 1
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    video_stream.__del__()


def gen_stored():
    img = cv2.imread("temp.jpg")
    ret, jpeg = cv2.imencode('.jpg', img)
    frame = jpeg.tobytes()
    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    global video_stream, flag
    if flag:
        video_stream = VideoCamera()
        flag = False
        return Response(gen(video_stream), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(gen_stored(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
