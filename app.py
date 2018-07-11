from flask import Flask,jsonify,url_for,redirect,abort,make_response,request,render_template
import os
from werkzeug import secure_filename
from fungalProc import predict

UPLOAD_FOLDER = os.path.join(os.getcwd(),'uploads')
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS



@app.route('/', methods=['GET', 'POST' ])
@app.route('/index', methods=['GET', 'POST' ])
def index():
   return render_template("index.html")
   
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['image']
		
        if file :
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)

            p = predict(img_path)
            print(p)
            return p

if __name__ == '__main__':
    app.run(debug=True,host='127.0.0.1')
