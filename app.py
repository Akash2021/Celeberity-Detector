from flask import Flask,render_template,request
from commons import get_tensor
from infrence import get_flower_name
import torch
import os
app = Flask(__name__)
@app.route('/',methods=['GET','POST'])
def hello_world() :
    if request.method == 'GET':
        return render_template('index.html', value="hello")
    if  request.method == 'POST':
        if 'file' not in request.files :
            print("File Not Uploaded")
            return
        file = request.files['file']
        image=file.read()
        category=get_flower_name(image_bytes=image)
        #print(get_tensor(image_bytes=image))
        #print(tensor.shape)
        return render_template('result.html',flower=category)
if __name__=='__main__':
    app.run(debug=True,port=os.getenv('PORT',5000))
