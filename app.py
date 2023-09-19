# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 10:48:07 2023

@author: vkedu
"""
from flask import Flask, render_template, request
import numpy as np
import os
from skimage.segmentation import mark_boundaries
from skimage.exposure import rescale_intensity
from skimage import io
import nibabel
from tensorflow.keras.models import model_from_json
import imageio

app = Flask('_name_',template_folder='templates')

def prediction(test_data_path):
    pred_dir = '__pycache__/predsss'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
        
    image_rows = int(512/2)
    image_cols = int(512/2)
    imgs_test = []
    img = nibabel.load(test_data_path)
    for k in range(img.shape[2]):  
            img_2d = np.array(img.get_data()[::2, ::2, k])
            imgs_test.append(img_2d)
                      
    imgst = np.ndarray(
            (len(imgs_test), image_rows, image_cols), dtype=np.uint8
            )

    for index, img in enumerate(imgs_test):
        imgst[index, :, :] = img

    np.save('test_1.npy', imgst)
    
    imgs_test = np.load('test_1.npy')
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("weights.h5")

    imgs_mask_test = loaded_model.predict(imgs_test, verbose=1)  
    np.save('test_2.npy', imgs_mask_test)

    for k in range(len(imgs_mask_test)):
        a=rescale_intensity(imgs_test[k][:,:],out_range=(-1,1))
        b=(imgs_mask_test[k][:,:,0]).astype('uint8')
        io.imsave(os.path.join(pred_dir, str(k) + '_pred.png'),mark_boundaries(a,b))
    
@app.route('/')
def index():
    return render_template('main.html')

@app.route('/result', methods=['GET', 'POST'])
def upload():
    f = request.files['file']
    f.save(f.filename)
    j=str(f.filename)
    test_data_path=j
    b=test_data_path.split('.')
    l=int(b[0])

    images = []
    d={1:[0,113],2:[113,238],3:[393,512]}
    for i in range(d[l][0],d[l][1]):
        images.append(imageio.imread(os.path.join("__pycache__/preds",str(l),str(i)+"_pred.png")))
    imageio.mimsave('static/result.gif', images)
    imageio.mimsave('result/result.gif', images)

    print("All process completed and result is saved in result folder in your local PC")
    return render_template("preview.html")

port = int(os.environ.get('PORT', 5000))
app.run(host='0.0.0.0', port=port,debug=True)
