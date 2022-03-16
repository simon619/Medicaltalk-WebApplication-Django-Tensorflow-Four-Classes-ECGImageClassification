from django.shortcuts import render, redirect
#from tensorflow.python.keras.preprocessing.image import img_to_array, load_img
from .forms import ImageUploadForm
from .models import EcgImageDataBase
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import cv2
from django.core.files.storage import FileSystemStorage, default_storage
import numpy as np
from django.contrib.auth.decorators import login_required

model = tf.keras.models.load_model('tensor\ECGConvNet.h5')

CATEGORIES = ['You Report Is Abnormal', 'You Have A Myocardial Infarction History', 'Myocardial Infarction Detected', 'Don\'t Worry You Are Fine']
@login_required
def classfication(request):
	if request.method == "POST":
		form = ImageUploadForm(request.POST, request.FILES)
		if form.is_valid():
			form.save()
			file = request.FILES["image"]
			file_name = default_storage.save(file.name, file)
			file_url = default_storage.path(file_name)
			predictions = model.predict([prepare(file_url)])
			#x = CATEGORIES[int(prediction[0][0])]
			arr = np.array(predictions)
			max_index_row = np.argmax(arr, axis=1)
			#print(max_index_row[0])
			#print("Probability")
			y = arr[0, max_index_row]*100			
			x = CATEGORIES[max_index_row[0]]

			
		return render(request=request, template_name="tensor/ecg.html", context={'form':form, 'x' : x, 'y' : y})
	form = ImageUploadForm()
    
	# movies = Movies.objects.image
	return render(request=request, template_name="tensor/ecg.html", context={'form':form})

def prepare(filepath):
    IMG_SIZE = 200 
    img_array = cv2.imread(filepath)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)