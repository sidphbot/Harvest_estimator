# Harvest_estimator
A Django based web project that uses custom YOLO model, trained on google open images, to detect and count number of fruits(Apple, Mango, Orange, Pomegranate, tomato) in an image.




# Requirements:

> python >3.4 with pip

	make sure the PATH is correctly modified for pip to work,

		e.g. C:\Python34\;C:\Python34\Scripts;

> setuptools

	pip install setuptools

> opencv 3.4

	pip install opencv-python

> django

	pip install django


# Setting up the project

- grab a copy of the project - unzip it	

- go to project root fruitDetect

- run 
	python manage.py runserver


The Project should be live and accessible at 
	http://localhost:8000/scan



# Project Structure

> submodule parseImage handles the scanning 

> submodule fruitDetect is teh main application and has teh settings.py for Django project.

> static directory contains the images which are generated after scanning

> yolo-fruits is the custom trained model to detect below fruits:

	Mango
	Apple
	Orange
	Pomegranate
	Tomato

> yolo-coco(not being used) is the pretrained yolo model that detects ovber 80 classes folow teh below url to know more. 

	to perform a generic object detection change the yolo-scan function in views.py in parseImage submodule. 


yolo-structure

	.names file - contains serialised names of classes trained on
	.cfg file - contains the yolo configuration that determine the whole model, its layers and other details. !! Please change only if you know what you are doing !!
	.weights file - contain the weights that are achieved by training Yolo of a set of classes mentioned in obj.names (this file is huge and has to be trained by yourself )
  
  

