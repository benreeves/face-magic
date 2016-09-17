#Face Magic

A small web application built with Flask for showcasing and experimenting with convolutional neural networks for age/gender classification and facial feature analysis. The web project contains everything needed once a keras model is trained and the model has been saved off for later usage. The trainer directory contains some of the code we used to create and train the models using convolutional neural networks in keras.

Dependencies:

* [dlib] (http://dlib.net/compile.html) with python bindings 
* [keras] (https://keras.io/) with Theano 
* Flask

To get this up and running, you will need to compile or install dlib with [boost python] (http://www.boost.org/doc/libs/1_60_0/libs/python/doc/html/building/installing_boost_python_on_your_.html). Once that is complete, you should be able to install keras through your python package manager such as pip3 install keras. You'll need to download trained facial feature detector for use in dlib if you want to overlay images with facial feature landmarks and save it in the res/ directory.  The one I use is [here] (http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).At the moment, a trained age/gender predictor model is not available on source control.

Set up flask and your venv, and start flask! 
![Alt text](/sample.png?raw=true "Example Result")
