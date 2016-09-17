import json
import numpy as np
from keras.models import Sequential
from keras.models import model_from_json
from keras.utils import np_utils
from keras.optimizers import SGD

class Predictor:
    """An object for evaluating ages and genders
    Will eventually implement age predictor, gender predictor,
    and combined predictors as subclasses"""
    def __init__(self, json_model='res/model.json'):
        self.gender_loaded = False
        self.age_loaded = False
        self.combined_loaded = False
        self.Model = self.load_model(json_model)

    def load_model(self, json_model):
        json_file = open(json_model, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        return loaded_model

    def load_weights(self, weights='res/gender_weights.hdf5'):
        #load weights into new model
        self.Model.load_weights(weights)
        # evaluate loaded model on test data
        #i had to redefine sgd since the model has to be compiled again with the weights
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.Model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        self.weights_loaded = True

    def score(self, img):
        #uses X_test and Y_test
        score = loaded_model.evaluate(X_test, Y_test, verbose = 0)

        print('Test score:', score[0])
        print('Test accuracy:', score[1]*100)

    def get_gender(self, img):
        if (img.shape != (112, 112)):
            #raise warning, resize image
            p = 'p'
        img = img.reshape(1,1,112,112)
        res = np.rint(self.Model.predict_on_batch(img))
        if (( res[0] == [1., 0] ).all()):
            return 'Male'
        elif (( res[0] == [0., 1.] ).all()):
            return 'Female'
        else:
            return "Unknown, probs too ugly"




