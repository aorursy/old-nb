#from GCForest import gcForest

from skimage import color, io

from scipy.misc import imresize

import os

import glob

import time

import numpy as np # linear algebra

from keras.utils import np_utils

import pandas as pd



import datetime



from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#!usr/bin/env python

"""

Version : 0.1.0

Date : 16th March 2017

Author : Pierre-Yves Lablanche

Email : plablanche@aims.ac.za

Affiliation : African Institute for Mathematical Sciences - South Africa

              Stellenbosch University - South Africa

License : MIT

Status : Under Development

Description :

Python3 implementation of the gcForest algorithm preesented in Zhou and Feng 2017

(paper can be found here : https://arxiv.org/abs/1702.08835 ).

It uses the typical scikit-learn syntax  with a .fit() function for training

and a .predict() function for predictions.

"""

import itertools

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



__author__ = "Pierre-Yves Lablanche"

__email__ = "plablanche@aims.ac.za"

__license__ = "MIT"

__version__ = "0.1.0"

__status__ = "Development"





# noinspection PyUnboundLocalVariable

class gcForest(object):



    def __init__(self, shape_1X, n_mgsRFtree=30, window=None, cascade_test_size=0.2, n_cascadeRF=2,

                 n_cascadeRFtree=101, cascade_layer=np.inf, min_samples=0.05, tolerance=0.0):

        """ gcForest Classifier.

        :param shape_1X: tuple list or np.array

            Shape of a single sample element.

        :param n_mgsRFtree: int (default=30)

            Number of trees in a Random Forest during Multi Grain Scanning.

        :param window: int (default=None)

            List of window sizes to use during Multi Grain Scanning.

            If 'None' no slicing will be done.

        :param cascade_test_size: float or int (default=0.2)

            Split fraction or absolute number for cascade training set splitting.

        :param n_cascadeRF: int (default=2)

            Number of Random Forests in a cascade layer.

            For each pseudo Random Forest a complete Random Forest is created, hence

            the total numbe of Random Forests in a layer will be 2*n_cascadeRF.

        :param n_cascadeRFtree: int (default=101)

            Number of trees in a single Random Forest in a cascade layer.

        :param min_samples: float or int (default=0.1)

            Minimum number of samples in a node to perform a split

            during the training of any Random Forest.

            If int number_of_samples = int.

            If float, min_samples represents the fraction of the initial n_samples to consider.

        :param cascade_layer: int (default=np.inf)

            mMximum number of cascade layers allowed.

            Useful to limit the contruction of the cascade.

        :param tolerance: float (default=0.0)

            Accuracy tolerance for the casacade growth.

            If the improvement in accuracy is not better than the tolerance the construction is

            stopped.

        """



        setattr(self, 'shape_1X', shape_1X)

        setattr(self, 'n_layer', 0)

        setattr(self, '_n_samples', 0)

        setattr(self, 'n_cascadeRF', int(n_cascadeRF))

        if isinstance(window, int):

            setattr(self, 'window', [window])

        elif isinstance(window, list):

            setattr(self, 'window', window)

        setattr(self, 'cascade_test_size', cascade_test_size)

        setattr(self, 'n_mgsRFtree', int(n_mgsRFtree))

        setattr(self, 'n_cascadeRFtree', int(n_cascadeRFtree))

        setattr(self, 'cascade_layer', cascade_layer)

        setattr(self, 'min_samples', min_samples)

        setattr(self, 'tolerance', tolerance)



    def fit(self, X, y):

        """ Training the gcForest on input data X and associated target y.

        :param X: np.array

            Array containing the input samples.

            Must be of shape [n_samples, data] where data is a 1D array.

        :param y: np.array

            1D array containing the target values.

            Must be of shape [n_samples]

        """



        if np.shape(X)[0] != len(y):

            raise ValueError('Sizes of y and X do not match.')

        setattr(self, 'n_layer', 0)

        if not getattr(self, 'window'):

            shape_1X = getattr(self, 'shape_1X')

            setattr(self, 'window', [shape_1X[0]])

        mgs_X = self.mg_scanning(X, y)

        _ = self.cascade_forest(mgs_X, y)



    def predict(self, X):

        """ Predict the class of unknown samples X.

        :param X: np.array

            Array containing the input samples.

            Must be of the same shape [n_samples, data] as the training inputs.

        :return: np.array

            1D array containing the predicted class for each input sample.

        """



        mgs_X = self.mg_scanning(X)

        cascade_all_pred_prob = self.cascade_forest(mgs_X)

        cascade_pred_prob = np.mean(cascade_all_pred_prob, axis=0)

        #predictions = np.argmax(cascade_pred_prob, axis=1)



        return cascade_pred_prob



    def mg_scanning(self, X, y=None):

        """ Performs a Multi Grain Scanning on input data.

        :param X: np.array

            Array containing the input samples.

            Must be of shape [n_samples, data] where data is a 1D array.

        :param y: np.array (default=None)

        :return: np.array

            Array of shape [n_samples, .. ] containing Multi Grain Scanning sliced data.

        """



        setattr(self, '_n_samples', np.shape(X)[0])

        shape_1X = getattr(self, 'shape_1X')

        if len(shape_1X) < 2:

            raise ValueError('shape parameter must be a tuple')



        mgs_pred_prob = []



        for wdw_size in getattr(self, 'window'):

            wdw_pred_prob = self.window_slicing_pred_prob(X, wdw_size, shape_1X, y=y)

            mgs_pred_prob.append(wdw_pred_prob)



        return np.concatenate(mgs_pred_prob, axis=1)



    def window_slicing_pred_prob(self, X, window, shape_1X, y=None):

        """ Performs a window slicing of the input data and send them through Random Forests.

        If target values 'y' are provided sliced data are then used to train the Random Forests.

        :param X: np.array

            Array containing the input samples.

            Must be of shape [n_samples, data] where data is a 1D array.

        :param window: int

            Size of the window to use for slicing.

        :param shape_1X: list or np.array

            Shape of a single sample.

        :param y: np.array (default=None)

            Target values. If 'None' no training is done.

        :return: np.array

            Array of size [n_samples, ..] containing the Random Forest.

            prediction probability for each input sample.

        """



        n_tree = getattr(self, 'n_mgsRFtree')

        min_samples = getattr(self, 'min_samples')



        if shape_1X[1] > 1:

            print('Slicing Images...')

            sliced_X, sliced_y = self._window_slicing_img(X, window, shape_1X, y=y)

        else:

            print('Slicing Sequence...')

            sliced_X, sliced_y = self._window_slicing_sequence(X, window, shape_1X, y=y)



        if y is not None:

            prf = RandomForestClassifier(n_estimators=n_tree, max_features='sqrt',

                                         min_samples_split=min_samples, oob_score=True)

            crf = RandomForestClassifier(n_estimators=n_tree, max_features=None,

                                         min_samples_split=min_samples, oob_score=True)

            print('Training MGS Random Forests...')

            prf.fit(sliced_X, sliced_y)

            crf.fit(sliced_X, sliced_y)

            setattr(self, '_mgsprf_{}'.format(window), prf)

            setattr(self, '_mgscrf_{}'.format(window), crf)

            pred_prob_prf = prf.oob_decision_function_

            pred_prob_crf = crf.oob_decision_function_



        if hasattr(self, '_mgsprf_{}'.format(window)) and y is None:

            prf = getattr(self, '_mgsprf_{}'.format(window))

            crf = getattr(self, '_mgscrf_{}'.format(window))

            pred_prob_prf = prf.predict_proba(sliced_X)

            pred_prob_crf = crf.predict_proba(sliced_X)



        pred_prob = np.c_[pred_prob_prf, pred_prob_crf]



        return pred_prob.reshape([getattr(self, '_n_samples'), -1])



    def _window_slicing_img(self, X, window, shape_1X, y=None):

        """ Slicing procedure for images

        :param X: np.array

            Array containing the input samples.

            Must be of shape [n_samples, data] where data is a 1D array.

        :param window: int

            Size of the window to use for slicing.

        :param shape_1X: list or np.array

            Shape of a single sample.

        :param y: np.array (default=None)

            Target values.

        :return: np.array and np.array

            Arrays containing the sliced images and target values (empty if 'y' is None).

        """



        if any(s < window for s in shape_1X):

            raise ValueError('window must be smaller than both dimensions for an image')



        sliced_imgs = []

        sliced_target = []

        refs = np.arange(0, window * shape_1X[1], shape_1X[0])



        iterx = list(range(shape_1X[0] - window + 1))

        itery = list(range(shape_1X[1] - window + 1))



        for img, ix, iy in itertools.product(enumerate(X), iterx, itery):

            rind = refs + ix + shape_1X[0] * iy

            sliced_imgs.append(np.ravel([img[1][i:i + window] for i in rind]))

            if y is not None:

                sliced_target.append(y[img[0]])



        return np.asarray(sliced_imgs), np.asarray(sliced_target)



    def _window_slicing_sequence(self, X, window, shape_1X, y=None):

        """ Slicing procedure for sequences (aka shape_1X = [.., 1]).

        :param X: np.array

            Array containing the input samples.

            Must be of shape [n_samples, data] where data is a 1D array.

        :param window: int

            Size of the window to use for slicing.

        :param shape_1X: list or np.array

            Shape of a single sample.

        :param y: np.array (default=None)

            Target values.

        :return: np.array and np.array

            Arrays containing the sliced sequences and target values (empty if 'y' is None).

        """

        if shape_1X[0] < window:

            raise ValueError('window must be smaller than the sequence dimension')



        sliced_sqce = []

        sliced_target = []



        for sqce in enumerate(X):

            slice_sqce = [sqce[1][i:i + window] for i in np.arange(shape_1X[0] - window + 1)]

            sliced_sqce.append(slice_sqce)

            if y is not None:

                sliced_target.append(np.repeat(y[sqce[0]], shape_1X[0] - window + 1))



        return np.reshape(sliced_sqce, [-1, window]), np.ravel(sliced_target)



    def cascade_forest(self, X, y=None):

        """ Perform (or train if 'y' is not None) a cascade forest estimator.

        :param X: np.array

            Array containing the input samples.

            Must be of shape [n_samples, data] where data is a 1D array.

        :param y: np.array (default=None)

            Target values. If 'None' perform training.

        :return: np.array

            1D array containing the predicted class for each input sample.

        """



        if y is not None:

            test_size = getattr(self, 'cascade_test_size')

            max_layers = getattr(self, 'cascade_layer')

            tol = getattr(self, 'tolerance')



            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)



            self.n_layer += 1

            prf_crf_pred_ref = self._cascade_layer(X_train, y_train)

            accuracy_ref = self._cascade_evaluation(X_test, y_test)

            feat_arr = self._create_feat_arr(X_train, prf_crf_pred_ref)



            self.n_layer += 1

            prf_crf_pred_layer = self._cascade_layer(feat_arr, y_train)

            accuracy_layer = self._cascade_evaluation(X_test, y_test)



            while accuracy_layer > (accuracy_ref + tol) and self.n_layer <= max_layers:

                accuracy_ref = accuracy_layer

                prf_crf_pred_ref = prf_crf_pred_layer

                feat_arr = self._create_feat_arr(X_train, prf_crf_pred_ref)

                self.n_layer += 1

                prf_crf_pred_layer = self._cascade_layer(feat_arr, y_train)

                accuracy_layer = self._cascade_evaluation(X_test, y_test)



        elif y is None:

            at_layer = 1

            prf_crf_pred_ref = self._cascade_layer(X, layer=at_layer)

            while at_layer < getattr(self, 'n_layer'):

                at_layer += 1

                feat_arr = self._create_feat_arr(X, prf_crf_pred_ref)

                prf_crf_pred_ref = self._cascade_layer(feat_arr, layer=at_layer)



        return prf_crf_pred_ref



    def _cascade_layer(self, X, y=None, cv=3, layer=0):

        """ Cascade layer containing Random Forest estimators.

        If y is not None the layer is trained.

        :param X: np.array

            Array containing the input samples.

            Must be of shape [n_samples, data] where data is a 1D array.

        :param y: np.array (default=None)

            Target values. If 'None' perform training.

        :param cv: int (default=3)

            Number of split for k-fold cross-validation.

        :param layer: int (default=0)

            Layer indice. Used to call the previously trained layer.

        :return: list

            List containing the prediction probabilities for all samples.

        """



        n_tree = getattr(self, 'n_cascadeRFtree')

        n_cascadeRF = getattr(self, 'n_cascadeRF')

        min_samples = getattr(self, 'min_samples')



        prf = RandomForestClassifier(n_estimators=n_tree, max_features='sqrt',

                                     min_samples_split=min_samples, oob_score=True)

        crf = RandomForestClassifier(n_estimators=n_tree, max_features=None,

                                     min_samples_split=min_samples, oob_score=True)



        prf_crf_pred = []

        if y is not None:

            print('Adding/Training Layer, n_layer={}'.format(self.n_layer))

            for irf in range(n_cascadeRF):

                prf.fit(X, y)

                crf.fit(X, y)

                setattr(self, '_casprf{}_{}'.format(self.n_layer, irf), prf)

                setattr(self, '_cascrf{}_{}'.format(self.n_layer, irf), crf)

                prf_crf_pred.append(prf.oob_decision_function_)

                prf_crf_pred.append(crf.oob_decision_function_)

        elif y is None:

            for irf in range(n_cascadeRF):

                prf = getattr(self, '_casprf{}_{}'.format(layer, irf))

                crf = getattr(self, '_cascrf{}_{}'.format(layer, irf))

                prf_crf_pred.append(prf.predict_proba(X))

                prf_crf_pred.append(crf.predict_proba(X))



        return prf_crf_pred



    def _cascade_evaluation(self, X_test, y_test):

        """ Evaluate the accuracy of the cascade using X and y.

        :param X_test: np.array

            Array containing the test input samples.

            Must be of the same shape as training data.

        :param y_test: np.array

            Test target values.

        :return: float

            the cascade accuracy.

        """



        casc_pred_prob = np.mean(self.cascade_forest(X_test), axis=0)

        casc_pred = np.argmax(casc_pred_prob, axis=1)

        casc_accuracy = accuracy_score(y_true=y_test, y_pred=casc_pred)

        print('Layer validation accuracy = {}'.format(casc_accuracy))



        return casc_accuracy



    def _create_feat_arr(self, X, prf_crf_pred):

        """ Concatenate the original feature vector with the predicition probabilities

        of a cascade layer.

        :param X: np.array

            Array containing the input samples.

            Must be of shape [n_samples, data] where data is a 1D array.

        :param prf_crf_pred: list

            Prediction probabilities by a cascade layer for X.

        :return: np.array

            Concatenation of X and the predicted probabilities.

            To be used for the next layer in a cascade forest.

        """



        swap_pred = np.swapaxes(prf_crf_pred, 0, 1)

        add_feat = swap_pred.reshape([np.shape(X)[0], -1])

        feat_arr = np.concatenate([add_feat, X], axis=1)



        return feat_arr
img_size_glob = 128



def get_im_cv2(path):

    img = io.imread(path)

    resized = imresize(img, (img_size_glob, img_size_glob, 3))

    return resized



def load_train():

    X_train = []

    X_train_id = []

    y_train = []

    start_time = time.time()



    print('Read train images')

    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

    for fld in folders:

        index = folders.index(fld)

        print('Load folder {} (Index: {})'.format(fld, index))

        path = os.path.join('..', 'Fishery','input', 'train_sample', fld, '*.jpg')

        files = glob.glob(path)   

        for fl in files:

            flbase = os.path.basename(fl)

            x = get_im_cv2(fl)

            X_train.append(x)

            X_train_id.append(flbase)

            y_train.append(index)



    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))

    return X_train, y_train, X_train_id





def load_test():

    path = os.path.join('..', 'Fishery','input', 'test', 'test_stg1', '*.jpg')

    files = sorted(glob.glob(path))



    X_test = []

    X_test_id = []

    for fl in files:

        flbase = os.path.basename(fl)

        img = get_im_cv2(fl)

        X_test.append(img)

        X_test_id.append(flbase)



    return X_test, X_test_id



def read_and_normalize_train_data():

    train_data, train_target, train_id = load_train()



    print('Convert to numpy...')

    train_data = np.array(train_data, dtype=np.uint8)

    train_target = np.array(train_target, dtype=np.uint8)

    #print('Current Shape: ', train_data.shape)

    print('Reshape...')

    #train_data = train_data.flatten('F')

    train_data = train_data.reshape(train_data.shape[0], -1)

    #print('Updated Shape: ', train_data.shape)

    print('Convert to float...')

    train_data = train_data.astype('float32')

    train_X = train_data / 255

    #train_target = np_utils.to_categorical(train_target, 8)



    return train_X, train_target, train_id





def read_and_normalize_test_data():

    start_time = time.time()

    test_data, test_id = load_test()



    test_data = np.array(test_data, dtype=np.uint8)

    test_data = test_data.reshape(test_data.shape[0], -1)



    test_data = test_data.astype('float32')

    test_data = test_data / 255



    print('Test shape:', test_data.shape)

    print(test_data.shape[0], 'test samples')

    print('test shape', test_data.shape)

    

    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))

    return test_data, test_id



def create_submission(predictions, test_id, info):

    result1 = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])

    result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)

    now = datetime.datetime.now()

    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'

    result1.to_csv(sub_file, index=False)
train_data, train_target, train_id = read_and_normalize_train_data()
#change windows size depending on your ram and core/thread 

gcf = gcForest(shape_1X=[128,128], window=[128,128], tolerance=0.0, min_samples=7)

gcf.fit(train_data, train_target)
test_data, test_id = read_and_normalize_test_data()
pred_X = gcf.predict(test_data)
clip = 0.02

preds = np.clip(pred_X, clip, 1-clip)
create_submission(preds, test_id, 'gc_forest')