#%% Classifier - Ron Danon
#%% Import
import glob
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV
from functions import *
import time
import pickle


#%% read data
cars_path = '../data/vehicles/'
non_cars_path = '../data/non-vehicles/'

cars_imgs =  glob.glob(cars_path + '/**/*.png', recursive=True)
non_cars_imgs =  glob.glob(non_cars_path + '/**/*.png', recursive=True)

#%% Etxract features from images
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 1 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 64    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

car_features = extract_features(cars_imgs, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(non_cars_imgs, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
## drop nan lines
nan_ind = np.isnan(car_features).nonzero()
nan_lines = np.unique(nan_ind)
car_features = np.delete(car_features,nan_lines,0)

nan_ind = np.isnan(notcar_features).nonzero()
nan_lines = np.unique(nan_ind)
notcar_features = np.delete(notcar_features,nan_lines,0)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(car_features[0]))

#%% Scale features and prpare traning and testing sets

X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

#%% Train
# Use a linear SVC 
#svc = LinearSVC()

# Use a non-linear SVC
svc = SVC(C=1.0, kernel='rbf', gamma='auto')

# Optimize SVC
#parameters = {'C':[1,2,3,4,5]}
#svr = SVC(kernel = 'linear')
#svc = GridSearchCV(svr, parameters)


# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()

# Save the model
with open('./classifiers/rbfSvcYCrCb2.pkl', 'wb') as fid:
    pickle.dump(svc, fid)    

#with open('./classifiers/rbfSvcYCrCb.pkl', 'rb') as fid:
#    gnb_loaded = pickle.load(fid)