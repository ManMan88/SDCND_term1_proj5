#%% Cars Finder - Ron Danon
#%% Import
import glob
import numpy as np
from moviepy.editor import VideoFileClip
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from pipeline_functions import slide_window, search_windows, add_heat, apply_threshold, draw_labeled_bboxes
from scipy.ndimage.measurements import label
import pickle

#%% Load classifier and set parameters
with open('./classifiers/rbfSvcYCrCb2.pkl', 'rb') as fid:
    clf = pickle.load(fid)
with open('./classifiers/rbfSvcYCrCb2SCALAR.pkl', 'rb') as fid:
    X_scalar = pickle.load(fid)
#with open('./classifiers/linearSvcYCrCb2.pkl', 'rb') as fid:
#    clf = pickle.load(fid)
#with open('./classifiers/linearSvcYCrCb2SCALAR.pkl', 'rb') as fid:
#    X_scalar = pickle.load(fid)

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
y_start = 350 # Min in y to search in slide_window()

#%% determine windows

# read a test image to set the windows
images = glob.glob('./test_images/*.jpg')
image = mpimg.imread(images[0])

# define the windows overlap
overlap = 0.75
# set different sizes of windows
windows1 = slide_window(image, x_start_stop=[300, 1000], y_start_stop=[y_start,500], 
                    xy_window=(64, 64), xy_overlap=(overlap, overlap))
windows2 = slide_window(image, x_start_stop=[None, None], y_start_stop=[y_start,550], 
                    xy_window=(96, 96), xy_overlap=(overlap, overlap))
windows3 = slide_window(image, x_start_stop=[None, None], y_start_stop=[y_start,600], 
                    xy_window=(128, 128), xy_overlap=(overlap, overlap))
windows4 = slide_window(image, x_start_stop=[None, None], y_start_stop=[y_start,None], 
                    xy_window=(160, 160), xy_overlap=(overlap, overlap))
windows5 = slide_window(image, x_start_stop=[None, None], y_start_stop=[y_start,None], 
                    xy_window=(256, 256), xy_overlap=(overlap, overlap))

windows = np.concatenate((windows1,windows2,windows3,windows4))#,windows5))
#draw_image = np.copy(image)
#test = draw_boxes(draw_image,windows5)
#plt.imshow(test)

# initiate hot windows buffer
hot_windows_buff = []
# define the buffer size
buffer_size = 5
# define heatmap threshold
threshold = 5
#%% Find cars
def find_cars(image):
    image = (image/255).astype(np.float32)  
    draw_image = np.copy(image)
    
    hot_windows = search_windows(image, windows, clf, X_scalar, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    
    hot_windows_buff.append(hot_windows)
    if len(hot_windows_buff) is buffer_size+1:
        hot_windows_buff.pop(0)
    heatmap = np.zeros_like(image)
    for bboxes in hot_windows_buff:
        heatmap = add_heat(heatmap,bboxes)
    heatmap = apply_threshold(heatmap, threshold)
    labels = label(heatmap)
    window_image = draw_labeled_bboxes(draw_image, labels, color=(0, 1.0, 0), thick=6)   
#    print(np.max(heatmap))
    return (window_image*255).astype(np.int)


#%% Find cars in the project video
vid_output = './output_videos/project_video6.mp4'
clip1 = VideoFileClip('./project_video.mp4')
proj_clip = clip1.fl_image(find_cars)
proj_clip.write_videofile(vid_output, audio=False)