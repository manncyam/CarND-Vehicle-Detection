from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from scipy.ndimage.measurements import label
from p5constants import *
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


X_scaler_filename = "x_scaler.pkl"
svc_filename = "svc.pkl"

'''
Use hog to get features and return it
'''
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # return features along with hog image if vis is true
    if vis==True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=True, feature_vector=feature_vec)
        return features, hog_image
    # otherwise return only features
    else:
        features = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=False, feature_vector=feature_vec)
        return features

'''
Resize img and return one long vector
'''
def bin_spatial(img, size=(32, 32)):
    return cv2.resize(img, size).ravel()

'''Create histogram of each channel, concatenate into hist_features and return the hist_feature'''
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
   
    return hist_features

'''Use opencv to convert color to YCrCb or LUV'''
def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

'''
    Define a function to extract features from a list of images
    Have this function call bin_spatial() and color_hist()
'''
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = cv2.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            feature_image = convert_color(image, color_space)
        else:
            feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                  
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

'''Read full path to cars and notcars images return lists of cars and notcars'''
def get_cars_and_notcars(image_path):
    import glob
    images = glob.glob(image_path)
    cars = []
    notcars = []
    for image in images:
        if 'image' in image or 'extra' in image:
            notcars.append(image)
        else:
            cars.append(image)
    return cars, notcars

'''Train LinearSVC model and save the model''' 
def train_model(image_path):
    # Read all cars and notcars paths
    cars, notcars = get_cars_and_notcars(image_path)
    '''get car features'''
    car_features = extract_features(cars, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=hog_orientation, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
    '''get not car features'''
    notcar_features = extract_features(notcars, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=hog_orientation, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
    '''stack car_features on top of notcar_features'''
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    '''Use StandardScaler to create a X_scaler'''
    X_scaler = StandardScaler().fit(X)
    '''Use X_scaler scaler to scale X'''
    scaled_X = X_scaler.transform(X)
    '''Create label y, 1 for car and 0 for notcar'''
    print (len(car_features))
    print (len(notcar_features))
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    print("start training")
    '''Split up data into randomized training and test sets'''
    rand_state = np.random.randint(0,100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y,
                                                       test_size=0.2,
                                                       random_state=rand_state)
    '''Create LearnSVC instance for training'''
    svc = LinearSVC()
    svc.fit(X_train, y_train)

    print ("Done with training")
    print ("Saving X_scaler and svc model")
    joblib.dump(X_scaler, X_scaler_filename)
    joblib.dump(svc, svc_filename)
    score = svc.score(X_test, y_test)
    print ("test score {:.3f}".format(score))
    
'''Read svc model and X_scaler from disk'''
def load_model():
    X_scaler = joblib.load(X_scaler_filename)
    svc = joblib.load(svc_filename)
    
    return X_scaler, svc

'''Find car in an img and return a list of bounding boxes'''
def find_cars(img, ystart, ystop, scale, svc, X_scaler):
    img_tosearch = img[ystart:ystop,:,:]
    if color_space != 'BGR':
        ctrans_tosearch = convert_color(img_tosearch, conv=color_space)
    else:
        ctrans_tosearch = img_tosearch
    # Resize the image if the scale is not 1
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch,
                                     (imshape[1]//scale, imshape[0]//scale))
    # Store each channel in ch1, ch2, ch3
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks in x and y direction
    nxblocks = (ch1.shape[1] // pix_per_cell) - 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - 1

    # Calculate n steps to move in x and y direction
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, hog_orientation, pix_per_cell, cell_per_block,
                           feature_vec=False)
    hog2 = get_hog_features(ch2, hog_orientation, pix_per_cell, cell_per_block,
                           feature_vec=False)
    hog3 = get_hog_features(ch3, hog_orientation, pix_per_cell, cell_per_block,
                           feature_vec=False)
    # To store bounding boxes of cars if found
    box_list = []
    
    for xb in range(nxsteps+1):
        for yb in range(nysteps+1):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
            '''
            subimg = ctrans_tosearch[ytop:ytop+window, xleft:xleft+window]
                        
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            m = np.hstack((spatial_features, hist_features, hog_features)).reshape(1,-1)
            scaled_m = X_scaler.transform(m)
            '''
            m = hog_features.reshape(1,-1)
            scaled_m = X_scaler.transform(m)
            
            prediction = svc.predict(scaled_m)
            
            if prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                box_list.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
            
    return box_list
    
'''Draw bounding box on the image return the image'''
def draw_boundingboxes(img, boxes):
    for box in boxes:
        cv2.rectangle(img, box[0], box[1], (0,0,255), 6)
    return img

'''Add 1 to pixels within bounding boxes return heatmap'''
def add_heat(heatmap, bboxes_n_frames):
    for bboxes in bboxes_n_frames:
        for bbox in bboxes:
            heatmap[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] += 1

    return heatmap

'''Set heatmap values to 0 if it is less than threshold'''
def apply_threshold(heatmap, threshold):
    heatmap[heatmap < threshold] = 0
    return heatmap

'''Create label'''
def get_labels(image, bboxes_n_frames, bboxes_list, counter, counter_thres):
    if counter < counter_thres:
        bboxes_n_frames.append(bboxes_list)
    else:
        indx = counter % counter_thres
        bboxes_n_frames[indx] = bboxes_list

    counter += 1
    heat = np.zeros_like(image[:,:,0]).astype(np.float)    
    heat = add_heat(heat, bboxes_n_frames)
    heat = apply_threshold(heat, counter_thres)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    return labels, bboxes_n_frames, counter
'''Draw bounding boxes on image'''
def draw_labeled_bboxes(img, labels):
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        # Return the image
    return img
    
