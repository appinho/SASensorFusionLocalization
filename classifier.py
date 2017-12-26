# Imports
import numpy as np
import os
import features
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import time
import cv2
import visualizer

class Classifier(object):

    # Constructor
    def __init__(self,num_train_examples,image):

        # Data directories
        self.root_car = 'vehicles'
        self.root_notcars = 'non-vehicles'

        # Data attributes
        self.image_width = image.shape[0]
        self.image_height = image.shape[1]
        
        # Define classifier attributes
        self.color_space = 'YUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = 9
        self.pix_per_cell = 8
        self.cell_per_block = 2
        self.hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
        self.spatial_size = (32,32)
        self.hist_bins = 32
        self.spatial_feat = True
        self.hist_feat = True
        self.hog_feat = True
        self.num_train_examples = num_train_examples

        # Sliding window search
        windows_1 = self.slide_window(x_start_stop=[540,925],  y_start_stop=[400,432], xy_window=(16,16),   xy_overlap=(0, 0))
        windows_2 = self.slide_window(x_start_stop=[600,1280], y_start_stop=[400,464], xy_window=(64,64),   xy_overlap=(0.5, 0))
        windows_3 = self.slide_window(x_start_stop=[500,1280], y_start_stop=[400,496], xy_window=(96,96),   xy_overlap=(0.5, 0))
        windows_4 = self.slide_window(x_start_stop=[520,1280], y_start_stop=[400,528], xy_window=(128,128), xy_overlap=(0.8, 0))
        windows_5 = self.slide_window(x_start_stop=[340,1280], y_start_stop=[400,656], xy_window=(256,256), xy_overlap=(0.8, 0.8))

        self.windows = windows_2 + windows_3 + windows_4
        print("Number of Windows: ", len(self.windows))

        # Debug
        self.debug = True

        if self.debug:
            windows_image = visualizer.draw_boxes(cv2.cvtColor(image,cv2.COLOR_BGR2RGB), self.windows)
            visualizer.draw_image(windows_image, "Windows",save=True)

    def train(self):
        # Load data
        cars = self.fill_data(self.root_car)
        notcars = self.fill_data(self.root_notcars)

        # Show an example of each kind and print number of examples
        if self.debug:
            test_car = visualizer.read_and_draw_image(cars[0],'Car')
            test_not_car = visualizer.read_and_draw_image(notcars[0],'No_Car')
            print("Number of Car examples: ", len(cars))
            print("Number of Non-Car examples: ", len(notcars))

        # TODO see if image ranges from 0 to 1
        car_features = features.extract_features(cars[0:self.num_train_examples], color_space=self.color_space,
                                                 spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                                 orient=self.orient, pix_per_cell=self.pix_per_cell,
                                                 cell_per_block=self.cell_per_block, hog_channel=self.hog_channel,
                                                 spatial_feat=self.spatial_feat, hist_feat=self.hist_feat,
                                                 hog_feat=self.hog_feat)
        notcar_features = features.extract_features(notcars[0:self.num_train_examples], color_space=self.color_space,
                                                 spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                                 orient=self.orient, pix_per_cell=self.pix_per_cell,
                                                 cell_per_block=self.cell_per_block, hog_channel=self.hog_channel,
                                                 spatial_feat=self.spatial_feat, hist_feat=self.hist_feat,
                                                 hog_feat=self.hog_feat)

        if self.debug:
            from skimage.feature import hog

            car_feat_image = cv2.cvtColor(test_car, cv2.COLOR_RGB2YCrCb)
            for channel in range(car_feat_image.shape[2]):
                channel_image = car_feat_image[:, :, channel]
                fd, hog_image = hog(channel_image,
                                    orientations=self.orient, pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                                cells_per_block=(self.cell_per_block, self.cell_per_block), visualise=True)
                visualizer.draw_two_images(channel_image,hog_image,title='Car_Channel_' + str(channel),save=True)

            notcar_feat_image = cv2.cvtColor(test_not_car, cv2.COLOR_RGB2YCrCb)
            for channel in range(notcar_feat_image.shape[2]):
                channel_image = notcar_feat_image[:, :, channel]
                fd, hog_image = hog(channel_image,
                                    orientations=self.orient, pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                                cells_per_block=(self.cell_per_block, self.cell_per_block), visualise=True)
                visualizer.draw_two_images(channel_image,hog_image,title='No_Car_Channel_' + str(channel),save=True)

        # TODO normalize data
        # TODO try different colorspaces
        # TODO try color HOG

        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        print(X.shape)
        X_scaler = StandardScaler().fit(X)
        self.X_scaler = X_scaler
        scaled_X = X_scaler.transform(X)
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)

        print('Using:', self.orient, 'orientations', self.pix_per_cell,
              'pixels per cell and', self.cell_per_block, 'cells per block')
        print('Feature vector length:', len(X_train[0]))
        # Use a linear SVC
        svc = LinearSVC(C=1000)
        self.svc = svc
        # Check the training time for the SVC
        t = time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

        # Check the prediction time for a single sample
        t = time.time()
        n_predict = 10
        print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
        print('For these', n_predict, 'labels: ', y_test[0:n_predict])
        t2 = time.time()
        print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')

    def debug_classify(self,image):
        t1 = time.time()


        hot_windows = self.search_windows(image, self.windows, self.svc, self.X_scaler, color_space=self.colorspace,
                                                spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                                orient=self.orient, pix_per_cell=self.pix_per_cell,
                                                cell_per_block=self.cell_per_block,
                                                hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
                                                hist_feat=self.hist_feat, hog_feat=self.hog_feat)

        draw_image = np.copy(image)
        window_img = visualizer.draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

        t2 = time.time()
        print(round(t2 - t1, 2), 'Seconds to classify one image')
        return window_img, hot_windows


    def classify(self,image):
        t1 = time.time()


        hot_windows = self.search_windows(image)

        # draw_image = np.copy(image)
        # window_img = visualizer.draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

        t2 = time.time()
        print(round(t2 - t1, 2), 'Seconds to classify one image')
        return hot_windows


    def fill_data(self,directory):
        data_list = []
        for root,dirs,files in os.walk(directory):
            for file in files:
                if file[0] == '.':
                    continue
                data_list.append(os.path.join(root,file))
        return data_list


    def slide_window(self, x_start_stop=(None, None), y_start_stop=(None, None),
                     xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        """
        Method to slide a window over the image
        :param img: Input image
        :param x_start_stop: Start and stop position on the x-axis
        :param y_start_stop: Start and stop position on the y-axis
        :param xy_window: Window size
        :param xy_overlap: Overlapping factor
        :return: List of windows
        """

        # If x and/or y start/stop positions not defined, set to image size
        if not x_start_stop[0]:
            x_start = 0
        else:
            x_start = x_start_stop[0]
        if not x_start_stop[1]:
            x_stop = self.image_width
        else:
            x_stop = x_start_stop[1]
        if not y_start_stop[0]:
            y_start = 0
        else:
            y_start = y_start_stop[0]
        if not y_start_stop[1]:
            y_stop = self.image_height
        else:
            y_stop = y_start_stop[1]

        # Compute the span of the region to be searched
        xspan = x_stop - x_start
        yspan = y_stop - y_start
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
        ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
        nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
        ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs * nx_pix_per_step + x_start
                endx = startx + xy_window[0]
                starty = ys * ny_pix_per_step + y_start
                endy = starty + xy_window[1]
                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list

    def search_windows(self,img):

        #1) Create an empty list to receive positive detection windows
        on_windows = []
        #2) Iterate over all windows in the list
        for window in self.windows:
            #3) Extract the test window from original image
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            #4) Extract features for that window using single_img_features()
            features = self.single_img_features(test_img)
            #5) Scale extracted features to be fed to classifier
            test_features = self.X_scaler.transform(np.array(features).reshape(1, -1))
            #6) Predict using your classifier
            prediction = self.svc.predict(test_features)
            #7) If positive (prediction == 1) then save the window
            if prediction == 1:
                on_windows.append(window)
                # visualizer.draw_image(test_img)
        #8) Return windows for positive detections
        return on_windows

    def single_img_features(self,img):
        #1) Define an empty list to receive features
        img_features = []
        #2) Apply color conversion if other than 'RGB'
        if self.color_space != 'RGB':
            if self.color_space == 'HSV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif self.color_space == 'LUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif self.color_space == 'HLS':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif self.color_space == 'YUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif self.color_space == 'YCrCb':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(img)
        #3) Compute spatial features if flag is set
        if self.spatial_feat == True:
            spatial_features = features.bin_spatial(feature_image, size=self.spatial_size)
            #4) Append features to list
            img_features.append(spatial_features)
        #5) Compute histogram features if flag is set
        if self.hist_feat == True:
            hist_features = features.color_hist(feature_image, nbins=self.hist_bins)
            #6) Append features to list
            img_features.append(hist_features)
        #7) Compute HOG features if flag is set
        if self.hog_feat == True:
            if self.hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.extend(features.get_hog_features(feature_image[:,:,channel],
                                        self.orient, self.pix_per_cell, self.cell_per_block,
                                        vis=False, feature_vec=True))
            else:
                hog_features = features.get_hog_features(feature_image[:,:,self.hog_channel], self.orient,
                                        self.pix_per_cell, self.cell_per_block, vis=False, feature_vec=True)
            #8) Append features to list
            img_features.append(hog_features)

        #9) Return concatenated array of features
        return np.concatenate(img_features)

    def convert_color(self,img, conv='RGB2YCrCb'):
        if conv == 'RGB2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        if conv == 'BGR2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        if conv == 'RGB2LUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

    # Define a single function that can extract features using hog sub-sampling and make predictions
    def find_cars(self,img):

        t1 = time.time()

        ystart = 400
        ystop = 656
        scale = 1
        orient = 9
        pix_per_cell = 8
        cell_per_block = 2
        spatial_size = (32,32)
        hist_bins = 32

        draw_img = np.copy(img)
        img = img.astype(np.float32) / 255

        img_tosearch = img[ystart:ystop, :, :]
        ctrans_tosearch = self.convert_color(img_tosearch, conv='RGB2YCrCb')
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))
            # visualizer.draw_image(ctrans_tosearch)

        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
        nfeat_per_block = orient * cell_per_block ** 2

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1 = features.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = features.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = features.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * pix_per_cell
                ytop = ypos * pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

                # Get color features
                spatial_features = features.bin_spatial(subimg, size=spatial_size)
                hist_features = features.color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                test_features = self.X_scaler.transform(
                    np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
                test_prediction = self.svc.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                                  (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)

        t2 = time.time()
        print(round(t2 - t1, 2), 'Seconds to classify one image')
        return draw_img