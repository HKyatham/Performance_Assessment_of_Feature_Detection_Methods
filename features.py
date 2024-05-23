#!/usr/bin/env python
# coding: utf-8

# Imports
import cv2 as cv
import globals
import numpy as np

# Call function SIFT
def SIFT():
    # Initiate SIFT detector
    SIFT = cv.xfeatures2d.SIFT_create()

    return SIFT

# Call function SURF
def SURF():
    # Initiate SURF descriptor
    SURF = cv.xfeatures2d.SURF_create()

    return SURF

# Call function KAZE
def KAZE():
    # Initiate KAZE descriptor
    KAZE = cv.KAZE_create()

    return KAZE

# Call function BRIEF
def BRIEF():
    # Initiate BRIEF descriptor
    BRIEF = cv.xfeatures2d.BriefDescriptorExtractor_create()

    return BRIEF

# Call function ORB
def ORB():
    # Initiate ORB detector
    ORB = cv.ORB_create()

    return ORB

# Call function BRISK
def BRISK():
    # Initiate BRISK descriptor
    BRISK = cv.BRISK_create()

    return BRISK

# Call function AKAZE
def AKAZE():
    # Initiate AKAZE descriptor
    AKAZE = cv.AKAZE_create()

    return AKAZE

# Call function FREAK
def FREAK():
    # Initiate FREAK descriptor
    FREAK = cv.xfeatures2d.FREAK_create()

    return FREAK

# Call function features
def features(image):
    # Find the keypoints
    keypoints = globals.detector.detect(image, None)

    # Compute the descriptors
    keypoints, descriptors = globals.descriptor.compute(image, keypoints)
    
    return keypoints, descriptors

# Call function prints
def prints(keypoints,
           descriptor):
    # Print detector
    print('Detector selected:', globals.detector, '\n')

    # Print descriptor
    print('Descriptor selected:', globals.descriptor, '\n')

    # Print number of keypoints detected
    print('Number of keypoints Detected:', len(keypoints), '\n')

    # Print the descriptor size in bytes
    print('Size of Descriptor:', globals.descriptor.descriptorSize(), '\n')

    # Print the descriptor type
    print('Type of Descriptor:', globals.descriptor.descriptorType(), '\n')

    # Print the default norm type
    print('Default Norm Type:', globals.descriptor.defaultNorm(), '\n')

    # Print shape of descriptor
    print('Shape of Descriptor:', descriptor.shape, '\n')

# Call function matcher
def matcher(image1,
            image2,
            keypoints1,
            keypoints2,
            descriptors1,
            descriptors2,
            matcher,
            descriptor):

    if matcher == 'BF':
        # Se descritor for um Descritor de Recursos Locais utilizar NOME
        if (descriptor == 'SIFT') or (descriptor == 'SURF') or (descriptor == 'KAZE'):
            normType = cv.NORM_L2
        else:
            normType = cv.NORM_HAMMING

        # Create BFMatcher object
        BFMatcher = cv.BFMatcher(normType = normType,
                                 crossCheck = True)

        # Matching descriptor vectors using Brute Force Matcher
        matches = BFMatcher.match(queryDescriptors = descriptors1,
                                  trainDescriptors = descriptors2)

        # Sort them in the order of their distance
        matches = sorted(matches, key = lambda x: x.distance)

        # Draw first 30 matches
        globals.output = cv.drawMatches(img1 = image1,
                                        keypoints1 = keypoints1,
                                        img2 = image2,
                                        keypoints2 = keypoints2,
                                        matches1to2 = matches[:30],
                                        outImg = None,
                                        flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        return globals.output

    elif matcher == 'FLANN':
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1

        index_params = dict(algorithm = FLANN_INDEX_KDTREE,
                            trees = 5)

        search_params = dict(checks = 50)

        # Converto to float32
        descriptors1 = np.float32(descriptors1)
        descriptors2 = np.float32(descriptors2)

        # Create FLANN object
        FLANN = cv.FlannBasedMatcher(indexParams = index_params,
                                     searchParams = search_params)

        # Matching descriptor vectors using FLANN Matcher
        matches = FLANN.knnMatch(queryDescriptors = descriptors1,
                                 trainDescriptors = descriptors2,
                                 k = 2)

        # Lowe's ratio test
        ratio_thresh = 0.7

        # "Good" matches
        good_matches = []

        # Filter matches
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

        # Draw only "good" matches
        globals.output = cv.drawMatches(img1 = image1,
                                        keypoints1 = keypoints1,
                                        img2 = image2,
                                        keypoints2 = keypoints2,
                                        matches1to2 = good_matches,
                                        outImg = None,
                                        flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        return globals.output

def precision_recall(matches, num_keypoints_img1, num_keypoints_img2):
    # True Positives (TP): Correctly matched keypoints
    TP = len(matches)
    
    # False Positives (FP): Keypoints in img1 that don't match any in img2
    FP = num_keypoints_img1 - TP

    # False Negatives (FN): Keypoints in img2 that don't match any in img1
    FN = num_keypoints_img2 - TP

    # Precision: TP / (TP + FP)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    # Recall: TP / (TP + FN)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    # Calculate repeatability
    # repeatability = TP / min(num_keypoints_img1, num_keypoints_img2) if min(num_keypoints_img1, num_keypoints_img2) > 0 else 0

    return precision, recall#, repeatability

def check_if_same_point(mouse_x, mouse_y, feature_x, feature_y, threshold):
    """ Check if two points are within a certain distance threshold """
    distance = np.sqrt((mouse_x - feature_x) ** 2 + (mouse_y - feature_y) ** 2)
    return distance <= threshold

def find_repeatability(mouse_x, mouse_y, keypoints, repeatability_threshold=5.0): 
    
    flag_consecutive = False

    # Process each image and detect features
    repeatable = False
    for kp in keypoints:
        if check_if_same_point(mouse_x, mouse_y, kp.pt[0], kp.pt[1], repeatability_threshold):
            repeatable = True
            flag_consecutive = True
            break
    
    if repeatable:
        globals.repeatability += 1
    else:
        flag_consecutive = False
    
    if flag_consecutive:
        globals.consecutive += 1
        if globals.consecutive > globals.max_num:
            globals.max_num = globals.consecutive
    else:
        if globals.consecutive > globals.max_num:
            globals.max_num = globals.consecutive
        globals.consecutive = 0
    
    percent_repeat = (globals.repeatability / (globals.image_count - 1)) * 100 if globals.image_count > 1 else 0

    result = (f"Repeatability of selected keypoint:<br/>Repeatability: {globals.repeatability}<br/>"
              f"Number of files: {globals.image_count}<br/>Repeatability %: {percent_repeat:.2f}%<br/>"
              f"Maximum consecutive frames for keypoint being detected: {globals.max_num}<br/>")

    return result

def average_distance(src_pts, dst_pts):
    distances = [np.linalg.norm(src_pts[i] - dst_pts[i]) for i in range(len(src_pts))]
    avg_distance = np.mean(distances)
    print(f"Average distance between matched keypoints: {avg_distance}")