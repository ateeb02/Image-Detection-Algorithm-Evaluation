import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load the image datasets (cats, dogs, and horses)
cat_images = []
dog_images = []
horse_images = []

for i in range(1, 200): # 200 images for each category
    cat_image = cv2.imread(f'cats/cat.{i}.jpg', cv2.IMREAD_GRAYSCALE)
    dog_image = cv2.imread(f'dogs/dog.{i}.jpg', cv2.IMREAD_GRAYSCALE)
    horse_image = cv2.imread(f'horses/horse.{i}.jpg', cv2.IMREAD_GRAYSCALE)

    cat_images.append(cat_image)
    dog_images.append(dog_image)
    horse_images.append(horse_image)

# Extract features from images using SIFT, SURF, and HOG algorithms
sift = cv2.SIFT_create()
surf = cv2.SIFT_create()
hog = cv2.HOGDescriptor()

cat_features_sift = []
dog_features_sift = []
horse_features_sift = []

cat_features_surf = []
dog_features_surf = []
horse_features_surf = []

cat_features_hog = []
dog_features_hog = []
horse_features_hog = []

for image in cat_images:
    _, des = sift.detectAndCompute(image, None)
    cat_features_sift.append(des.flatten())

    _, des = surf.detectAndCompute(image, None)
    cat_features_surf.append(des.flatten())

    hog_feature = hog.compute(image)
    cat_features_hog.append(hog_feature.flatten())

for image in dog_images:
    _, des = sift.detectAndCompute(image, None)
    dog_features_sift.append(des.flatten())

    _, des = surf.detectAndCompute(image, None)
    dog_features_surf.append(des.flatten())

    hog_feature = hog.compute(image)
    dog_features_hog.append(hog_feature.flatten())

for image in horse_images:
    _, des = sift.detectAndCompute(image, None)
    horse_features_sift.append(des.flatten())

    _, des = surf.detectAndCompute(image, None)
    horse_features_surf.append(des.flatten())

    hog_feature = hog.compute(image)
    horse_features_hog.append(hog_feature.flatten())

# Combine features and labels
Xsift = np.concatenate((cat_features_sift, dog_features_sift, horse_features_sift), axis=0)
Ysift = np.concatenate((np.zeros(len(cat_features_sift)), np.ones(len(dog_features_sift)), np.full(len(horse_features_sift), 2)))

Xsurf = np.concatenate((cat_features_surf, dog_features_surf, horse_features_surf), axis=0)
Ysurf = np.concatenate((np.zeros(len(cat_features_surf)), np.ones(len(dog_features_surf)), np.full(len(horse_features_surf), 2)))

Xhog = np.concatenate((cat_features_hog, dog_features_hog, horse_features_hog), axis=0)
Yhog = np.concatenate((np.zeros(len(cat_features_hog)), np.ones(len(dog_features_hog)), np.full(len(horse_features_hog), 2)))


# Split the data into training and test sets (70/30 split)
Xsift_train, Xsift_test, Ysift_train, Ysift_test = train_test_split(Xsift, Ysift, test_size=0.3, random_state=42)

Xsurf_train, Xsurf_test, Ysurf_train, Ysurf_test = train_test_split(Xsurf, Ysurf, test_size=0.3, random_state=42)

Xhog_train, Xhog_test, Yhog_train, Yhog_test = train_test_split(Xhog, Yhog, test_size=0.3, random_state=42)


# Train classifiers (ANN and Random Forest)
#for SIFT
ann_classifier_sift = MLPClassifier(hidden_layer_sizes=(256,), max_iter=500)
ann_classifier_sift.fit(Xsift_train, Ysift_train)

rf_classifier_sift = RandomForestClassifier(n_estimators=100)
rf_classifier_sift.fit(Xsift_train, Ysift_train)

#for SURF
ann_classifier_surf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=500)
ann_classifier_surf.fit(Xsurf_train, Ysurf_train)

rf_classifier_surf = RandomForestClassifier(n_estimators=100)
rf_classifier_surf.fit(Xsurf_train, Ysurf_train)

#for HOG
ann_classifier_hog = MLPClassifier(hidden_layer_sizes=(256,), max_iter=500)
ann_classifier_hog.fit(Xhog_train, Yhog_train)

rf_classifier_hog = RandomForestClassifier(n_estimators=100)
rf_classifier_hog.fit(Xhog_train, Yhog_train)


# Test the classifiers and calculate accuracy
ann_predictions_sift = ann_classifier_sift.predict(Xsift_test)
rf_predictions_sift = rf_classifier_sift.predict(Xsift_test)

ann_predictions_surf = ann_classifier_surf.predict(Xsurf_test)
rf_predictions_surf = rf_classifier_surf.predict(Xsurf_test)

ann_predictions_hog = ann_classifier_hog.predict(Xhog_test)
rf_predictions_hog = rf_classifier_hog.predict(Xhog_test)

#Calculate accuracy
ann_accuracy_sift = accuracy_score(Ysift_test, ann_predictions_sift)
rf_accuracy_sift = accuracy_score(Ysift_test, rf_predictions_sift)

ann_accuracy_surf = accuracy_score(Ysurf_test, ann_predictions_surf)
rf_accuracy_surf = accuracy_score(Ysurf_test, rf_predictions_surf)

ann_accuracy_hog = accuracy_score(Yhog_test, ann_predictions_hog)
rf_accuracy_hog = accuracy_score(Yhog_test, rf_predictions_hog)


# Output accuracy
print("ANN Accuracy for SIFT:             ", ann_accuracy_sift)
print("Random Forest Accuracy for SIFT:   ", rf_accuracy_sift)

print("ANN Accuracy for SURF:             ", ann_accuracy_surf)
print("Random Forest Accuracy for SURF:   ", rf_accuracy_surf)

print("ANN Accuracy for HOG:              ", ann_accuracy_hog)
print("Random Forest Accuracy for HOG:    ", rf_accuracy_hog)

