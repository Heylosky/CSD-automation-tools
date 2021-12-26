from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from models.basemodel import baseModel
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import random
import pickle
import cv2
import os
import tensorflow as tf

dataset = '.\dataset'

EPOCHS = 50
INIT_LR = 1e-3
BS = 16
IMAGE_DIMS = (96, 96, 3)

# disable eager execution
tf.compat.v1.disable_eager_execution()

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images("dataset")))
random.seed(42)
random.shuffle(imagePaths)

# initialize the data and labels
data = []
labels = []

# loop over the images and do the pre-process
for imagePath in imagePaths:
	# store in data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image = img_to_array(image)
	data.append(image)
	# extract set of class labels from the image path and update the labels list
	l = label = imagePath.split(os.path.sep)[-2]
	labels.append(l)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {} images ({:.2f}MB)".format(
	len(imagePaths), data.nbytes / (1024 * 1000.0)))

# binarize the labels using scikit-learn's special multi-label
# binarizer implementation
print("[INFO] class labels:")
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# loop over each of the possible class labels and show them
for (i, label) in enumerate(lb.classes_):
	print("{}. {}".format(i+1, label))

# 80% of training and 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# using a sigmoid activation as the final layer
print("[INFO] compiling model...")
# model = vgg16.build(IMAGE_DIMS=IMAGE_DIMS, classes=len(mlb.classes_), finalAct="sigmoid")
# model = likevgg16.build(IMAGE_DIMS=IMAGE_DIMS, classes=len(mlb.classes_), finalAct="sigmoid")

inputs = baseModel.initialize_inputs(IMAGE_DIMS)
# model = baseModel.base_likevgg(inputs=inputs, classes=len(mlb.classes_), finalAct="sigmoid")
model = baseModel.base_smallervggnet(inputs=inputs, classes=len(lb.classes_), finalAct="softmax")
# model = baseModel.base_restnet50(inputs=inputs, classes=len(mlb.classes_), finalAct="sigmoid")

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# training
print("[INFO] training network...")
H = model.fit(
	x=aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

# save model
print("[INFO] serializing network...")
model.save("model-smallervggnet", save_format="h5")

# save binarizer labels
print("[INFO] serializing label binarizer...")
f = open("labelbin-smallervggnet", "wb")
f.write(pickle.dumps(lb))
f.close()

# save the acc/loss plot to disk
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig("plot-acc-test-smallervggnet")

plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig("plot-loss-test-smallervggnet")