#%%
import os
import glob
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
#%%
# -------- CONFIGURATION ----------
DATA_DIR = "flowers"               # structure: data/<class>/*.jpg
IMAGE_MAX_SIZE = 400           # max dimension (to limit computation)
VOCAB_SIZE = 150               # number of visual words (k)
RANDOM_STATE = 42
TEST_SIZE = 0.2
KNN_NEIGHBORS = 5
BATCH_KMEANS = False
# ---------------------------------
#%%
def load_images_and_labels(data_dir):
    images = []
    labels = []
    classes = sorted(os.listdir(data_dir))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for c in classes:
        files = glob.glob(os.path.join(data_dir, c, "*"))
        for f in files:
            img = cv2.imread(f)
            if img is None:
                continue
            # Resize to limit computation:
            h, w = img.shape[:2]
            scale = IMAGE_MAX_SIZE / max(h, w) if max(h, w) > IMAGE_MAX_SIZE else 1.0
            if scale != 1.0:
                img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            images.append(img)
            labels.append(class_to_idx[c])
    return images, np.array(labels), classes
#%%
# 1) Load
images, labels, classes = load_images_and_labels(DATA_DIR)
print(f"Loaded {len(images)} images from {len(classes)} classes.")
#%%
sift = cv2.SIFT_create()  # if this raises an error, install opencv-contrib-python
# 3) Extract descriptors for all images
image_descriptors = []  # list of desc arrays (N_i x 128)
all_descriptors = []    # will stack to (M x 128)
# what N_i and M means here ?
# N_i: number of keypoints detected in image i
# M: total number of keypoints detected across all images

for img in images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, desc = sift.detectAndCompute(gray, None)
    if desc is None:
        # no keypoints detected (very rare), use an empty array
        desc = np.zeros((0, 128), dtype=np.float32)
    image_descriptors.append(desc)
    if desc.shape[0] > 0:
        all_descriptors.append(desc)

if len(all_descriptors) == 0:
    raise RuntimeError("No descriptors found in any image.")
all_descriptors_stacked = np.vstack(all_descriptors).astype(np.float32)
print("Total descriptors:", all_descriptors_stacked.shape)

# save description as numPy array
np.save('all_descriptors_stacked.npy', all_descriptors_stacked)


#%%
print("Tok descriptors from image:", image_descriptors[0].shape)
#%%
# 4) Build vocabulary with k-means (MiniBatchKMeans for speed)
if BATCH_KMEANS:
    kmeans = MiniBatchKMeans(n_clusters=VOCAB_SIZE, random_state=RANDOM_STATE, batch_size=1000)
else:
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=VOCAB_SIZE, random_state=RANDOM_STATE)

kmeans.fit(all_descriptors_stacked)
vocab = kmeans.cluster_centers_
print("KMeans done. Vocab shape:", vocab.shape)
#%%
image_histograms = np.zeros((len(images), VOCAB_SIZE), dtype=np.float32)
for i, desc in enumerate(image_descriptors):
    if desc.shape[0] == 0:
        hist = np.zeros(VOCAB_SIZE, dtype=np.float32)
    else:
        words = kmeans.predict(desc)  # nearest centroid index for each descriptor
        hist, _ = np.histogram(words, bins=np.arange(VOCAB_SIZE+1))
    # Normalize histogram (L2)
    if hist.sum() > 0:
        hist = hist.astype(np.float32)
        hist = hist / np.linalg.norm(hist)
    image_histograms[i] = hist
#%%
# 6) Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    image_histograms, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels)

#%%
# 7) KNN classifier
knn = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS, metric='euclidean')
knn.fit(X_train, y_train)
#%%
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Test accuracy:", acc)
print(classification_report(y_test, y_pred, target_names=classes))
#%%
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
plt.imshow(cm, interpolation='nearest')
plt.title("Confusion matrix")
plt.colorbar()
plt.xticks(np.arange(len(classes)), classes, rotation=90)
plt.yticks(np.arange(len(classes)), classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
#%%
