import os
import cv2 as cv
import numpy as np
import json

train_folders = ['imagedb_train']
sift = cv.xfeatures2d_SIFT.create()


def extract_local_features(path):
    img = cv.imread(path)
    kp = sift.detect(img) #find the keypoints
    desc = sift.compute(img, kp)#calculate descriptors for every keypoint
    desc = desc[1]
    return desc


def create_svm(labels, bow_descs,kernel):
    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_C_SVC) #uses C as the tradeoff parameter between the size of margin and the number of training points which are misclassified
    # svm.setKernel(cv.ml.SVM_LINEAR)
    # svm.setKernel(cv.ml.SVM_CHI2)
    # svm.setKernel(cv.ml.SVM_INTER)
    # svm.setKernel(cv.ml.SVM_POLY)
    # svm.setKernel(cv.ml.SVM_SIGMOID)

    svm.setKernel(kernel)
    svm.setTermCriteria((cv.TERM_CRITERIA_COUNT, 40, 1.e-06))#termination criteria
    svm.trainAuto(bow_descs.astype(np.float32), cv.ml.ROW_SAMPLE, labels)
    svm.save('svm')
    return svm


def match(image, myvocabulary):
    n1 = image.shape[0]
    matches = np.zeros((1, myvocabulary.shape[0]))
    for m in range(n1):
        fv = image[m, :]  # for every keypoint with 127 vectors)
        diff = (fv - myvocabulary) ** 2  # find the difference in distance between clusters and keypoints squared
        distances = np.sum(diff, axis=1)  # add the differences for every vector of each keypoint with each cluster
        i2 = np.argmin(distances)  # find which cluster is the closest one to the keypoint
        matches[0, i2] += 1 # our keypoint belongs to the closest cluster so increase the amount of times this cluster is seen in the image

    return matches #array with the frequencies of the appearences of each cluster in an image


def fncorrect(max):

    correct = 0  # initialize a valiable to help us with the success rate
    if max == 0:
        # print('It is a fighter-jet')
        name = "fighter-jet"
        if name in path:
            correct = correct + 1
    elif max == 1:
        # print('It is a motorbike')
        name = "fire-truck"
        if name in path:
            correct = correct + 1
    elif max == 2:
        # print('It is a school-bus')
        name = "school-bus"
        if name in path:
            correct = correct + 1
    elif max == 3:
        # print('It is a touring-bike')
        name = "touring-bike"
        if name in path:
            correct = correct + 1
    elif max == 4:
        # print('It is an airplane')
        name = "airplane"
        if name in path:
            correct = correct + 1
    elif max == 5:
        # print('It is a car-side')
        name = "car-side"
        if name in path:
            correct = correct + 1

    return correct


# Extract Database
print('Extracting features...')
train_descs = np.zeros((0, 128))
for folder in train_folders:
    files = os.listdir(folder)
    for file in files:
        path1 = os.path.join(folder, file)
        images = os.listdir(path1)
        for image in images:
            path = os.path.join(folder, file, image)
            desc = extract_local_features(path)
            if desc is None:
                continue
            train_descs = np.concatenate((train_descs, desc),
                                         axis=0)  # all the keypoints of all the images in the train folder

# Create vocabulary
print('Creating vocabulary...')
term_crit = (cv.TERM_CRITERIA_EPS, 30, 0.1)
trainer = cv.BOWKMeansTrainer(50, term_crit, 1,
                                  cv.KMEANS_PP_CENTERS)  # 50 clusters,termination criteria,Use kmeans++ center initialization by Arthur and Vassilvitskii
vocabulary = trainer.cluster(train_descs.astype(np.float32))  # make the numbers floats anf train the clusters
np.save('vocabulary.npy', vocabulary)
# Load vocabulary
vocabulary = np.load('vocabulary.npy')
# Create Index
print('Creating index...')
img_paths = []
train_descs = np.zeros((0, 128))
bow_descs = np.zeros((0, vocabulary.shape[0]))  # number of clusters

for folder in train_folders:
    files = os.listdir(folder)
    for file in files:
        path1 = os.path.join(folder, file)
        images = os.listdir(path1)
        for image in images:
            path = os.path.join(folder, file, image)
            desc = extract_local_features(path)

            if desc is None:
                continue
            bow_desc = match(desc, vocabulary)  # array that has for every photo which keypoint goes to each cluster
            img_paths.append(path)
            bow_descs = np.concatenate((bow_descs, bow_desc), axis=0)
np.save('index.npy', bow_descs)
with open('index_paths.txt', mode='w+') as file:
    json.dump(img_paths, file)

# Load Index
bow_descs = np.load('index.npy')
with open('index_paths.txt', mode='r') as file:
    img_paths = json.load(file)



test_folders = ['imagedb_test']

numberofx=[]
differentK=[]
#========= CLASSIFICATION k_NN ==========
    # Διατρέχω την κάθε εικόνα
for x in range(2, 30, 2):
    item1 = 0
    correct = 0
    for folder in test_folders:
        files = os.listdir(folder)
        for file in files:
            path1 = os.path.join(folder, file)
            images = os.listdir(path1)
            for image in images:
                item1 = item1 + 1
                path = os.path.join(folder, file, image)
                desc = extract_local_features(path)
                bow_desc = match(desc, vocabulary)
                K = x  # how many neighbours to take into consideration
                distances = np.sum((bow_desc - bow_descs) ** 2, axis=1)
                sorted_ids = np.argsort(distances)  # clusters sorted from closest to farthest
                sum_fighter_jet = 0
                sum_motorbike = 0
                sum_school_bus = 0
                sum_touring_bike = 0
                sum_airplanes = 0
                sum_car_side = 0
                for i in range(K):
                    if 'fighter-jet' in img_paths[sorted_ids[i]]:
                        sum_fighter_jet += 1 / (distances[sorted_ids[i]] + 1)
                    elif 'motorbikes' in img_paths[sorted_ids[i]]:
                        sum_motorbike += 1 / (distances[sorted_ids[i]] + 1)
                    elif 'school-bus' in img_paths[sorted_ids[i]]:
                        sum_school_bus += 1 / (distances[sorted_ids[i]] + 1)
                    elif 'touring-bike' in img_paths[sorted_ids[i]]:
                        sum_touring_bike += 1 / (distances[sorted_ids[i]] + 1)
                    elif 'airplanes' in img_paths[sorted_ids[i]]:
                        sum_airplanes += 1 / (distances[sorted_ids[i]] + 1)
                    elif 'car-side' in img_paths[sorted_ids[i]]:
                        sum_car_side += 1 / (distances[sorted_ids[i]] + 1)
                sums = [sum_fighter_jet, sum_motorbike, sum_school_bus, sum_touring_bike, sum_airplanes,
                        sum_car_side]

                max1 = np.argmax(sums)  # maximum
                correct = fncorrect(max1) + correct
        # print("")
        # print((correct / item1) * 100, "%\n")
    numberofx.append(x)
    differentK.append((correct / item1) * 100)
print(numberofx)
print(differentK)
pass
# ======================================================= 1 vs all  svm =====================================================

correct_1_vs_all = 0
labels0 = np.array(['fighter-jet' in a for a in img_paths], np.int32)
labels1 = np.array(['motorbikes' in a for a in img_paths], np.int32)
labels2 = np.array(['school-bus' in a for a in img_paths], np.int32)
labels3 = np.array(['touring-bike' in a for a in img_paths], np.int32)
labels4 = np.array(['airplanes' in a for a in img_paths], np.int32)
labels5 = np.array(['car-side' in a for a in img_paths], np.int32)
i=0
svm_kernel = [cv.ml.SVM_LINEAR, cv.ml.SVM_CHI2, cv.ml.SVM_INTER, cv.ml.SVM_SIGMOID, cv.ml.SVM_RBF]
svm_kernel_name = ['SVM_LINEAR', 'SVM_CHI2', 'SVM_INTER', 'SVM_SIGMOID', 'SVM_RBF']
for kernel_type in svm_kernel:

    item2 = 0
    correct_1_vs_all = 0
    svm0 = create_svm(labels0, bow_descs, kernel_type)
    svm1 = create_svm(labels1, bow_descs, kernel_type)
    svm2 = create_svm(labels2, bow_descs, kernel_type)
    svm3 = create_svm(labels3, bow_descs, kernel_type)
    svm4 = create_svm(labels4, bow_descs, kernel_type)
    svm5 = create_svm(labels5, bow_descs, kernel_type)
    for folder in test_folders:
        files = os.listdir(folder)
        for file in files:
            path1 = os.path.join(folder, file)
            images = os.listdir(path1)
            for image in images:
                item2 = item2 + 1  # number of images
                path = os.path.join(folder, file, image)
                desc = extract_local_features(path)
                bow_desc = match(desc, vocabulary)
                responses_1_vs_all = np.zeros(6)
                responses_1_vs_all[0] = \
                    svm0.predict(bow_desc.astype(np.float32), flags=cv.ml.STAT_MODEL_RAW_OUTPUT)[1]
                responses_1_vs_all[1] = \
                    svm1.predict(bow_desc.astype(np.float32), flags=cv.ml.STAT_MODEL_RAW_OUTPUT)[1]
                responses_1_vs_all[2] = \
                    svm2.predict(bow_desc.astype(np.float32), flags=cv.ml.STAT_MODEL_RAW_OUTPUT)[1]
                responses_1_vs_all[3] = \
                    svm3.predict(bow_desc.astype(np.float32), flags=cv.ml.STAT_MODEL_RAW_OUTPUT)[1]
                responses_1_vs_all[4] = \
                    svm4.predict(bow_desc.astype(np.float32), flags=cv.ml.STAT_MODEL_RAW_OUTPUT)[1]
                responses_1_vs_all[5] = \
                    svm5.predict(bow_desc.astype(np.float32), flags=cv.ml.STAT_MODEL_RAW_OUTPUT)[1]
                final_response_1_vs_all = np.argmin(responses_1_vs_all)
                correct_1_vs_all += fncorrect(final_response_1_vs_all)

    print('Success rate of 1 vs all: ' + str((correct_1_vs_all / item2) * 100) + '%' + 'with kernel type ' +
          svm_kernel_name[i])
    i = i + 1

pass
