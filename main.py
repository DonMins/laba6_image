import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


def addNoise(image, noise_percentage):
    vals = len(image.flatten())
    out = np.copy(image)
    nose = int(np.ceil(noise_percentage * vals / 100))
    # Salt mode
    num_salt = int(nose)
    listAllCoord = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            listAllCoord.append([i, j])

    random.shuffle(listAllCoord)
    for i in range(0, int(num_salt)):
        coord = listAllCoord.pop(np.random.randint(0, len(listAllCoord)))
        nose = np.array(image[coord[0], coord[1]] + [random.normalvariate(0, 20), random.normalvariate(0, 20),
                                                     random.normalvariate(0, 20)], dtype=int)
        for i in range(len(nose)):
            if nose[i] > 255:
                nose[i] = 255
            elif nose[i] < 0:
                nose[i] = 0
        out[coord[0], coord[1]] = nose

    return out

def getError(img, img2):
    img3 = np.copy(img)
    img4 = np.copy(img2)
    img3 = img3 / 255
    img4 = img4 / 255
    width = img.shape[1]
    height = img.shape[0]
    error = (np.sum((np.array(img3.flatten()) - np.array(img4.flatten())) ** 2,
                    dtype=np.float64) / (width * height * 3)) ** 0.5
    return error

def getMediana(img, listNotZero, kernerl):
    listPixel = []
    for m, n in listNotZero:
        count = kernerl[m,n]
        for k in range(count):
            listPixel.append(img[m, n])
    return listPixel

def medfilt(image, kernel,rank = None):

    heightM, widthM = kernel.shape
    height, width = image.shape

    centerHeightM = int(np.ceil(heightM / 2) - 1)
    centerwidthM = int(np.ceil(widthM / 2) - 1)

    listNotZero = []
    for i in range(heightM):
        for j in range(widthM):
            if kernel[i, j] != 0:
                listNotZero.append([i, j])

    img2 = np.zeros((height + centerHeightM * 2, width + centerwidthM * 2), np.uint8)
    img2[centerHeightM:height + centerHeightM, centerwidthM:width + centerwidthM] = image

    for i in range(centerHeightM+1):
        img2[i, :] = img2[centerHeightM, :]
        img2[height + 1 + i, :] = img2[height, :]

    for i in range(centerwidthM+1):
        img2[:, i] = img2[:, centerwidthM]
        img2[:, width + 1 + i] = img2[:, width]


    new_image = np.zeros((height, width), np.uint8)

    for i in range(centerHeightM, height + centerHeightM):
        for j in range(centerwidthM, width + centerwidthM):
            chunk = img2[i - centerHeightM: i + centerHeightM + 1, j - centerwidthM: j + centerwidthM + 1]
            listPixel = getMediana(chunk, listNotZero, kernel)
            listPixel.sort()
            if rank == None:
                medianIndex = int(np.ceil(len(listPixel) / 2)) - 1
            else:
                medianIndex = rank
            new_image[i - centerHeightM][j - centerwidthM] = listPixel[medianIndex]

    return new_image



if __name__ == '__main__':
    window = np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 2, 1, 0],
        [1, 2, 3, 2, 1],
        [0, 1, 2, 1, 0],
        [0, 0, 1, 0, 0]])


    img = cv2.imread("putin.png")
    imgNose = cv2.imread("imgNose.jpg")
    imgNose5 = cv2.imread("Res.jpg")

    cv2.imshow("putin", imgNose)

    b, g, r = cv2.split(img)
    bNose, gNose, rNose = cv2.split(imgNose)

    bm = medfilt(bNose,window)
    gm = medfilt(gNose,window)
    rm = medfilt(rNose,window)

    res = cv2.merge((bm, gm, rm))

    cv2.imshow("res", res)

    print("------------------------------------------------------------------")
    print("Исходная с исходной, ошибка восстановления", getError(img, img))
    print("Исходная с зашумленной, ошибка восстановления", getError(img, imgNose))
    print("Исходная с отфильтрованной, ошибка восстановления", getError(img, res))
    print("Исходная с отфильтрованной, из 5 лабы ошибка восстановления", getError(img, imgNose5))
    print("В процентном соотношении стало лучше на ",
          100 - (getError(img, res) * 100 / getError(img, imgNose)), " %")

    cv2.waitKey(0)




