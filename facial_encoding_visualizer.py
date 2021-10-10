import face_recognition
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import time

# load the images and calculate the facial encodings beforehand
images = []
encodings = []
def load_images():
    total = len(os.listdir(root))
    for index, directory in enumerate(os.listdir(root)):
        path = root + '/' + directory + '/' + directory + '_0001.jpg'
        image = face_recognition.load_image_file(path)
        images.append(image)
        encodings.append(face_recognition.face_encodings(image))

def evaluate_images(feature = 0):
    placeholder_image = face_recognition.load_image_file('placeholder.jpg')
    top5 = [-1] * 5
    top5_images = [placeholder_image] * 5
    bottom5 = [1] * 5
    bottom5_images = [placeholder_image] * 5

    total = len(encodings)
    for index, encoding in enumerate(encodings):
        if len(encoding) == 1:
            face_encoding = face_recognition.face_encodings(images[index])[0]
        else:
            continue
        value = face_encoding[feature]
        if value > min(top5):
            top5_images[np.argmin(top5)] = images[index]
            top5[np.argmin(top5)] = value
        if value < max(bottom5):
            bottom5_images[np.argmax(bottom5)] = images[index]
            bottom5[np.argmax(bottom5)] = value

    plot_results(top5, top5_images, bottom5, bottom5_images, feature)

def plot_results(top5, top5_images, bottom5, bottom5_images, feature, save = True):
    fig = plt.figure(figsize=(10., 5.))
    plt.title('Top and Bottom 5 Values for feature ' + str(feature))
    plt.axis('off')
    
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(2, 5),  # creates 2x5 grid of axes
                    axes_pad=0.3,  # pad between axes in inch.
                    )

    for index, (ax, im) in enumerate(zip(grid, [top5_images[0], top5_images[1], top5_images[2], top5_images[3], top5_images[4], 
                             bottom5_images[0], bottom5_images[1], bottom5_images[2], bottom5_images[3], bottom5_images[4]])):
        # Iterating over the grid returns the Axes.
        if index < 5:
            ax.set_title(str(top5[index%5])[:8], fontdict=None, loc='center', color = "k")
        if index >= 5:
            ax.set_title(str(bottom5[index%5])[:8], fontdict=None, loc='center', color = "k")
        # Turn off tick labels
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.imshow(im)

    if save:
        plt.savefig('results/feature_' + str(feature))
    else:
        plt.show()

root = 'data'
start_time = time.time()
print('Loading Images')
load_images()
print("--- Load Images: %s seconds ---" % (time.time() - start_time))
for i in range(0, 128):
    start_time = time.time()
    print('Evaluating Feature ' + str(i))
    evaluate_images(feature = i)
    print("--- Evaluate Feature: %s seconds ---" % (time.time() - start_time))