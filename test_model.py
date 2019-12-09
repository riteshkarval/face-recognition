import warnings
warnings.filterwarnings('ignore')

from keras.models import load_model
from keras import backend as K
import re
import numpy as np
from PIL import Image
from collections import Counter

def compute_accuracy(predictions, labels):
    return labels[predictions.ravel() < 0.5].mean()

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def read_image(filename, byteorder='>'):
    
    #first we read the image, as a raw file to the buffer
    with open(filename, 'rb') as f:
        buffer = f.read()
    
    #using regex, we extract the header, width, height and maxval of the image
    header, width, height, maxval = re.search(
        b"(^P5\s(?:\s*#.*[\r\n])*"
        b"(\d+)\s(?:\s*#.*[\r\n])*"
        b"(\d+)\s(?:\s*#.*[\r\n])*"
        b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    
    #then we convert the image to numpy array using np.frombuffer which interprets buffer as one dimensional array
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))


n_of_persons = 40
def get_person(size, total_sample_size, p_id):
    image = read_image('att-database-of-faces/s' + str(1) + '/' + str(1) + '.pgm', 'rw+')
    image = image[::size, ::size]
    dim1 = image.shape[0]
    dim2 = image.shape[1]
    p_imgs = np.zeros([total_sample_size, 1, dim1, dim2]) 
    count = 0    
    for j in range(10):
        img1 = read_image('att-database-of-faces/s' + str(p_id+1) + '/' + str(j + 1) + '.pgm', 'rw+')
        img1 = img1[::size, ::size]
        p_imgs[count, 0, :, :] = img1
        count += 1
    return p_imgs/255


model = load_model('face_model.h5', custom_objects={'contrastive_loss': contrastive_loss})


size = 2
X = []
for i in range(n_of_persons):
    X.append(get_person(size,10,i))

p_id = np.random.randint(36,40)
p_imgs = get_person(size,10,p_id)
st = 0
for i in range(n_of_persons):
    pred = model.predict([X[i], p_imgs]).ravel() < 0.2
    if Counter(pred)[True] >8:
        print("Member")
        print(i+1, p_id+1)
        st = 1
        exit(0)
        
print("Not a member")