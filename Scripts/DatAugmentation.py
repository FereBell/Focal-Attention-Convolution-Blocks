import tensorflow as tf
import numpy as np
import random
import cv2



def Final_Augm(img, msk):
  random_bit = np.random.randint(10)
  img= horizontal_flip(img, random_bit)
  msk= horizontal_flip(msk, random_bit)

  random_bit = np.random.randint(10)
  img= vertical_flip(img, random_bit)
  msk= vertical_flip(msk, random_bit)

  random_bit = np.random.randint(10)
  ratio_v= round(np.random.random(1)[0], 1)
  ratio_v = random.uniform(-ratio_v, ratio_v)
  img= vertical_shift(img, ratio_v, random_bit)
  msk= vertical_shift(msk, ratio_v, random_bit)

  random_bit = np.random.randint(10)
  ratio_h= round(np.random.random(1)[0], 1)
  ratio_h = random.uniform(-ratio_h, ratio_h)
  img= horizontal_shift(img, ratio_h, random_bit)
  msk= horizontal_shift(msk, ratio_h, random_bit)

  random_bit = np.random.randint(10)
  img= brillo_aug(img, random_bit)

  random_bit = np.random.randint(10)
  img= contrastre_agu(img, random_bit)

  random_bit = np.random.randint(5)
  img= noisy(img, random_bit)

  random_bit = np.random.randint(10)
  img, msk= cot_auto(img, msk, random_bit)

  random_bit = np.random.randint(10)
  if 4> random_bit:
    angle= int(round(np.random.random(1)[0], 2)*100)
    angle = int(random.uniform(-angle, angle))
    img= rotation(img, angle)
    msk= rotation(msk, angle)

  return img, msk

def brillo_aug(img, flag):
    if 4> flag:
        value = np.random.choice(np.array([-70, -40, -20, 20, 40, 70]))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if value >= 0:
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value
        else:
            lim = np.absolute(value)
            v[v < lim] = 0
            v[v >= lim] -= np.absolute(value)

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img
    else:
        return img

def contrastre_agu(img, flag):
    if 4> flag:
        value = np.random.choice(np.array([-70, -40, -20, 20, 40, 70]))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if value >= 0:
            lim = 255 - value
            s[s > lim] = 255
            s[s <= lim] += value
        else:
            lim = np.absolute(value)
            s[s < lim] = 0
            s[s >= lim] -= np.absolute(value)

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img
    else:
        return img

def fill(img, h, w):
    img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
    return img

def vertical_shift(img, ratio, flag):
    if 2> flag:
        if ratio > 1 or ratio < 0:
            return img
        h = img.shape[1]
        w = img.shape[0]
        to_shift = h*ratio
        if ratio > 0:
            img = img[:int(h-to_shift), :, :]
        if ratio < 0:
            img = img[int(-1*to_shift):, :, :]
        img = fill(img, h, w)
        return img
    else:
        return img

def horizontal_shift(img, ratio, flag):
    if 2> flag:
        if ratio > 1 or ratio < 0:
            return img
        h, w = img.shape[:2]
        to_shift = w*ratio
        if ratio > 0:
            img = img[:, :int(w-to_shift), :]
        if ratio < 0:
            img = img[:, int(-1*to_shift):, :]
        img = fill(img, h, w)
        return img
    else:
        return img

def horizontal_flip(img, flag):
    if 4> flag:
        return cv2.flip(img[:, :, :], 1)
    else:
        return img

def vertical_flip(img, flag):
    if 4> flag:
        return cv2.flip(img[:, :, :], 0)
    else:
        return img

def rotation(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img

def noisy(img, noise_type= 0):

    if noise_type == 0:
        image=img.copy()
        mean=0
        st=0.7
        gauss = np.random.normal(mean,st,image.shape)
        gauss = gauss.astype('uint8')
        image = cv2.add(image,gauss)
        return image

    elif noise_type == 1:
        image=img.copy()
        prob = 0.05
        if len(image.shape) == 2:
            black = 0
            white = 255
        else:
            colorspace = image.shape[2]
            if colorspace == 3:  # RGB
                black = np.array([0, 0, 0], dtype='uint8')
                white = np.array([255, 255, 255], dtype='uint8')
            else:  # RGBA
                black = np.array([0, 0, 0, 255], dtype='uint8')
                white = np.array([255, 255, 255, 255], dtype='uint8')
        probs = np.random.random(image.shape[:2])
        image[probs < (prob / 2)] = black
        image[probs > 1 - (prob / 2)] = white
        return image
    else:
        return img
    
def cot_auto(img, msk, flag):
    if 4< flag:
        l= img.shape[0]
        p= img.shape[1]
        f1= np.random.randint(30)
        f2= np.random.randint(30)
        p1= np.random.randint(f1, l-f1)
        p2= np.random.randint(f2, p-f2)
        rect= np.zeros([f1, f2, 3])
        img[p1:f1+p1, p2:f2+p2, :] = rect
        msk[p1:f1+p1, p2:f2+p2, :] = rect

        return img, msk
    else:
        return img, msk

class MiClasificacion(tf.keras.utils.Sequence):
    def __init__(self, input_img_paths, target_img_paths, ID_input, batch_size= 32, img_size= (304, 304, 3), train= True):

        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.ID_input = ID_input
        self.cont= 0

    def __len__(self):
        return len(self.ID_input) // self.batch_size

    def __getitem__(self, idx):
        if (self.cont== len(self.ID_input) // self.batch_size) or (self.cont== 0):
            np.random.shuffle(self.ID_input)
            self.cont= 0

        i = idx * self.batch_size
        batch_img= []
        batch_tar= []
        self.cont += 1

        for ig in self.ID_input[i : i + self.batch_size]:
            batch_img.append(self.input_img_paths[ig])
            batch_tar.append(self.target_img_paths[ig])

        X, Y= self.__data_generation(batch_img, batch_tar)

        return X, Y

    def __data_generation(self, biip, btip):
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        j= 0

        for path_i, path_m in zip(biip, btip):
            try:
              img= cv2.imread(path_i)
              msk= cv2.imread(path_m)
              img, msk= Final_Augm(img, msk)
              img= cv2.resize(img, self.img_size, interpolation = cv2.INTER_AREA)
              msk= cv2.resize(msk, self.img_size, interpolation = cv2.INTER_AREA)
            except:
              img= cv2.imread(path_i)
              msk= cv2.imread(path_m)
            img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            msk= cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)

            x[j] = np.array(img)/255.0
            y[j] = np.expand_dims(msk, 2)

            for r in range(y.shape[1]):
                for g in range(y.shape[2]):
                    if y[j, r, g, 0]!= 0:
                        y[j, r, g, 0]= 1
            j+= 1
        x= tf.convert_to_tensor(x, dtype=tf.float32)
        
        return tf.image.per_image_standardization(x), tf.convert_to_tensor(y, dtype=tf.uint8)