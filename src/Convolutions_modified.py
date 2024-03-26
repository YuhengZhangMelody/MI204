import numpy as np
import cv2

from matplotlib import pyplot as plt

#Lecture image en niveau de gris et conversion en float64
img=np.float64(cv2.imread('./Image_Pairs/FlowerGarden2.png',0))
(h,w) = img.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes")

#Méthode directe
# Noyaux Sobel pour le calcul du gradient en x et en y
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])

# Méthode directe pour les gradients
t1 = cv2.getTickCount()

# Initialisation des images de gradients à zéro
I_x = np.zeros_like(img)
I_y = np.zeros_like(img)

# Calcul du gradient en x (I_x)
for y in range(1, h-1):
    for x in range(1, w-1):
        I_x[y, x] = (sobel_x[0, 0] * img[y-1, x-1] + sobel_x[0, 1] * img[y-1, x] + sobel_x[0, 2] * img[y-1, x+1] +
                     sobel_x[1, 0] * img[y, x-1]   + sobel_x[1, 1] * img[y, x]   + sobel_x[1, 2] * img[y, x+1] +
                     sobel_x[2, 0] * img[y+1, x-1] + sobel_x[2, 1] * img[y+1, x] + sobel_x[2, 2] * img[y+1, x+1])

# Calcul du gradient en y (I_y)
for y in range(1, h-1):
    for x in range(1, w-1):
        I_y[y, x] = (sobel_y[0, 0] * img[y-1, x-1] + sobel_y[0, 1] * img[y-1, x] + sobel_y[0, 2] * img[y-1, x+1] +
                     sobel_y[1, 0] * img[y, x-1]   + sobel_y[1, 1] * img[y, x]   + sobel_y[1, 2] * img[y, x+1] +
                     sobel_y[2, 0] * img[y+1, x-1] + sobel_y[2, 1] * img[y+1, x] + sobel_y[2, 2] * img[y+1, x+1])

t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Méthode directe :",time,"s")

# Calcul de la norme du gradient
gradient_magnitude = np.sqrt(I_x**2 + I_y**2)

# Affichage correct de la norme du gradient
# Normalisation des valeurs pour l'affichage
gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

cv2.imshow('Magnitude du Gradient', gradient_magnitude)
# cv2.waitKey(0)

plt.subplot(121)
plt.imshow(gradient_magnitude, cmap='gray')
plt.title('Magnitude du Gradient - Méthode Directe')


#Méthode filter2D
t1 = cv2.getTickCount()

kernel_x = np.array([[1, 0, -1]])
kernel_y = np.array([[1], [0], [-1]])

I_x = cv2.filter2D(img, -1, kernel_x)
I_y = cv2.filter2D(img, -1, kernel_y)

gradient_magnitude = np.sqrt(I_x**2 + I_y**2)

gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)

img3 = np.uint8(gradient_magnitude)

t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Méthode filter2D :",time,"s")

cv2.imshow('Avec filter2D',img3/255.0)
#Convention OpenCV : une image de type flottant est interprétée dans [0,1]
# cv2.waitKey(0)

plt.subplot(122)
plt.imshow(img3,cmap = 'gray',vmin = 0.0,vmax = 255.0)
#Convention Matplotlib : par défaut, normalise l'histogramme !
plt.title('Convolution - filter2D')

plt.show()
