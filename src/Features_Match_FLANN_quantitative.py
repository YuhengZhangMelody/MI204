import numpy as np
import cv2

from matplotlib import pyplot as plt

import sys
if len(sys.argv) != 2:
  print ("Usage :",sys.argv[0],"detector(= orb ou kaze)")
  sys.exit(2)
if sys.argv[1].lower() == "orb":
  detector = 1
elif sys.argv[1].lower() == "kaze":
  detector = 2
else:
  print ("Usage :",sys.argv[0],"detector(= orb ou kaze)")
  sys.exit(2)

#Lecture de la paire d'images
img1 = cv2.imread('./Image_Pairs/torb_small1.png')
print("Dimension de l'image 1 :",img1.shape[0],"lignes x",img1.shape[1],"colonnes")
print("Type de l'image 1 :",img1.dtype)
# img2 = cv2.imread('./Image_Pairs/torb_small2.png')

# Définition de la transformation
rows, cols = img1.shape[:2]
M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1) 
img2 = cv2.warpAffine(img1, M, (cols, rows))


print("Dimension de l'image 2 :",img2.shape[0],"lignes x",img2.shape[1],"colonnes")
print("Type de l'image 2 :",img2.dtype)

#Début du calcul
t1 = cv2.getTickCount()
#Création des objets "keypoints"
if detector == 1:
  kp1 = cv2.ORB_create(nfeatures = 500,#Par défaut : 500
                       scaleFactor = 1.2,#Par défaut : 1.2
                       nlevels = 8)#Par défaut : 8
  kp2 = cv2.ORB_create(nfeatures=500,
                        scaleFactor = 1.2,
                        nlevels = 8)
  print("Détecteur : ORB")
else:
  kp1 = cv2.KAZE_create(upright = False,#Par défaut : false
    		        threshold = 0.001,#Par défaut : 0.001
  		        nOctaves = 4,#Par défaut : 4
		        nOctaveLayers = 4,#Par défaut : 4
		        diffusivity = 2)#Par défaut : 2
  kp2 = cv2.KAZE_create(upright = False,#Par défaut : false
	  	        threshold = 0.001,#Par défaut : 0.001
		        nOctaves = 4,#Par défaut : 4
		        nOctaveLayers = 4,#Par défaut : 4
		        diffusivity = 2)#Par défaut : 2
  print("Détecteur : KAZE")
#Conversion en niveau de gris
gray1 =  cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 =  cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
#Détection et description des keypoints
pts1, desc1 = kp1.detectAndCompute(gray1,None)
pts2, desc2 = kp2.detectAndCompute(gray2,None)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Détection points et calcul descripteurs :",time,"s")
# Calcul de l'appariement
t1 = cv2.getTickCount()
# Paramètres de FLANN 
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50) 

flann = cv2.FlannBasedMatcher(index_params,search_params)

# Ensure descriptors are of type float32
if desc1.dtype != np.float32:
    desc1 = desc1.astype(np.float32)
if desc2.dtype != np.float32:
    desc2 = desc2.astype(np.float32)


matches = flann.knnMatch(desc1,desc2,k=2) 
# Application du ratio test
good = []
for m,n in matches:
  if m.distance < 0.7*n.distance:
    good.append([m])
    # good.append(m)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Calcul de l'appariement :",time,"s")



errors = []
for sublist in good:
    m = sublist[0]
    # Récupérer les points de l'original et de l'image transformée
    original_pt = np.array([pts1[m.queryIdx].pt[0], pts1[m.queryIdx].pt[1], 1])
    transformed_pt = np.array([pts2[m.trainIdx].pt[0], pts2[m.trainIdx].pt[1]])
    
    # Appliquer la transformation connue au point d'intérêt de l'original
    predicted_pt = M.dot(original_pt)
    
    # Calculer l'erreur
    error = np.linalg.norm(transformed_pt - predicted_pt[:2])
    errors.append(error)

# Calcul des métriques
mean_error = np.mean(errors)
median_error = np.median(errors)
print("Erreur moyenne :", mean_error)
print("Erreur médiane :", median_error)


# Affichage
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   flags = 0)

# Affichage des appariements qui respectent le ratio test
img3 = cv2.drawMatchesKnn(gray1,pts1,gray2,pts2,good,None,**draw_params)

Nb_ok = len(good)
plt.imshow(img3),plt.title('%i appariements OK'%Nb_ok)
plt.show()
