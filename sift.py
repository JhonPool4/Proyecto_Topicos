# Descriptor: Sift

# ============================ #
#		Librerías
# ============================ #
import cv2 
import numpy as np
import matplotlib.pyplot as plt
from funciones import plot_data, normalizar, h, calcular_costo, gradient_descent, plot_frontera, predecir

# Carga imagen en forma de matriz
img_jirafa = cv2.imread("jirafa.jpg")
img_pug = cv2.imread("pug.jpg")
img_test = cv2.imread("jirafa_test.jpg")

# Convierte la imagen de RGB a escala de grises
gray_jirafa = cv2.cvtColor(img_jirafa, cv2.COLOR_BGR2GRAY)
gray_pug = cv2.cvtColor(img_pug, cv2.COLOR_BGR2GRAY)
gray_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)

# Crea un objeto SIFT
sift = cv2.xfeatures2d.SIFT_create()

# Detecta los KeyPoints y calcula los descriptores
kp_jirafa, des_jirafa = sift.detectAndCompute(gray_jirafa, None)
kp_pug, des_pug = sift.detectAndCompute(gray_pug, None)
kp_test, des_test = sift.detectAndCompute(gray_test, None)


# Dibuja los descriptores y su orientación sobre la imagen en escala de grises
img_jirafa = cv2.drawKeypoints(gray_jirafa,kp_jirafa,img_jirafa, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_pug = cv2.drawKeypoints(gray_pug,kp_pug,img_pug, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Muestra la imagen en esacala de grises con los descriptores y su orientación
#cv2.imshow("Imagen JIRAFA con descriptores", img_jirafa)
#cv2.imshow("Imagen PUG con descriptores", img_pug)



# ================================ #
# 	 Reconocimiento de imágenes    
# ================================ #

# Parámetros de entrenamiento
y_pug = np.ones([des_pug.shape[0] , 1]) 			# Pug = 1
y_jirafa = np.zeros([des_jirafa.shape[0] , 1])		#  Jirafa = 0

# Vector de salida
y = np.concatenate((y_pug, y_jirafa), axis = 0)
# Vector de entrada
X = np.concatenate((des_pug.T, des_jirafa.T), axis = 1)
# Vector de entrada normalizado
Xn, mu, sigma = normalizar(X)

n = y.size
print("#========================================#")
print("\n\n\t\t Empieza programa\n")
print("#========================================#")
print ("Cantidad de instancias: ", n)
print ("Cantidad de caracteristicas: ", Xn.shape[0] )
#plot_data(Xn,y)
#plt.show()

# ============================= #
# 	Regresión Logistica
# ============================= #
# Entrenamiento con datos normalizados

# Inicialización de los parámetros
w = np.zeros((Xn.shape[0],1)) 	# Vector que contiene w1, w2, w3, . . . w128
w0 = 0               	 		# Escalar

# Entrenamiento
w, w0, J_history = gradient_descent(Xn, y, w, w0, 0.0001, 1500)

# Gráfica de la función costo
plt.plot(J_history); plt.ylabel('Costo  $J(w_0,w_1)$'); plt.grid()
plt.xlabel('Iteraciones'); plt.title('Evolución de la función de costo');
plt.show()

# ================================ #
# 			Resultado
# ================================ #
print("\n\n#========================================#")
print("\t Resultado del entrenamiento")
print("#========================================#\n\n")
# Exactitud de entrenamiento
ypred, yprob = predecir(Xn, w, w0)
accuracy = np.mean(ypred == y)*100
print('Exactitud de entrenamiento: {:.2f} %'.format(accuracy))



# ================================ #
# 			Test1
# ================================ #
ypred_t1, yprob_t1 = predecir(Xn, w, w0)

print("\n\n#========================================#")
print("\t Test1")
print("#========================================#\n\n")
print('Probabilidad de predicción:', yprob_t1.squeeze())
print('Clase a la cual pertenece la instancia de prueba:', ypred_t1.squeeze())
salida_t1 = np.sum(ypred_t1)/ypred_t1.size
if salida_t1 >0.5:
	print("Foto de un Pug")
else:
	print("foto de una Jirafa")



# ================================ #
# 			Test2
# ================================ #
Xn_test = des_test.T
ypred_t2, yprob_t2 = predecir(Xn_test, w, w0)
print("\n\n#========================================#")
print("\t Test2")
print("#========================================#\n\n")
print('Probabilidad de predicción:', yprob_t2.squeeze())
print('Clase a la cual pertenece la instancia de prueba:', ypred_t2.squeeze())
salida_t2 = np.sum(ypred_t2)/ypred_t2.size
if salida_t2 >0.5:
	print("Foto de un Pug")
else:
	print("foto de una Jirafa")