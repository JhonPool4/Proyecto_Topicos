import numpy as np
import matplotlib.pyplot as plt


def plot_data(X, y):
    """
    Grafica los puntos X (compuesto por x1 y x2) en una figura. Se grafica los
    datos positivos (1) con triángulos amarillos, y los negativos (0) con
    círculos verdes.
    
    Argumentos
    ----------
        X - Matriz (2,n) que contiene cada una de las n instancias como columnas.
            Solo se grafica los dos primeros atributos (dos primeras filas)
        y - Vector (1,n) que contiene las clases de las instancias
    
    """
    pos = np.where(y.flatten()==1)[0]
    neg = np.where(y.flatten()==0)[0]
    Xpos = X[:,pos]; Xneg = X[:,neg];
    plt.figure(figsize=(6,6))
    plt.plot(Xpos[0,:], Xpos[1,:],'y^',label='Clase 1')
    plt.plot(Xneg[0,:], Xneg[1,:],'go',label='Clase 0')
    plt.xlabel('$x_1$'); plt.ylabel('$x_2$')
    plt.legend(); plt.axis('scaled')


def  normalizar(X):
    """
    Normaliza cada atributo de X usando la media y la desviación estándar
    
    Argumentos
    ----------
        X - Matriz (d,n) que contiene todos cada una de las n instancias como columnas. Se considera
            que cada instancia tiene d atributos.
    
    Retorna
    -------
        Xnorm - Matriz (d,n) que contiene cada atributo normalizado.
          mu  - Vector de tamaño (d,1) que contiene las medias de cada atributo 
        sigma - Vector de tamaño (d,1) que contiene las desviaciones estándar de cada atributo
        
    """
    Xnorm = X.copy()
    mu = np.mean(X, axis=1, keepdims=True)
    sigma = np.std(X, axis=1, keepdims=True)
    X_norm = (X-mu)/sigma
    return X_norm, mu, sigma


def generacion_bases(X, grado=6):
    """
    Genera bases polinomiales x1, x2, x1^2, x1*x2, x2^2, ... hasta un cierto grado

    Argumentos
    ----------
          X - Matriz de tamaño (2,n) donde n es el número de instancias
      grado - grado de las bases polinomiales

    Retorna
    -------
       Xout - Matriz de tamaño (2+m, n) donde se añade m filas según el grado

    """
    X1 = X[0,:]; X2 = X[1,:]
    res = []
    for i in range(1, grado + 1):
        for j in range(i + 1):
            res.append((X1 ** (i - j)) * (X2 ** j))
    return np.array(res)


def plot_frontera(w0, w):
    """
    Grafica la frontera de decisión definida por w y w0

    Argumentos
    ----------
        w0 - Bias del modelo
         w - Vector (d,1) que contiene los parámetros del modelo (w1, w2, ... wd)

    """
    # Rango de las celdas
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    # Celdas
    z = np.zeros((u.size, v.size))
    # Evaluación de cada una de las celdas
    for i, ui in enumerate(u):
        for j, vj in enumerate(v):
            base = generacion_bases(np.array([[ui], [vj]]))
            print("base: ", base)
            z[i,j] = np.dot(w.T, base) + w0
            #z[i,j] = np.dot(w.T, X) + w0
            
    z = z.T
    # Gráfico de z = 0
    plt.contour(u, v, z, levels=[0], linewidths=2, colors='g')
    plt.contourf(u, v, z, levels=[np.min(z), 0, np.max(z)], cmap='Greens', alpha=0.4)


def h(X, w, w0):
    """
    Calcula la función de hipótesis para la regresión logística.
    
    Argumentos
    ----------
        X - Matriz de tamaño (d,n) que contiene cada una de las n instancias como columnas. Se 
            considera que cada instancia tiene d atributos.
        w - Vector columna [w1;w2;w3;...] de tamaño (d,1) que contiene los parámetros wi (i=1,...,d)
            del modelo
        w0 - "Bias" del modelo (escalar)
        
    Retorna
    -------
        h(X) - Vector fila de tamaño (d,1) que contiene la predicción de cada una de las instancias.
        
    """
    g = 0
    # ====================== Completar aquí ======================
    z = np.dot(w.T, X) + w0
    g = 1/(1 + np.exp(-z ))
    
    
    # =============================================================
    return g


def calcular_costo(X, y, w, w0):
    """
    Calcula el costo para la regresión logística.
    
    Argumentos
    ----------
         X - Matriz de tamaño (d,n) que contiene cada una de las n instancias como columnas. Se considera
            que cada instancia tiene d atributos.
         y - Vector fila de tamaño (1,n) que contiene los valores que se desea predecir.
         w - Vector columna de tamaño (d,1) que contiene los parámetros wi (i=1,...,d) del modelo
        w0 - Parámetro que representa el "bias" del modelo (escalar)
    
    Retorna
    -------
        J - Costo de la regresión logística
        
    """
    # ====================== Completar aquí ======================
    hw_x = h(X,w,w0)
    J = -(1/y.size)*np.sum(y*np.log(hw_x) + (1-y)*np.log(1-hw_x))
    
    
    # =============================================================
    # Dependiendo de cómo se implemente, se puede retornar J o J.squeeze() para tener un escalar
    return J.squeeze()    


def gradient_descent(X, y, w, w0, alfa, num_iters=1500):
    """
    Calcula los parámetros que minimizan la función de costo usando descenso de gradiente.
    
    Argumentos
    ----------
           X - Matriz de tamaño (d,n) que contiene cada una de las n instancias como columnas. Se considera
               que cada instancia tiene d atributos.
           y - Vector fila de tamaño (1,n) que contiene los valores que se desea predecir.
           w - Vector columna de tamaño (d,1) que contiene los parámetros inniciales wi (i=1,...,d) del modelo
          w0 - Parámetro que representa el "bias" inicial del modelo (escalar)
        alfa - Factor de aprendizaje
        num_iters - Número de iteraciones que realizará el algoritmo
    
    Retorna
    -------
        w - Vector columna que contiene los parámetros optimizados del modelo
        w0 - Parámetro (escalar) que representa el bias optimizado
        J_hist - Vector que almacena el valor de la función de costo en cada iteración
        
    """
    n = y.size
    J_history = np.zeros(num_iters)
    cnt = 0
    
    # ====================== Completar aquí =====================
    while (cnt < num_iters):
        e = h(X,w,w0) - y
        w = w - alfa*(1/n)*np.dot(X, e.T  )
        w0 = w0 - alfa*(1/n)*np.sum(e)
        J_history[cnt] = calcular_costo(X, y, w, w0)
        if( (cnt>2) & ( abs(J_history[cnt-1] - J_history[cnt]) < 10e-6   ) ):
            break
            print ("criterio de convergencia alcanzado con iteraciones: ", cnt)

    
        cnt = cnt + 1
        print ("cnt: ", cnt); print("w: ", w); print("w0: ",w0)
      
    # ===========================================================
    return w, w0, J_history


def predecir(X, w, w0):
    """
    Predice si la(s) instancia(s) pertenecen a la clase 0 o 1 usando regresión logística
    
    Argumentos
    ----------
        X - Matriz de tamaño (d,np) que contiene las np instancias a predecir como columnas. Cada
            instancia tiene d atributos.
        w - Vector columna de tamaño (d,1) que contiene los parámetros entrenados wi (i=1,...,d)
       w0 - Parámetro que representa el "bias" entrenado del modelo (escalar)
 
    Retorna
    -------
      ypred - Vector fila de tamaño (1,np) que contiene las predicciónes (0 o 1)
      yprob - Vector fila de tamaño (1,np) que contiene la probabilidad de pertenecer a la clase 1
    
    """
    # ====================== Completar aquí =====================
    h_w = h(X,w,w0) 
  
    ypred = np.where(h_w >= 0.5, 1,0)
    
    
    yprob = pow((h_w),ypred) * pow((1 - h_w),(1 - ypred))
    yprob = yprob.round(2)
    
    # ============================================================
    return ypred, yprob