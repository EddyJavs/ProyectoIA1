import numpy as np
#np.set_printoptions(threshold=100000) #Esto es para que al imprimir un arreglo no me muestre puntos suspensivos


class NN_Model:

    def __init__(self, train_set, layers, alpha=0.3, iterations=300000, lambd=0, keep_prob=1):
        self.data = train_set
        self.alpha = alpha
        self.max_iteration = iterations
        self.lambd = lambd
        self.kp = keep_prob
        # Se inicializan los pesos
        self.parametros = self.Inicializar(layers)

    def Inicializar(self, layers):
        parametros = {}
        L = len(layers)
        print('layers:', layers)
        for l in range(1, L):
            #np.random.randn(layers[l], layers[l-1])
            #Crea un arreglo que tiene layers[l] arreglos, donde cada uno de estos arreglos tiene layers[l-1] elementos con valores aleatorios
            #np.sqrt(layers[l-1] se saca la raiz cuadrada positiva de la capa anterior ---> layers[l-1]
            parametros['W'+str(l)] = np.random.randn(layers[l], layers[l-1]) / np.sqrt(layers[l-1])
            parametros['b'+str(l)] = np.zeros((layers[l], 1))
            #print(layers[l], layers[l-1], np.random.randn(layers[l], layers[l-1]))
            #print(np.sqrt(layers[l-1]))
            #print(np.random.randn(layers[l], layers[l-1]) / np.sqrt(layers[l-1]))

        return parametros

    def training(self, show_cost=False):
        self.bitacora = []
        for i in range(0, self.max_iteration):
            y_hat, temp = self.propagacion_adelante(self.data)
            cost = self.cost_function(y_hat)
            gradientes = self.propagacion_atras(temp)
            self.actualizar_parametros(gradientes)
            if i % 50 == 0:
                self.bitacora.append(cost)
                if show_cost:
                    print('Iteracion No.', i, 'Costo:', cost, sep=' ')


    def propagacion_adelante(self, dataSet):
        #print("Propagacion hacia adelanta")
        #print(dataSet)
        # Se extraen las entradas
        X = dataSet.x
        
        keys = list(self.parametros.keys())
        #print(keys)
        #print(len(self.parametros))
        # Extraemos los pesos
        #print(keys[0])
        #print(keys[1])
        temp = []
        Ai = X
        cont = 0 
        
        for i in range(int(len(self.parametros)/2)-1):
            #print(i)
            Wi = self.parametros[keys[cont]]
            Bi = self.parametros[keys[cont+1]]
            #print(keys[cont])
            #print(keys[cont+1])
            # ------ Primera capa
            Zi = np.dot(Wi, Ai) + Bi
            Ai = self.activation_function('relu', Zi)
            #Se aplica el Dropout Invertido
            Di = np.random.rand(Ai.shape[0], Ai.shape[1]) #Se generan número aleatorios para cada neurona
            Di = (Di < self.kp).astype(int) #Mientras más alto es kp mayor la probabilidad de que la neurona permanezca
            Ai *= Di
            Ai /= self.kp
            temp.append(Zi)
            temp.append(Ai)
            temp.append(Di)
            cont += 2
        
        W3 = self.parametros[keys[len(keys)-2]]
        b3 = self.parametros[keys[len(keys)-1]]
        #print(keys[len(keys)-2])
        #print(keys[len(keys)-1])
        
        # ------ Tercera capa
        Z3 = np.dot(W3, Ai) + b3
        A3 = self.activation_function('sigmoide', Z3)

        #temp = (Z1, A1, D1, Z2, A2, D2, Z3, A3)
        temp.append(Z3)
        temp.append(A3)
        #print("Temp")
        #print(temp)
        #En A3 va la predicción o el resultado de la red neuronal
        
        return A3, temp

    def propagacion_atras(self, temp):
        #print("Propagacion hacia atras")
        #print(temp)
        # Se obtienen los datos
        m = self.data.m
        Y = self.data.y
        X = self.data.x
        
        W1 = self.parametros["W1"]
        W2 = self.parametros["W2"]
        W3 = self.parametros["W3"]
        
        #(Z1, A1, D1, Z2, A2, D2, Z3, A3) = temp

        #obtengo el ultimo W
        keys = list(self.parametros.keys())
        Wi = self.parametros[keys[len(keys)-2]]
        #derivadas parciales ultima capa
        dZi = temp[len(temp)-1] - Y
        dWi = (1 / m) * np.dot(dZi, temp[len(temp)-4].T) + (self.lambd / m) * Wi
        dbi = (1 / m) * np.sum(dZi, axis=1, keepdims=True)
        gradientes= {}
        gradientes["dZ"+str(int(len(self.parametros)/2))]=dZi
        gradientes["dW"+str(int(len(self.parametros)/2))]=dWi
        gradientes["db"+str(int(len(self.parametros)/2))]=dbi
        
        count = 2 #para llevar el conteo de los pesos
        count2= 3 # para llevar el conteo dentro de temp
        for i in range(int(len(self.parametros)/2)-1):
            
            Wi = self.parametros[keys[len(keys)-count]]
            dAi = np.dot(Wi.T, dZi)
            dAi *= temp[len(temp)-count2]
            dAi /= self.kp
            dZi = np.multiply(dAi, np.int64(temp[len(temp)-count2-1] > 0))
            if i == int(len(self.parametros)/2)-2:
                #print("asdjadadas")
                dWi = 1. / m * np.dot(dZi, X.T) + (self.lambd / m) * self.parametros[keys[len(keys)-count-2]]
            else:
                #print("no")
                dWi = 1. / m * np.dot(dZi, temp[len(temp)-count2-4].T) + (self.lambd / m) * self.parametros[keys[len(keys)-count-2]]
            dbi = 1. / m * np.sum(dZi, axis=1, keepdims=True)
            
            gradientes["dA"+str(int(len(self.parametros)/2)-(i+1))]=dAi
            gradientes["dZ"+str(int(len(self.parametros)/2)-(i+1))]=dZi
            gradientes["dW"+str(int(len(self.parametros)/2)-(i+1))]=dWi
            gradientes["db"+str(int(len(self.parametros)/2)-(i+1))]=dbi
            count += 2
            count2 += 3
        
        #print(gradientes)
        return gradientes

    def actualizar_parametros(self, grad):
        # Se obtiene la cantidad de pesos
        L = len(self.parametros) // 2
        for k in range(L):
            self.parametros["W" + str(k + 1)] -= self.alpha * grad["dW" + str(k + 1)]
            self.parametros["b" + str(k + 1)] -= self.alpha * grad["db" + str(k + 1)]

    def cost_function(self, y_hat):
        # Se obtienen los datos
        Y = self.data.y
        m = self.data.m
        # Se hacen los calculos
        temp = np.multiply(-np.log(y_hat), Y) + np.multiply(-np.log(1 - y_hat), 1 - Y)
        result = (1 / m) * np.nansum(temp)
        # Se agrega la regularizacion L2
        if self.lambd > 0:
            L = len(self.parametros) // 2
            suma = 0
            for i in range(L):
                suma += np.sum(np.square(self.parametros["W" + str(i + 1)]))
            result += (self.lambd/(2*m)) * suma
        return result

    def predict(self, dataSet):
        # Se obtienen los datos
        m = dataSet.m
        Y = dataSet.y
        p = np.zeros((1, m), dtype= np.int)
        # Propagacion hacia adelante
        y_hat, temp = self.propagacion_adelante(dataSet)
        # Convertir probabilidad
        for i in range(0, m):
            p[0, i] = 1 if y_hat[0, i] > 0.5 else 0
        exactitud = np.mean((p[0, :] == Y[0, ]))
        print("Exactitud: " + str(exactitud))
        return y_hat,exactitud


    def activation_function(self, name, x):
        result = 0
        if name == 'sigmoide':
            result = 1/(1 + np.exp(-x))
        elif name == 'tanh':
            result = np.tanh(x)
        elif name == 'relu':
            result = np.maximum(0, x)
        
        #print('name:', name, 'result:', result)
        return result