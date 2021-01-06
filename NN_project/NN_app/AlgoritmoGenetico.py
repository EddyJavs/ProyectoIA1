#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .Model import NN_Model
import random
import time

#CONSTANTES DEL ALGORITMO
maximo_generaciones = 3 #Número máximo de generaciones que va a tener el algoritmo
suma_anterior = 1 #Para guardar la suma de la población anterior
tabla_hiperparametros = [
    {"alpha":0.00001,"lambda":0,"max_iteration":100,"keep_prob":1},
    {"alpha":0.00005,"lambda":0.01,"max_iteration":200,"keep_prob":0.97},
    {"alpha":0.0001,"lambda":0.05,"max_iteration":400,"keep_prob":0.95},
    {"alpha":0.0005,"lambda":0.1,"max_iteration":800,"keep_prob":0.9},
    {"alpha":0.001,"lambda":0.5,"max_iteration":1600,"keep_prob":0.85},
    {"alpha":0.005,"lambda":1,"max_iteration":2000,"keep_prob":0.75},
    {"alpha":0.01,"lambda":1.5,"max_iteration":2500,"keep_prob":0.65},
    {"alpha":0.05,"lambda":2,"max_iteration":3000,"keep_prob":0.5},
    {"alpha":0.1,"lambda":3,"max_iteration":3500,"keep_prob":0.3},
    {"alpha":0.5,"lambda":5,"max_iteration":6000,"keep_prob":0.1}
]



"""
*   Función que crea la población
"""
def inicializarPoblacion(NoIndividuos,train_set,val_set,capas1):
    poblacion = []

    for i in range(NoIndividuos):
        #La población inicial ya la definió el ingeniero en la tabla
        #Individuo 1
        arregloRandom=[random.randint(0, 9),random.randint(0, 9),random.randint(0, 9),random.randint(0, 9)]
        solucion = Nodo(arregloRandom ,  evaluarFitness(arregloRandom,train_set,val_set,capas1))
        poblacion.append(solucion)
    return poblacion #Retorno la población ya creada






"""
*   Función que verifica si el algoritmo ya llegó a su fin
"""
def verificarCriterio(poblacion, generacion):
    #maximo numero de generaciones
    
        #Si ya llegó al máximo de generaciones lo detengo
    if generacion >= maximo_generaciones:
        return True
    
    return


"""
*   Función que evalúa qué tan buena es una solución, devuelve el valor fitness de la solución
*   @solucion = el número viene en un arreglo como este [0, 1, 1, 1]
"""
def evaluarFitness(solucion,train_set,val_set,capas1):
    valorFitness=0
    print('pos 1:',solucion[0])
    print('pos 2: ',solucion[1])
    print('pos 3:',solucion[2])
    print('pos 4:',solucion[3])
    alph = tabla_hiperparametros[solucion[0]]['alpha']
    lamb = tabla_hiperparametros[solucion[1]]['lambda']
    it = tabla_hiperparametros[solucion[2]]['max_iteration']
    kp = tabla_hiperparametros[solucion[3]]['keep_prob']
    print('alpha:',alph)
    print('lambda: ',lamb)
    print('max it:',it)
    print('kp:',kp)

    nn1 = NN_Model(train_set, capas1, alpha=alph, iterations=it, lambd=lamb, keep_prob=kp)
    nn1.training(False)
    print('Entrenamiento Modelo 1')
    nn1.predict(train_set)
    print('Validacion Modelo 1')
    prediccion,exactitud=nn1.predict(val_set)
    
    return exactitud


"""
*   Función que toma a los mejores padres para luego crear una nueva generación
"""
def seleccionarPadres(poblacion):
    #Se seleccionan los 5 mejores padres
    padres = []
    #seleccion mejores
    poblacion = sorted(poblacion, key=lambda item: item.fitness, reverse=True)[:len(poblacion)] #Los ordena de mayor a menor
    for i in range(int(len(poblacion)/2)):
        padres.append(poblacion[i])
    
    print(len(padres))
    if len(poblacion)/2 == len(padres):
        return padres
    else:
        print("Ocurrio un error al seleccionar padres")
        return "Ocurrio un error al seleccionar padres"
    


"""
*   Función que toma dos soluciones padres y las une para formar una nueva solución hijo
*   Se va a alternar los bits de ambos padres
*   Se va a tomar un bit del padre 1, un bit del padre 2 y así sucesivamente
"""
def cruzar(padre1, padre2):
    #Cada posicion:
    #60% de ser del primer padre
    #40% de ser del segundo padre
    hijo = [0,0,0,0]
    for i in range(4):
        valRandom =random.randrange(100)
        
        if valRandom < 60:
            hijo[i] = padre1[i]
        else:
            hijo[i] = padre2[i]
    return hijo #Retorno al hijo ya cruzado


"""
*   Función que toma una solución y realiza la mutación
*   
"""
def mutar(solucion):
    #Se tiene un 50% de posibilidad de mutar
    valRandom =random.randrange(100)
    if valRandom > 50:
        #si va a mutar
        #verrifico si cada posicion muta o no.
        for i in range(4):
            valRandom =random.randrange(100)
            if valRandom > 50:
                #si muta esa posicion:
                solucion[i] = random.randint(0, 9)
                
        return solucion #Retorno la misma solución, solo que ahora mutó
    else:
        #No va a mutar, retorno el mismo valor
        return solucion
            

"""
*   Función que toma a los mejores padres y genera nuevos hijos
"""
def emparejar(padres,train_set,val_set,capas1):
    nuevaPoblacion = []
    for padre in padres:
        nuevaPoblacion.append(padre) 
    #Genero a los hijos que hagan falta,
    # genero tantos hijos como padres hayan
    for i in range(len(padres)):
        hijo = Nodo()
        if i+1 < len(padres):
            hijo.solucion = cruzar(padres[i].solucion, padres[i+1].solucion)
            hijo.solucion = mutar(hijo.solucion) 
        else:
            hijo.solucion = cruzar(padres[0].solucion, padres[i].solucion)
            hijo.solucion = mutar(hijo.solucion)
        
        hijo.fitness= evaluarFitness(hijo.solucion,train_set,val_set,capas1)
        nuevaPoblacion.append(hijo)
    return nuevaPoblacion


"""
*   Método para imprimir los datos de una población
"""
def imprimirPoblacion(poblacion):
    for individuo in poblacion:
        print('Individuo: ', individuo.solucion, ' Fitness: ', individuo.fitness)


"""
*   Método que ejecutará el algoritmo genético para obtener
*   los coeficientes del filtro
"""
def ejecutar(train_set,val_set,capas1):
    #np.seterr(over='raise')
    print("Algoritmo corriendo")
    generacion = 0
    poblacion = inicializarPoblacion(10,train_set,val_set,capas1)
    fin = verificarCriterio(poblacion, generacion)

    #Imprimo la población
    print('*************** GENERACION ', generacion, " ***************")
    imprimirPoblacion(poblacion)

    while(fin == None):
        padres = seleccionarPadres(poblacion)
        poblacion = emparejar(padres,train_set,val_set,capas1)
        generacion += 1 #Lo pongo aquí porque en teoría ya se creó una nueva generación
        fin = verificarCriterio(poblacion, generacion)
        #generacion += 1

        #Imprimo la población
        #print('*************** GENERACION ', generacion, " ***************")
        #imprimirPoblacion(poblacion)

    #print('Cantidad de generaciones:', generacion)
    #imprimirPoblacion(poblacion) #Población final

    #Obtengo la mejor solución y la muestro
    arregloMejorIndividuo = sorted(poblacion, key=lambda item: item.fitness, reverse=True)[:len(poblacion)] #Los ordena de menor a mayor
    mejorIndividuo = arregloMejorIndividuo[0]

    print('\n\n*************** MEJOR SOLUCION***************')
    print('Individuo: ', mejorIndividuo.solucion,  ' Fitness: ', mejorIndividuo.fitness,    'Generacion: ', generacion)
    #Escribir_en_bitacora(archivoNAME,generacion,mejorIndividuo.solucion)
    alph = tabla_hiperparametros[mejorIndividuo.solucion[0]]['alpha']
    lamb = tabla_hiperparametros[mejorIndividuo.solucion[1]]['lambda']
    it = tabla_hiperparametros[mejorIndividuo.solucion[2]]['max_iteration']
    kp = tabla_hiperparametros[mejorIndividuo.solucion[3]]['keep_prob']

    return [alph,lamb,it,kp]

def Escribir_en_bitacora(archivo,CF,CP,NG,MS):
    ar =  "C:\\Users\\eddja\\Desktop\\Vacas Diciembre 2020\\IA\\LAB\\Proyecto\\Bitacora.bca"
    f = open(ar,'a')
    localtime = time.asctime( time.localtime(time.time()) )
    f.write("-*-*-*-*-*-*-*-*-*-*-*--*-*-*-*-*-*-*-*-*-*-*-*-*--*-*-*-*-*-*-*-*-*-*-*-*-*\n")
    f.write("Fecha y hora de ejecucion: "+localtime+"\n")
    f.write("Nombre del documento CSV utilizado: "+archivo+"\n" )
    f.write("Criterio de finalización utilizado: "+ CF+"\n")
    f.write("Criterio de selección de padres utilizado: "+CP+"\n")
    f.write("Número de generaciones generadas: "+str(NG)+"\n")
    f.write("Mejor solución encontrada por el algoritmo: "+ str(MS)+"\n")

class Nodo:
    #solucion = []
    #fitness = 0 #Valor fitness
    #x = 0 #Para la tarea se guarda el valor de x

    #Le defino parámetros al constructor y le pongo valores por defecto por si no se envían
    def __init__(self, solucion = [], fitness = 0, x = 0):
        self.solucion = solucion
        self.fitness = fitness
        self.x = x



