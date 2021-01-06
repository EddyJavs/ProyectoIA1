from django.shortcuts import render
from pathlib import Path
from .ReadFile import *
from .Data import Data
from .Model import NN_Model
from .Plotter import *
from .AlgoritmoGenetico import *
import os
import csv
# Create your views here.
municipios = []
departamentos = []
nn1 = ""
def NeuronalNetwork(request):
    global municipios
    global departamentos
    global nn1
    if request.method == 'POST':
        datos=request.POST
        #obtengo codigo depto y municipio
        cod_dept=get_cod_dept(datos['departamento'])
        cod_mun=get_cod_mun(datos['municipio'])
        print(cod_dept)
        print(cod_mun)
        cargar_municipios()
        lat,lon=get_coordenadas(cod_dept,cod_mun)
        distancia = get_distancia(float(lat),float(lon))
        arreglo = [[float(conv_genero(datos['genero'])),float(datos['edad']),float(datos['anio']),distancia]]
        #datos_escalados = escalamiento(arreglo)
        print(arreglo)
        arrNP = numpy.array(arreglo)
        print(arrNP.shape)
        arrY = numpy.array([[1]])
        print(arrY.shape)
        predict_set = Data(arrNP.T,arrY )
        prediccion,exactitud = nn1.predict(predict_set)
        print(prediccion)
        msj = ""
        if prediccion > 0.5:
            msj = "el estudiante seguira activo"
        else:
            msj = "el estudiante se trasladara"
        return render(request,'home.html',{'deptos':departamentos,'muns':municipios,'msj':msj})
    else:
        if len(municipios) == 0:
            #para llenar los campos de departamento y municipio
            read_files()
            train_X, train_Y, val_X, val_Y = get_dataFile()
            print(train_X.shape)
            print(train_Y.shape)
            print(val_X.shape)
            print(val_Y.shape)
            #X = get_dataFile()
            # Definir los conjuntos de datos
            
            train_set = Data(train_X, train_Y)
            val_set = Data(val_X, val_Y)
            capas1 = [train_set.n, 3, 3,3,3,3,3,1]
            #algoritmo generico
            print("ALGORITMO GENETICO")
            #hiperparametros = ejecutar(train_set,val_set,capas1)
            #nn1 = NN_Model(train_set, capas1, alpha=hiperparametros[0], iterations=hiperparametros[2], lambd=hiperparametros[1], keep_prob=hiperparametros[3])
            
            nn1 = NN_Model(train_set, capas1, alpha=0.1, iterations=50000, lambd=1, keep_prob=0.97)
            nn1.training(False)
            show_Model([nn1])
            print("HILA")
            print('Entrenamiento Modelo 1')
            nn1.predict(train_set)
            print('Validacion Modelo 1')
            nn1.predict(val_set)
        return render(request,'home.html',{'deptos':departamentos,'muns':municipios})

def read_files():
    global municipios
    global departamentos
    BASE_DIR = Path(__file__).resolve().parent.parent
    directorio=os.path.join(BASE_DIR,'datasets')
    print(directorio)
    deptos=[]
    muns=[]
    with open(directorio+"/Dataset.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            deptos.append(row['nombre'])
    departamentos = set(deptos)
    print(len(departamentos))
    with open(directorio+"/Municipios.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            muns.append(row['Nombre'])
    municipios = set(muns)
    print(len(municipios))
    
def get_cod_dept(dep):
    BASE_DIR = Path(__file__).resolve().parent.parent
    directorio=os.path.join(BASE_DIR,'datasets')
    with open(directorio+"/Dataset.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['nombre'] == dep:
                return row['cod_depto']

def get_cod_mun(mun):
    BASE_DIR = Path(__file__).resolve().parent.parent
    directorio=os.path.join(BASE_DIR,'datasets')
    with open(directorio+"/Municipios.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['Nombre'] == mun:
                return row['Muni']
