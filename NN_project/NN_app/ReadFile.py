#import scipy.io
import os
import csv
import numpy
import math
import random
from pathlib import Path
municipios = []

def get_dataFile():
    BASE_DIR = Path(__file__).resolve().parent.parent
    directorio=os.path.join(BASE_DIR,'datasets')
    #data = scipy.io.loadmat('datasets/data.mat')
    data = []
    
    with open(directorio+"/Dataset.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append([row['Estado'],row['Genero'],row['edad'],row['cod_depto'],row['nombre'],row['cod_muni'],row['municipio'],row['Anio']])
            
    #print(len(data))
    cargar_municipios()
    #train_X = data['X'].T
    #train_Y = data['y'].T
    #val_X = data['Xval'].T
    #val_Y = data['yval'].T

    #print('data[\'y\']', data['y'])
    #print('data[\'y\'].T', data['y'].T)
    #datos = get_data(data)
    #print(datos)
    random.shuffle(data)
    X,Y = get_data(data)
    datos = escalamiento(X)
    arrX = numpy.array(datos)
    #print(arrX.shape)
    arrY = numpy.array(Y)
    #print(arrY.shape)
    slice_point = int(arrX.shape[0]*0.6)
    train_X = arrX[0:slice_point]
    val_X = arrX[slice_point:]
    train_Y = arrY[0:slice_point]
    val_Y = arrY[slice_point:]
    #return arrX,arrY
    return train_X.T, train_Y.T, val_X.T, val_Y.T



def get_data(data):
    datos_utiles = []
    respuestas = []
    for i in data:
        lat,lon=get_coordenadas(i[3],i[5])
        datos_utiles.append([conv_genero(i[1]),i[2],i[7],get_distancia(float(lat),float(lon))])
        if i[0] == "Activo":
            respuestas.append([1])
        else:
            respuestas.append([0])
    return datos_utiles, respuestas

def conv_genero(genero):
    if genero == "MASCULINO":
        return 1
    else:
        return 0

def get_coordenadas(depto,mun):
    for m in municipios:
        if m[0] == depto and m[1] == mun:
            return m[3],m[4]
    return "No se encontro coordenadas"

def get_distancia(lat1, lon1):
    lat2 = 14.589246 
    lon2 = -90.551449
    rad=math.pi/180
    dlat=lat2-lat1
    dlon=lon2-lon1
    R=6372.795477598
    a=(math.sin(rad*dlat/2))**2 + math.cos(rad*lat1)*math.cos(rad*lat2)*(math.sin(rad*dlon/2))**2
    distancia=2*R*math.asin(math.sqrt(a))
    return distancia

def escalamiento(datos):
    edades = []
    anios = []
    distancia = []
    for dato in datos:
        #print(dato)
        edades.append(dato[1])
        anios.append(dato[2])
        distancia.append(dato[3])
    edad_escalonada = []
    anio_escalonada = []
    distancia_escalonada = []
    for i in range(len(edades)):
        edad_escalonada.append((float(edades[i])-float(min(edades)))/(float(max(edades))-float(min(edades))))
        anio_escalonada.append((float(anios[i])-float(min(anios)))/(float(max(anios))-float(min(anios))))
        distancia_escalonada.append((float(distancia[i])-float(min(distancia)))/(float(max(distancia))-float(min(distancia))))
    data = []
    i = 0
    for dato in datos:
        data.append([dato[0],edad_escalonada[i],anio_escalonada[i],distancia_escalonada[i]])
        i += 1
    return data






def cargar_municipios():
    global municipios
    municipios = []
    BASE_DIR = Path(__file__).resolve().parent.parent
    directorio=os.path.join(BASE_DIR,'datasets')
    with open(directorio+"/Municipios.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            municipios.append([row['Depto'],row['Muni'],row['Nombre'],row['Lat'],row['Lon']])