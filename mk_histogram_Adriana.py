import csv

import matplotlib.pyplot as plt

import numpy as np



def Histogram(N, divisions, file_csv):
    """
    Genera un histograma a partir de los datos de las posiciones de N discos.

    Examples:
        >>> Histogram(6, 50, "data.csv")
        
    Args:
        N (int): Número de discos registrados en el archivo .csv (Solo importante para el título)
        divisions (int): Número de divisiones del intervalo
        file_csv (str): Nombre del archivo en el que está guardado el arreglo de posiciones.

    Returns: 
        None
    """

    all_positions_str = []

    with open(file_csv, 'r') as file_csv:

        lector_csv = csv.reader(file_csv)
    
        for row in lector_csv:

            all_positions_str.extend(row)

    all_positions = [float(i) for i in all_positions_str]

    plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})

    weights = np.ones_like(all_positions) / len(all_positions)

    plt.hist(all_positions, bins=divisions, weights=weights)

    plt.gca().set(title=f'Histograma de probabilidad para {N} discos', ylabel='Probabilidad')

    plt.xlim(-2.5, 2.5)

    plt.xlabel("Posición x")

    plt.show()        

Histogram(25, 100, "data25.csv")
