from Discos import DiscoSimulation

import csv

import numpy as np


def save_data(array, file_csv):
    """
    Guarda la información de un arreglo en un archivo.

    Examples:
        >>> my_array = [1, 4, 7, 3]
        >>> save_data(my_array, "data.csv")

    Args:
        array (array): Arreglo que será guardado en file.
        file (str): Nombre del archivo en el que será guardado el arreglo.

    Returns: 
        None
    """

    with open(file_csv, mode='w', newline='') as file_csv:

        writer_csv = csv.writer(file_csv)

        for row in array:

            writer_csv.writerow(row)



def run_and_save_data(N, M, file_csv):
    """
    Corre la animación y guarda los datos de x_positions de todos los discos si se deja correr por un tiempo establecido.

    Examples:
        >>> run_and_save_data(6, 2000, "data.csv")
        >>> # Cierre la ventana de animación manualmente una vez transcurrido el tiempo establecido.
        Lista de tamaño 6 x 2000 guardada en data.csv.
        

    Args:
        N (int): Número de discos que se registrarán en el archivo.
        M (int): Número de instantes de tiempo que transcurriran, cada instante de tiempo dura 0.05 seg, entonces el tiempo establecido será 0.05 * M seg.
        file_csv (str): Nombre del archivo en el que será guardado el arreglo de posiciones.

    Returns: 
        None
    """

    Radius = np.sqrt(4/N)*0.5

    sim = DiscoSimulation(N, 5, 5, Radius)

    disks = sim.disk_creation()

    sim.animate_movement()
    
    while True:

        print("Transcurrieron " + str(len(disks[0].x_positions)) + " instantes de tiempo")

        if len(disks[0].x_positions) > M:

            positions = [[sim.get_positions()[j][0][i] for i in range(M)] for j in range(N)]

            if __name__ == "__main__":

                save_data(positions, file_csv)

                print("Lista de tamaño " + str(len(positions)) + " x " + str(len(positions[0])) + f" guardada en {file_csv}.")
        break
    


run_and_save_data(25, 6000, "data25.csv")

print("Ya el programa terminó de correr.")
