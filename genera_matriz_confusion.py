from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')



#DATOS PREVIAMENTE OBTENIDOS DEL ALGORITMO DE CLASIFICACIÓN
multiclass = np.array([[17290,766],
                       [387, 12570]])

class_names = ['N', 'S']

plt.figure(figsize=(20,30))
fig, ax = plot_confusion_matrix(conf_mat=multiclass,
                                colorbar=True,
                                show_absolute   =False,
                                show_normed=True,
                                class_names=class_names)
#plt.show()

plt.savefig('matrix.png', dpi=100)