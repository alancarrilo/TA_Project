import pandas as pd

# Cargar el dataset (asegúrate de que el archivo está en la misma carpeta o especifica la ruta completa)
file_path = 'aapl_5m_train.csv'  # Cambia la ruta si es necesario
data = pd.read_csv(file_path)

# Ver las primeras filas del dataset para asegurarnos de que se ha cargado correctamente
print(data.head())
