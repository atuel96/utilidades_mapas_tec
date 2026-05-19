## Prerar Entorno Virtual (Recomendado)

### Usando venv
``` bash
python venv .tec
source .tec/bin/activate
pip install -r requirements.txt
```
### Usando uv venv
``` bash
uv venv .tec
source .tec/bin/activate
uv pip install -r requirements.txt
```

### Usando conda

``` bash
conda create --name tec python
conda activate tec
pip install -r requirements.txt
```

## Sin Entorno Virtual

``` bash
pip install -r requirements.txt
```

## Crear Dataset

``` python
from parse_tec import convert_to_parquet

source_folder = "path_to_files/" # Datos procesados (Gopi o Ciraolo)
output_folder = "path_to_dataset/" # Donde va a crearse la base de datos

# Crear base de datos
convert_to_parquet(source_folder, output_folder)
```

## Crear Mapa

``` python
import pandas as pd
import matplotlib.pyplot as plt
from make_map import plot_map

dataset_folder = "path_to_dataset/" # Donde va a crearse la base de datos

spacing = 1 # grados de resolución
region = [-75, -50, -55, -18]  # Grados Oeste, Este, Sor, Norte
date = pd.to_datetime("2024-05-10 01:00:00") # fecha
target = "vTEC"
cmap = "jet"

ax = plot_map(date, dataset_path=dataset_path, maxdist=2)

# puedo modificar el plot:
# ax.title("nuevo titulo")

# puedo guardar el plot
# plt.savefig("mapa.png")
```




