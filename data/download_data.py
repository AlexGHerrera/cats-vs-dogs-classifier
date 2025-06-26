import os
import zipfile
import gdown

# ID del archivo en Google Drive
FILE_ID = "1uqJ3QW7vY8tqOCz3xmJWPB7uyy9L_RwT"
OUTPUT_ZIP = "data/dataset.zip"
EXTRACT_FOLDER = "data/"

# Crear carpeta si no existe
os.makedirs(EXTRACT_FOLDER, exist_ok=True)

# Descargar solo si no existe
if not os.path.exists(os.path.join(EXTRACT_FOLDER, "train")):  # o el nombre de tu carpeta raíz
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    print("Descargando dataset desde Google Drive...")
    gdown.download(url, OUTPUT_ZIP, quiet=False)

    # Descomprimir
    print("Descomprimiendo...")
    with zipfile.ZipFile(OUTPUT_ZIP, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_FOLDER)
    
    os.remove(OUTPUT_ZIP)
    print(" Listo.")
else:
    print(" Dataset ya existe, no se descarga de nuevo.")