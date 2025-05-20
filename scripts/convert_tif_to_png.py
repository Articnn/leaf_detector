from PIL import Image
import os
import glob

ROOT = "./data"
EXT_IN = ".tif"
EXT_OUT = ".png"

for folder in [f"leaf{i}" for i in range(1, 8)]:
    path = os.path.join(ROOT, folder)
    tif_files = glob.glob(os.path.join(path, f"*{EXT_IN}"))
    
    print(f"🖼️ Convirtiendo {len(tif_files)} imágenes en {folder}...")

    for file in tif_files:
        try:
            img = Image.open(file)
            new_path = file.replace(EXT_IN, EXT_OUT)
            img.save(new_path)
            os.remove(file)
        except Exception as e:
            print(f"⚠️ Error con {file}: {e}")

print("✅ Conversión completada: .tif → .png")