import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from ultralytics import YOLO

DATASET_PATH = r'C:\Users\su_20\OneDrive\Documentos\Mestrado\Visao computacional\Face detection YOLO'  
CHECKPOINT_DIR = r'C:\Users\su_20\OneDrive\Documentos\Mestrado\Visao computacional\Face detection YOLO\checkpoints'    
EXPERIMENT_NAME = 'yolo_face_reduzido'                    
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_EPOCHS = 3
BATCH_SIZE = 8

# Criação de diretório de projeto
project_dir = CHECKPOINT_DIR
name = EXPERIMENT_NAME

train_images_dir = os.path.join(DATASET_PATH, 'images', 'train')
val_images_dir = os.path.join(DATASET_PATH, 'images', 'val')

# Criação do arquivo de configuração YAML  
dataset_yaml_path = os.path.join(DATASET_PATH, 'dataset.yaml')
dataset_yaml_content = f"""
path: {DATASET_PATH}
train: images/train
val: images/val
names: ['face']
"""

with open(dataset_yaml_path, 'w') as f:
    f.write(dataset_yaml_content)

# Carregar o modelo 
model = YOLO('yolov8n.pt') 

# Treinar o modelo
model.train(
    data=dataset_yaml_path,
    epochs=MAX_EPOCHS,
    batch=BATCH_SIZE,
    imgsz=640,    
    device=DEVICE,
    project=project_dir,
    name=name,
    exist_ok=True
)

# Carregar o melhor modelo treinado
best_model_path = os.path.join(project_dir, name, 'weights', 'best.pt')
best_model = YOLO(best_model_path)

# Fazer uma predição de teste
img_1 = os.path.join(val_images_dir, os.listdir(val_images_dir)[0])
results = best_model.predict(source=img_1, save=False, show=True, device=DEVICE)