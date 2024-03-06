import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import os

# Инициализация модели для извлечения признаков
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')

def preprocess_img(img_path):
    """Функция для предобработки изображения."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

def extract_features(img_path, model):
    """Функция для извлечения признаков из изображения."""
    preprocessed_img = preprocess_img(img_path)
    features = model.predict(preprocessed_img)
    return features

def compare_images(user_img_path, db_images_paths, model):
    """Функция для сравнения изображения пользователя с изображениями из БД."""
    user_features = extract_features(user_img_path, model)
    
    similarities = {}
    for db_img_path in db_images_paths:
        db_features = extract_features(db_img_path, model)
        similarity = cosine_similarity(user_features, db_features)
        similarities[db_img_path] = similarity
    
    return similarities

# Загрузите пути к изображениям в БД здесь
db_images_paths = ['path_to_db_image1.jpg', 'path_to_db_image2.jpg', ...]

# Путь к изображению, загруженному пользователем
user_img_path = 'path_to_user_image.jpg'

# Сравнение изображения пользователя с изображениями из БД
similarities = compare_images(user_img_path, db_images_paths, model)

# Отбор изображений с высокой степенью сходства
threshold = 0.8  # Значение порога сходства
similar_images = [db_img for db_img, sim in similarities.items() if sim >= threshold]

print("Схожие изображения:", similar_images)

