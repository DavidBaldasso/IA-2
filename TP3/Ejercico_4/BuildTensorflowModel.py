import subprocess
try:
    import tensorflow as tf
except ImportError:
    subprocess.check_call(['pip', 'install', 'tensorflow'])
    subprocess.check_call(['pip', 'install', 'Pillow'])
    import tensorflow as tf

# Intentar importar matplotlib; si no existe, instalarlo
try:
    import matplotlib.pyplot as plt
except ImportError:
    subprocess.check_call(['pip', 'install', 'matplotlib'])
    import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import random

# ——————————————————————————————————————————————————————————————————————————
# Paths de carpetas: “raw” (up, down, right), y subdirectorios train/ y test/
# ——————————————————————————————————————————————————————————————————————————
source_dir = "images"
train_dir  = os.path.join(source_dir, "train")
test_dir   = os.path.join(source_dir, "test")

# ——————————————————————————————————————————————————————————————————————————
# Clases: nombres EXACTOS de las carpetas en minúsculas
# ——————————————————————————————————————————————————————————————————————————
classes = ["jump", "duck", "right"]

# ——————————————————————————————————————————————————————————————————————————
# Crear (si no existen) train/<clase>/ y test/<clase>/
# ——————————————————————————————————————————————————————————————————————————
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir,  exist_ok=True)
for c in classes:
    os.makedirs(os.path.join(train_dir,  c), exist_ok=True)
    os.makedirs(os.path.join(test_dir,   c), exist_ok=True)

# ——————————————————————————————————————————————————————————————————————————
# Proporción para split train/test
# ——————————————————————————————————————————————————————————————————————————
train_ratio = 0.8

# ——————————————————————————————————————————————————————————————————————————
# Parámetros de entrada
# ——————————————————————————————————————————————————————————————————————————
batch_size = 32
image_size = (224, 224)         # (ancho, alto)
input_shape = image_size + (1,) # (224, 224, 1)

def load_and_preprocess_image(file_path, target_size):
    """
    Carga la imagen en grayscale, la redimensiona a 'target_size' (224×224),
    la convierte en array y la devuelve SIN normalizar (ImageDataGenerator hará el rescale).
    """
    img = load_img(file_path, color_mode='grayscale', target_size=target_size)
    arr = img_to_array(img)  # shape = (alto, ancho, 1), valores 0..255
    return arr

# ——————————————————————————————————————————————————————————————————————————
# Repartir carpetas crudas → train/ y test/
# ——————————————————————————————————————————————————————————————————————————
for class_name in classes:
    source_class_dir = os.path.join(source_dir, class_name)
    if not os.path.isdir(source_class_dir):
        print(f"[WARNING] Falta carpeta de origen: {source_class_dir}")
        continue

    all_images = os.listdir(source_class_dir)
    random.shuffle(all_images)
    num_train   = int(len(all_images) * train_ratio)

    # Enviar primer 80% a train/<clase>/
    for img_name in all_images[:num_train]:
        src  = os.path.join(source_class_dir, img_name)
        dst  = os.path.join(train_dir, class_name, img_name)
        arr  = load_and_preprocess_image(src, image_size)
        tf.keras.preprocessing.image.save_img(dst, arr)

    # Enviar restante 20% a test/<clase>/
    for img_name in all_images[num_train:]:
        src  = os.path.join(source_class_dir, img_name)
        dst  = os.path.join(test_dir, class_name, img_name)
        arr  = load_and_preprocess_image(src, image_size)
        tf.keras.preprocessing.image.save_img(dst, arr)

# ——————————————————————————————————————————————————————————————————————————
# Generadores: rescale=1./255 (solo una normalización)
# ——————————————————————————————————————————————————————————————————————————
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,       # (224, 224)
    batch_size=batch_size,
    classes=classes,              # ["up","down","right"]
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=True
)
print("Mapa de clases (train):", train_generator.class_indices)
# Ejemplo: {'down': 0, 'right': 1, 'up': 2}

validation_datagen = ImageDataGenerator(rescale=1.0/255.0)
validation_generator = validation_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    classes=classes,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=False
)
print("Validation samples:", validation_generator.samples)

# Diccionario para convertir índice numérico → etiqueta en mayúsculas
idx_to_label = {
    train_generator.class_indices['jump']:    "JUMP",
    train_generator.class_indices['duck']:  "DUCK",
    train_generator.class_indices['right']: "RIGHT"
}

# ========================== Construcción de la CNN ==========================================
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,  (3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64,  (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(len(classes), activation='softmax')
])
# ============================================================================================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks: parada temprana y guardar mejor modelo
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
]

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=callbacks
)

model.save('tensorflow_nn.h5')
print("Modelo entrenado y guardado en 'tensorflow_nn.h5'.")

def plot_training_curves(history):
    epochs = range(1, len(history.history['loss']) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['loss'], 'b-', label='Training loss')
    plt.plot(epochs, history.history['val_loss'], 'r-', label='Validation loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['accuracy'], 'b-', label='Training accuracy')
    plt.plot(epochs, history.history['val_accuracy'], 'r-', label='Validation accuracy')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

# Mostrar curvas de entrenamiento (opcional)
plot_training_curves(history)
