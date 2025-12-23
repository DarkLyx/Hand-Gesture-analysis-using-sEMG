from tensorflow.keras import layers, models, Input #type: ignore
from tensorflow.keras.applications import VGG16 #type: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input #type: ignore

def create_vgg_adapter_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = layers.Concatenate(axis=-1)([inputs, inputs, inputs])

    # Resize compatible VGG
    x = layers.Resizing(224, 224)(x)

    # Preprocessing ImageNet
    x = preprocess_input(x)

    # Backbone
    base_model = VGG16(include_top=False, weights='imagenet', input_tensor=x)
    base_model.trainable = False

    # Classification head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="VGG_sEMG")
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
