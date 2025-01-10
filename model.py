# From the TensorFlow.org Docs Tutorial @ https://www.tensorflow.org/tutorials/keras/text_classification#download_and_explore_the_imdb_dataset

import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

# these two imports throw an error cause tensorflow.keras is "loaded lazily"
# tell your ide to ignore, if possible. it works fine

from tensorflow.keras import layers  # type: ignore
from tensorflow.keras import losses  # type: ignore


dataset_dir = "./aclImdb"

train_dir = os.path.join(dataset_dir, "train")

sample_file = os.path.join(train_dir, "pos/1181_9.txt")

"""

here an extra directory is removed in order to format the training directory into a binary format

remove_dir = os.path.join(train_dir, "unsup")

shutil.rmtree(remove_dir)

"""


batch_size = 32
seed = 42

"""
this expects the binary directory format where one of the subdirectories is a selection and the other is its opposite
we will use 80% of the data for training, leaving 20% of the data for validation


"""

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=seed,
    shuffle=True,
)
"""

# test to preview how the dataset is used for training

for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(3):
        print("Review", text_batch.numpy()[i])
        print("Label", label_batch.numpy()[i])

print("Label 0 corresponds to", raw_train_ds.class_names[0])
print("Label 1 corresponds to", raw_train_ds.class_names[1])

here we learn that label 0 corresponds to negative
and that label 1 corresponds to positive
"""

# now we will create a validation and test dataset using the remaining 5k reviews

raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    shuffle=False,
)

# Note: at this point, I set the shuffle parameter to false for val datasets to make sure it stays the same,
# while training data will keep changing to avoid false pattern recognition

raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    "./aclImdb/test", batch_size=batch_size
)


"""
    Now we need to prepare the dataset for training

    We will standardize (remove punctuation or html elements), 
    tokenize (split the string into tokens, like a sentence into words, splitting on the whispace), 
    and vectorize (converts tokens into numbers so to be received by a Neural Network) the data using the keras TextVectorization layer
"""

# we need custom standardization because keras TextVectorization doesnt handle remove html, it just makes texts lowercase and strips punctuation


@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape(string.punctuation), ""
    )


# define the text vectorization layer


max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode="int",  # this will create unique intergers to represent each token
    output_sequence_length=sequence_length,
)

print("New dataset without labels...")

train_text = raw_train_ds.map(
    lambda x, y: x
)  # creating a text only dataset without layers

print("Adapting new dataset to the vectorization state...")
vectorize_layer.adapt(
    train_text
)  # passing the training text to the vectorize layer to adapt the data to fit the preprocessing layer definition


# function to see the result of the processing with labels
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


print("Retrieving a batch from the cleaned dataset...")
text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print("Review:", first_review)
print("Label:", raw_train_ds.class_names[first_label])
print("Vectorized review:", vectorize_text(first_review, first_label))

# we can look up the token-string to integer association with .get_vocabulary method on the layer

print("65 ---> ", vectorize_layer.get_vocabulary()[65])
print("Vocabulary size: {}".format(len(vectorize_layer.get_vocabulary())))

# we are just about ready to train the model =========

train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

"""
we can make performance configurations with .cache() (keeps data in memory after
its loaded off disk) and with .prefetch() (overlaps data preprocessing and model
execution while training)

if the dataset is too large to fit into memory, .cache can be used to create an on-
disk cache for more efficient reading than many small files.

further documentation on both methods and caching data @ https://www.tensorflow.org/guide/data_performance

"""

AUTOTUNE = tf.data.AUTOTUNE  # tunes this value dynamically at runtime

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ----------------------------------------------------------------

# Creating the neural network

embedding_dim = 16

model = tf.keras.Sequential(
    [
        layers.Embedding(max_features, embedding_dim),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling1D(),
        # layers.Bidirectional(layers.LSTM(32)),
        # layers.Dense(32, activation="relu"),
        # layers.Dropout(0.8),
        layers.Dense(1, activation="sigmoid"),
    ]
)

model.summary()

"""
-   The embedding layer takes the integer encoded review and looks up an embedding vector for each word-index
    . These vectors are learned as trained. The vector add a deimension to the output array, the resulting
    dimensions are (batch, sequence, embedding).
    
-   the GlobalAveragePooling1D layer returns a fixed-length output vector for each example by averaging over
    the sequence dimension. This allows the model to handle input of variable length, in the simples way 
    possible.
    
-   The last layer is densely connected with a single output node.

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ embedding (Embedding)           │ ?                      │   0 (unbuilt) │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ ?                      │   0 (unbuilt) │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ global_average_pooling1d        │ ?                      │   0 (unbuilt) │
│ (GlobalAveragePooling1D)        │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (Dropout)             │ ?                      │   0 (unbuilt) │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ ?                      │   0 (unbuilt) │
└─────────────────────────────────┴────────────────────────┴───────────────┘

"""

# a model needs a loss function and an optimizer for training.

model.compile(
    loss=losses.BinaryCrossentropy(),
    optimizer="adam",
    metrics=[tf.metrics.BinaryAccuracy(threshold=0.5)],
)

# since this is a binary classification problem and the model outputs
# a probability (single-unit layer with a sigmoid activation),
# we use losses.BinaryCrossEntropy loss function

# training the model

epochs = 50

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)
history = model.fit(
    train_ds, validation_data=val_ds, epochs=epochs, callbacks=[early_stopping]
)

loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)


# plot the loss and accuracy over time

history_dict = history.history
history_dict.keys()

acc = history_dict["binary_accuracy"]
val_acc = history_dict["val_binary_accuracy"]
loss = history_dict["loss"]
val_loss = history_dict["val_loss"]

epochs = range(1, len(acc) + 1)
"""
train loss and val loss (epoch x loss y)
    # 'bo' for blue dot
    plt.plot(epochs, loss, "bo", label="Training loss")
    # 'b' for solid blue line
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()
"""

"""
train loss and val loss (epoch x acc y)
    plt.plot(epochs, acc, "bo", label="Training acc")
    plt.plot(epochs, val_acc, "b", label="Validation acc")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")

    plt.show()
"""

if not os.path.exists("accuracy.txt"):
    with open("accuracy.txt", "w") as f:
        f.write(str(accuracy))

    export_model = tf.keras.Sequential(
        [vectorize_layer, model, layers.Activation("sigmoid")]
    )

    export_model.compile(
        loss=losses.BinaryCrossentropy(from_logits=False),
        optimizer="adam",
        metrics=["accuracy"],
    )

    metrics = export_model.evaluate(raw_test_ds, return_dict=True)
    print("Saving Updated Model---")
    export_model.save("SentimentBeta.keras", include_optimizer=False)
    print("Metrics:\n", metrics)


else:

    with open("accuracy.txt", "r+") as f:
        acc_content = float(f.read())

        if acc_content < accuracy:
            f.seek(0)
            f.write(str(accuracy))
            f.truncate()

            export_model = tf.keras.Sequential(
                [vectorize_layer, model, layers.Activation("sigmoid")]
            )

            export_model.compile(
                loss=losses.BinaryCrossentropy(from_logits=False),
                optimizer="adam",
                metrics=["accuracy"],
            )

            metrics = export_model.evaluate(raw_test_ds, return_dict=True)
            print("Saving Updated Model---")
            export_model.save("SentimentBeta.keras", include_optimizer=False)

            print("Metrics:\n", metrics)
