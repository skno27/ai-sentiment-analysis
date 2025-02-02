import tensorflow as tf
import re
import string


@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape(string.punctuation), ""
    )


sentiment_model = "./SentimentBeta.keras"
loaded_model = tf.keras.models.load_model(
    sentiment_model, custom_objects={"custom_standardization": custom_standardization}
)


def analyze(comment):
    if isinstance(comment, str):
        comment = tf.convert_to_tensor([comment])

    prediction = loaded_model.predict(comment)

    return prediction
