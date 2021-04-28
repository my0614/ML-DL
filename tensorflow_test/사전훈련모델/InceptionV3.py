import tensorflow_hub as hub
import tensorflow as tf
inception_url = 'https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4'
feature_model = tf.keras.Sequential([
   hub.KerasLayer(inception_url, output_shape = (2048,), trainable = False)
])
feature_model.build([None, 299,299,3])
feature_model.summary()
