import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer

class DeformableSentenceSplit(Layer):
    def __init__(self, num_sentences, max_sentence_length, embedding_dim, **kwargs):
        super(DeformableSentenceSplit, self).__init__(**kwargs)
        self.num_sentences = num_sentences
        self.max_sentence_length = max_sentence_length
        self.embedding_dim = embedding_dim
        self.offset_dense = Dense(units=2 * num_sentences)

    def build(self, input_shape):
        super(DeformableSentenceSplit, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        print("Input shape:", tf.shape(inputs))
        batch_size = tf.shape(inputs)[0]
        text_length = tf.shape(inputs)[1]

        offsets = self.offset_dense(tf.reduce_mean(inputs, axis=1))
        start_offsets = offsets[:, :self.num_sentences]
        end_offsets = offsets[:, self.num_sentences:]

        start_offsets = tf.clip_by_value(start_offsets, 0.0, tf.cast(self.max_sentence_length - 1, tf.float32))
        end_offsets = tf.clip_by_value(end_offsets, 0.0, tf.cast(self.max_sentence_length - 1, tf.float32))

        start_offsets = tf.cast(start_offsets, tf.int32)
        end_offsets = tf.cast(end_offsets, tf.int32)

        output = tf.TensorArray(inputs.dtype, size=batch_size)

        for b in tf.range(batch_size):
            sentences = tf.TensorArray(inputs.dtype, size=self.num_sentences)
            for i in range(self.num_sentences):
                base_start_index = i * self.max_sentence_length
                base_end_index = base_start_index + self.max_sentence_length

                start_index = tf.clip_by_value(base_start_index + start_offsets[b, i], 0, text_length - self.max_sentence_length)
                end_index = tf.clip_by_value(base_end_index + end_offsets[b, i], start_index, text_length)

                sentence = inputs[b, start_index:end_index, :]
                sentence_length = tf.shape(sentence)[0]
                padded_sentence = tf.pad(sentence, [[0, self.max_sentence_length - sentence_length], [0, 0]])
                sentences = sentences.write(i, padded_sentence)
            output = output.write(b, sentences.stack())

        result = output.stack()
        print("Output shape:", tf.shape(result))
        return tf.reshape(result, [batch_size, self.num_sentences, self.max_sentence_length, self.embedding_dim])

    def compute_output_shape(self, input_shape):
        return (None, self.num_sentences, self.max_sentence_length, self.embedding_dim)
