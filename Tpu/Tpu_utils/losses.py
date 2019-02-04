import tensorflow as tf

def loss_func(logits_in, labels_in):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_in, labels=labels_in))



def loss(labels, source_logits_real, class_logits_real, source_logits_fake,
         class_logits_fake,n_classes):

    labels_one_hot = tf.one_hot(labels, n_classes)

    source_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            source_logits_real, tf.ones_like(source_logits_real)))

    source_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            source_logits_fake, tf.zeros_like(source_logits_fake)))

    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            source_logits_fake, tf.ones_like(source_logits_fake)))

    class_loss_real = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(class_logits_real,
                                                labels_one_hot))
    class_loss_fake = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(class_logits_fake,
                                                labels_one_hot))

    d_loss = source_loss_real + source_loss_fake + class_loss_real + class_loss_fake

    g_loss =  g_loss + class_loss_real + class_loss_fake

    return d_loss, g_loss