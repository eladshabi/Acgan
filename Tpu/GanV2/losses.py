import tensorflow as tf

def loss_func(logits_in, labels_in):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_in, labels=labels_in))



def loss(labels, source_logits_real, class_logits_real, source_logits_fake,
         class_logits_fake,generated_images):


    print(labels)

    labels_one_hot = tf.one_hot(labels, 10)

    source_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=source_logits_real, labels=tf.ones_like(source_logits_real)))

    source_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=source_logits_fake, labels = tf.zeros_like(source_logits_fake)))

    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=source_logits_fake, labels = tf.ones_like(source_logits_fake)))

    class_loss_real = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=class_logits_real,
                                                labels=labels_one_hot))
    class_loss_fake = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=class_logits_fake,
                                                labels=labels_one_hot))

    d_loss = source_loss_real + source_loss_fake + class_loss_real + class_loss_fake

    g_loss =  g_loss + class_loss_real + class_loss_fake

    return d_loss, g_loss