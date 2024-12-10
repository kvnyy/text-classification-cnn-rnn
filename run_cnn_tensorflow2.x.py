import os
import sys
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from sklearn import metrics

from cnn_model_for_tensorflow2_x import TCNNConfig, TextCNN
from data.cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab

base_dir = 'data/cnews'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation.keras')  # 最佳验证结果保存路径


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def evaluate(model, data, loss_object):
    """评估在某一数据上的准确率和损失"""
    x_data, y_data = data
    data_len = len(x_data)
    batch_eval = batch_iter(x_data, y_data, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        predictions = model(x_batch, training=False)
        loss = loss_object(y_batch, predictions)
        acc = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y_batch, predictions))
        total_loss += loss.numpy() * len(x_batch)
        total_acc += acc.numpy() * len(x_batch)

    return total_loss / data_len, total_acc / data_len


def train():
    print("Configuring TensorBoard and Saver...")
    # 配置 TensorBoard
    tensorboard_dir = 'tensorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    writer = tf.summary.create_file_writer(tensorboard_dir)

    print("Loading training and validation data...")
    start_time = time.time()
    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
    x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建模型和优化器
    model = TextCNN(config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    print('Training and evaluating...')
    total_batch = 0
    best_acc_val = 0.0
    last_improved = 0
    require_improvement = 1000

    for epoch in range(config.num_epochs):
        print(f'Epoch: {epoch + 1}')
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            with tf.GradientTape() as tape:
                predictions = model(x_batch, training=True)
                loss = loss_object(y_batch, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            if total_batch % config.print_per_batch == 0:
                acc_train = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y_batch, predictions))
                loss_val, acc_val = evaluate(model, (x_val, y_val), loss_object)

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    last_improved = total_batch
                    model.save(save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = f'Iter: {total_batch}, Train Loss: {loss:.2f}, Train Acc: {acc_train:.2%},' \
                      f' Val Loss: {loss_val:.2f}, Val Acc: {acc_val:.2%}, Time: {time_dif} {improved_str}'
                print(msg)

            total_batch += 1

            if total_batch - last_improved > require_improvement:
                print("No optimization for a long time, auto-stopping...")
                model.save(save_path)
                return



def test():
    print("Loading test data...")
    start_time = time.time()
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.seq_length)

    print("Loading saved model...")
    model = tf.keras.models.load_model(
        save_path,
        custom_objects={"TextCNN": TextCNN}
    )

    print('Testing...')
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    loss_test, acc_test = evaluate(model, (x_test, y_test), loss_object)
    msg = f'Test Loss: {loss_test:.2f}, Test Acc: {acc_test:.2%}'
    print(msg)

    batch_size = 128
    y_test_cls = np.argmax(y_test, axis=1)
    y_pred_cls = []

    for start_id in range(0, len(x_test), batch_size):
        end_id = min(start_id + batch_size, len(x_test))
        x_batch = x_test[start_id:end_id]
        predictions = model(x_batch, training=False)
        y_pred_cls.extend(tf.argmax(predictions, axis=1).numpy())

    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


if __name__ == '__main__':
    print('Configuring CNN model...')
    config = TCNNConfig()
    if not os.path.exists(vocab_dir):
        build_vocab(train_dir, vocab_dir, config.vocab_size)
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)
    config.vocab_size = len(words)
    model2 = TextCNN(config)
    # 保存模型
    # model2.save("textcnn_model.keras")
    #
    # # 加载模型
    # loaded_model = tf.keras.models.load_model(
    #     "textcnn_model.keras",
    #     custom_objects={"TextCNN": TextCNN}
    # )

    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("""usage: python run_cnn.py [train / test]""")

    if sys.argv[1] == 'train':
        train()
    else:
        test()
