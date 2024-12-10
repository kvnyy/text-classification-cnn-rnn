import tensorflow as tf


class TRNNConfig(object):
    """RNN配置参数"""
    embedding_dim = 64  # 词向量维度
    seq_length = 600  # 序列长度
    num_classes = 10  # 类别数
    vocab_size = 5000  # 词汇表大小

    num_layers = 2  # 隐藏层层数
    hidden_dim = 128  # 隐藏层神经元
    rnn = 'gru'  # lstm 或 gru

    dropout_keep_prob = 0.8  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 32  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard


class TextRNN(tf.keras.Model):
    """文本分类，RNN模型"""

    def __init__(self, config):
        super(TextRNN, self).__init__()
        self.config = config

        # 词向量层
        self.embedding = tf.keras.layers.Embedding(input_dim=self.config.vocab_size, output_dim=self.config.embedding_dim)


        # RNN 层
        self.rnn_cells = []
        for _ in range(self.config.num_layers):
            if self.config.rnn == 'lstm':
                self.rnn_cells.append(tf.keras.layers.LSTM(self.config.hidden_dim, return_sequences=True,
                                                           dropout=1 - self.config.dropout_keep_prob))
            else:
                self.rnn_cells.append(tf.keras.layers.GRU(self.config.hidden_dim, return_sequences=True,
                                                          dropout=1 - self.config.dropout_keep_prob))

        # Dropout层
        self.dropout = tf.keras.layers.Dropout(1 - self.config.dropout_keep_prob)

        # 全连接层
        self.fc1 = tf.keras.layers.Dense(self.config.hidden_dim, activation='relu')
        self.fc2 = tf.keras.layers.Dense(self.config.num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.embedding(inputs)

        # RNN层
        for rnn_cell in self.rnn_cells:
            x = rnn_cell(x)

        # 取最后一个时序输出
        x = x[:, -1, :]  # 只取最后一个时序输出

        # 全连接层
        x = self.fc1(x)
        x = self.dropout(x, training=training)  # 应用Dropout
        output = self.fc2(x)

        return output

    def compute_loss(self, y_true, y_pred):
        return tf.reduce_mean(tf.losses.categorical_crossentropy(y_true, y_pred))

    def compute_accuracy(self, y_true, y_pred):
        y_pred_cls = tf.argmax(y_pred, axis=1)
        y_true_cls = tf.argmax(y_true, axis=1)
        return tf.reduce_mean(tf.cast(tf.equal(y_true_cls, y_pred_cls), tf.float32))

    def get_config(self):
        """返回模型的配置字典，用于保存和加载模型"""
        return {
            "config": self.config.__dict__
        }

    @classmethod
    def from_config(cls, config, **kwargs):
        """从配置字典中重建模型实例"""
        # 将配置字典重新转为 TCNNConfig 实例
        config_obj = TRNNConfig()
        config_obj.__dict__.update(config["config"])
        return cls(config=config_obj, **kwargs)

