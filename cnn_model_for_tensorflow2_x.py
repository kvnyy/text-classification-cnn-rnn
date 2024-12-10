import tensorflow as tf


class TCNNConfig(object):
    """CNN配置参数"""
    embedding_dim = 64  # 词向量维度
    seq_length = 600  # 序列长度
    num_classes = 10  # 类别数
    num_filters = 256  # 卷积核数目
    kernel_size = 5  # 卷积核尺寸
    vocab_size = 5000  # 词汇表达小

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard


class TextCNN(tf.keras.Model):
    """文本分类，CNN模型"""

    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.config = config

        # 词向量层
        self.embedding = tf.keras.layers.Embedding(
            input_dim=config.vocab_size,
            output_dim=config.embedding_dim,
            input_length=config.seq_length,
            name="embedding"
        )

        # 卷积层
        self.conv = tf.keras.layers.Conv1D(
            filters=config.num_filters,
            kernel_size=config.kernel_size,
            activation='relu',
            name="conv"
        )

        # 全局最大池化层
        self.global_max_pool = tf.keras.layers.GlobalMaxPooling1D(name="gmp")

        # 全连接层
        self.fc1 = tf.keras.layers.Dense(
            units=config.hidden_dim,
            activation='relu',
            name="fc1"
        )
        self.dropout = tf.keras.layers.Dropout(rate=1 - config.dropout_keep_prob)

        # 输出层
        self.fc2 = tf.keras.layers.Dense(
            units=config.num_classes,
            activation=None,  # 线性输出，用于交叉熵计算
            name="fc2"
        )

    def call(self, inputs, training=False):
        # 前向传播
        x = self.embedding(inputs)
        x = self.conv(x)
        x = self.global_max_pool(x)
        x = self.fc1(x)
        if training:
            x = self.dropout(x, training=training)
        logits = self.fc2(x)
        return logits

    def get_config(self):
        """返回模型的配置字典，用于保存和加载模型"""
        return {
            "config": self.config.__dict__
        }

    @classmethod
    def from_config(cls, config, **kwargs):
        """从配置字典中重建模型实例"""
        # 将配置字典重新转为 TCNNConfig 实例
        config_obj = TCNNConfig()
        config_obj.__dict__.update(config["config"])
        return cls(config=config_obj, **kwargs)


