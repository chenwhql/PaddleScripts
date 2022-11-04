import warnings
warnings.filterwarnings('ignore')

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from Encoder import CNNEncoder

wide_size = 82
deep_size = 82
seq_embed_size = 26
common_size = 49

class RankNet(paddle.nn.Layer):
    def __init__(self, seq_len_size, num_filter = 16, ngram_filter_sizes = (3,5,11)):
        super(RankNet, self).__init__()
        ###### deep layer ######
        self.deep_fc_1 = paddle.nn.Linear(in_features=deep_size, out_features=128,
                                            weight_attr=paddle.ParamAttr(name="deep_fc_w1"),
                                            bias_attr=paddle.ParamAttr(name="deep_fc_b1"))
        self.deep_fc_2 = paddle.nn.Linear(in_features=128, out_features=64,
                                            weight_attr=paddle.ParamAttr(name="deep_fc_w2"),
                                            bias_attr=paddle.ParamAttr(name="deep_fc_b2"))
        self.deep_fc_3 = paddle.nn.Linear(in_features=64, out_features=32,
                                            weight_attr=paddle.ParamAttr(name="deep_fc_w3"),
                                            bias_attr=paddle.ParamAttr(name="deep_fc_b3"))
        self.deep_bn_1 = paddle.nn.BatchNorm(num_channels=128,
                                    param_attr=paddle.ParamAttr(name="deep_norm_w1"),
                                    bias_attr=paddle.ParamAttr(name="deep_norm_b1"))
        self.deep_bn_2 = paddle.nn.BatchNorm(num_channels=64,
                                    param_attr=paddle.ParamAttr(name="deep_norm_w2"),
                                    bias_attr=paddle.ParamAttr(name="deep_norm_b2"))
        self.deep_bn_3 = paddle.nn.BatchNorm(num_channels=32,
                                    param_attr=paddle.ParamAttr(name="deep_norm_w3"),
                                    bias_attr=paddle.ParamAttr(name="deep_norm_b3"))
        self.deep_dropout_1 = paddle.nn.Dropout(0.3)
        self.deep_relu_1 = paddle.nn.ReLU()
        self.deep_relu_2 = paddle.nn.ReLU()
        self.deep_relu_3 = paddle.nn.ReLU()
        
        ###### wide layer ######
        self.wide_fc_1 = paddle.nn.Linear(in_features=wide_size, out_features=1,
                                            weight_attr=paddle.ParamAttr(name="wide_fc_w1"),
                                            bias_attr=paddle.ParamAttr(name="wide_fc_b1"))

        ###### seq layer ######
        self.seq_len_size = seq_len_size
        self.ngram_filter_sizes = ngram_filter_sizes
        self.num_filter = num_filter
        self.seq_encoder = CNNEncoder(emb_dim=seq_embed_size, num_filter=self.num_filter, ngram_filter_sizes=self.ngram_filter_sizes)
        self.seq_fc = nn.Linear(in_features=self.seq_encoder.get_output_dim(), out_features=16,
                                weight_attr=paddle.ParamAttr(name="seq_fc_w1"),
                                bias_attr=paddle.ParamAttr(name="seq_fc_b1"))
        self.seq_dropout_1 = paddle.nn.Dropout(0.3)


        ###### common layer ######
        self.com_fc_1 = paddle.nn.Linear(in_features=common_size, out_features=64,
                                            weight_attr=paddle.ParamAttr(name="com_fc_w1"),
                                            bias_attr=paddle.ParamAttr(name="com_fc_b1"))
        self.com_fc_2 = paddle.nn.Linear(in_features=64, out_features=32,
                                            weight_attr=paddle.ParamAttr(name="com_fc_w2"),
                                            bias_attr=paddle.ParamAttr(name="com_fc_b2"))
        self.com_fc_3 = paddle.nn.Linear(in_features=32, out_features=1,
                                            weight_attr=paddle.ParamAttr(name="com_fc_w3"),
                                            bias_attr=paddle.ParamAttr(name="com_fc_b3"))
        self.com_bn_1 = paddle.nn.BatchNorm(num_channels=64,
                                    param_attr=paddle.ParamAttr(name="com_norm_w1"),
                                    bias_attr=paddle.ParamAttr(name="com_norm_b1"))
        self.com_bn_2 = paddle.nn.BatchNorm(num_channels=32,
                                    param_attr=paddle.ParamAttr(name="com_norm_w2"),
                                    bias_attr=paddle.ParamAttr(name="com_norm_b2"))
        self.com_relu_1 = paddle.nn.ReLU()
        self.com_relu_2 = paddle.nn.ReLU()

        ###### position encoding ######
        self.PE = self.position_encoding(seq_embed_size, self.seq_len_size)

        ###### preprocess feature ######
        self.emb_tm_wday = paddle.nn.Embedding(num_embeddings=7, embedding_dim=2, weight_attr=paddle.ParamAttr(name="emb_tm_wday_w1"))
        self.emb_tm_hour = paddle.nn.Embedding(num_embeddings=24, embedding_dim=4, weight_attr=paddle.ParamAttr(name="emb_tm_hour_w1"))
        self.emb_tm_mins = paddle.nn.Embedding(num_embeddings=4, embedding_dim=2, weight_attr=paddle.ParamAttr(name="emb_tm_mins_w1"))
        self.emb_length_bin = paddle.nn.Embedding(num_embeddings=7, embedding_dim=2, weight_attr=paddle.ParamAttr(name="emb_length_bin_w1"))

        ###### seq feature embedding #####
        self.emb_seq2id = paddle.nn.Embedding(num_embeddings=7501, embedding_dim=8, weight_attr=paddle.ParamAttr(name="emb_seq2id_w1"))
        self.emb_status = paddle.nn.Embedding(num_embeddings=6, embedding_dim=2, weight_attr=paddle.ParamAttr(name="emb_status_w1"))
        self.emb_status_road_level = paddle.nn.Embedding(num_embeddings=51, embedding_dim=4, weight_attr=paddle.ParamAttr(name="emb_status_road_level_w1"))
        self.emb_status_width = paddle.nn.Embedding(num_embeddings=21, embedding_dim=4, weight_attr=paddle.ParamAttr(name="emb_status_width_w1"))
        self.emb_status_path_class = paddle.nn.Embedding(num_embeddings=31, embedding_dim=4, weight_attr=paddle.ParamAttr(name="emb_status_path_class_w1"))
        self.emb_status_pred_status = paddle.nn.Embedding(num_embeddings=26, embedding_dim=4, weight_attr=paddle.ParamAttr(name="emb_status_pred_status_w1"))

    def position_encoding(self, d_pe, max_len):
        PE = paddle.zeros(shape=[max_len, d_pe])
        position = paddle.arange(0, max_len).unsqueeze(1).astype("float32")
        super_pos = paddle.multiply(paddle.arange(0, d_pe, 2).astype("float32"),
                                        paddle.to_tensor(-(math.log(10000.0)/d_pe)))
        div_term = paddle.exp(super_pos)
        PE[:, 0::2] = paddle.sin(paddle.multiply(position, div_term))
        PE[:, 1::2] = paddle.cos(paddle.multiply(position, div_term))
        return PE

    def seq_layer(self, seq_features):
        # pe = self.PE[:seq_features.shape[1], :].unsqueeze(0).clone()
        # tile_pe = paddle.tile(pe,repeat_times=[seq_features.shape[0],1,1])
        # print("seq_features:", seq_features.dtype, seq_features.shape)
        # print("pe:", pe.dtype, pe.shape)
        # print("tile_pe:", tile_pe.dtype, tile_pe.shape)
        # feature_add_pos = paddle.add(seq_features, tile_pe)
        encoder_out = self.seq_encoder(seq_features)
        encoder_out = paddle.tanh(encoder_out)
        seq_out = self.seq_fc(encoder_out)
        seq_out = self.seq_dropout_1(seq_out)
        return seq_out

    def deep_layer(self, deep_features):
        deep_hidden1 = self.deep_fc_1(deep_features)
        deep_hidden1 = self.deep_bn_1(deep_hidden1)
        # deep_hidden1 = F.dropout(deep_hidden1, 0.5)
        # deep_hidden1 = F.relu(deep_hidden1)
        deep_hidden1 = self.deep_dropout_1(deep_hidden1)
        deep_hidden1 = self.deep_relu_1(deep_hidden1)


        deep_hidden2 = self.deep_fc_2(deep_hidden1)
        deep_hidden2 = self.deep_bn_2(deep_hidden2)
        # deep_hidden2 = F.relu(deep_hidden2)
        deep_hidden2 = self.deep_relu_2(deep_hidden2)

        deep_hidden3 = self.deep_fc_3(deep_hidden2)
        deep_hidden3 = self.deep_bn_3(deep_hidden3)
        # deep_out = F.relu(deep_hidden3)
        deep_out = self.deep_relu_3(deep_hidden3)
        return deep_out

    def wide_layer(self, wide_features):
        wide_out = self.wide_fc_1(wide_features)
        return wide_out

    # 特征预处理
    def format_data(self, inputs, seq_input):
        # 分离关键特征
        key_features_1 = paddle.slice(input=inputs, axes=[1], starts=[0], ends=[2])
        key_features_1 = paddle.cast(key_features_1, dtype="float32")
        key_features_2 = paddle.slice(input=inputs, axes=[1], starts=[3], ends=[6])
        key_features_2 = paddle.cast(key_features_2, dtype="float32")
        key_features_3 = paddle.slice(input=inputs, axes=[1], starts=[7], ends=[12])
        key_features_3 = paddle.cast(key_features_3, dtype="float32")
        key_features_4 = paddle.slice(input=inputs, axes=[1], starts=[13], ends=[17])
        key_features_4 = paddle.cast(key_features_4, dtype="float32")
        key_features_5 = paddle.slice(input=inputs, axes=[1], starts=[21], ends=[29])
        key_features_5 = paddle.cast(key_features_5, dtype="float32")
        key_features_6 = paddle.slice(input=inputs, axes=[1], starts=[39], ends=[49])
        key_features_6 = paddle.cast(key_features_6, dtype="float32")
        key_features_7 = paddle.slice(input=inputs, axes=[1], starts=[61], ends=[69])
        key_features_7 = paddle.cast(key_features_7, dtype="float32")
        # 分离普通离散特征
        tm_wday_feature = paddle.slice(input=inputs, axes=[1], starts=[69], ends=[70])
        tm_wday_feature_int = paddle.cast(tm_wday_feature, dtype="int64")
        # tm_wday_feature_int.stop_gradient = True
        tm_hour_feature = paddle.slice(input=inputs, axes=[1], starts=[70], ends=[71])
        tm_hour_feature_int = paddle.cast(tm_hour_feature, dtype="int64")
        # tm_hour_feature_int.stop_gradient = True
        tm_mins_feature = paddle.slice(input=inputs, axes=[1], starts=[71], ends=[72])
        tm_mins_feature_int = paddle.cast(tm_mins_feature, dtype="int64")
        # tm_mins_feature_int.stop_gradient = True
        length_bin_feature = paddle.slice(input=inputs, axes=[1], starts=[72], ends=[73])
        length_bin_feature_int = paddle.cast(length_bin_feature, dtype="int64")
        # length_bin_feature_int.stop_gradient = True
        # 普通离散特征one-hot
        tm_wday_one_hot = F.one_hot(tm_wday_feature_int, num_classes=7)
        tm_wday_one_hot = paddle.squeeze(tm_wday_one_hot, axis=1)
        tm_hour_one_hot = F.one_hot(tm_hour_feature_int, num_classes=24)
        tm_hour_one_hot = paddle.squeeze(tm_hour_one_hot, axis=1)
        tm_mins_one_hot = F.one_hot(tm_mins_feature_int, num_classes=4)
        tm_mins_one_hot = paddle.squeeze(tm_mins_one_hot, axis=1)
        length_bin_one_hot = F.one_hot(length_bin_feature_int, num_classes=7)
        length_bin_one_hot = paddle.squeeze(length_bin_one_hot, axis=1)
        # 普通稀疏特征embedded
        tm_wday_embedding = self.emb_tm_wday(tm_wday_feature_int)
        tm_wday_embedding = paddle.squeeze(tm_wday_embedding, axis=1)
        tm_hour_embedding = self.emb_tm_hour(tm_hour_feature_int)
        tm_hour_embedding = paddle.squeeze(tm_hour_embedding, axis=1)
        tm_mins_embedding = self.emb_tm_mins(tm_mins_feature_int)
        tm_mins_embedding = paddle.squeeze(tm_mins_embedding, axis=1)
        length_bin_embedding = self.emb_length_bin(length_bin_feature_int)
        length_bin_embedding = paddle.squeeze(length_bin_embedding, axis=1)

        # seq2id序列特征embedding
        seq2id_seq = paddle.slice(input=seq_input, axes=[2], starts=[0], ends=[1])
        seq2id_seq_int = paddle.cast(seq2id_seq, dtype="int64")
        # seq2id_seq_int.stop_gradient = True
        seq2id_seq_emd = self.emb_seq2id(seq2id_seq_int)
        seq2id_seq_emd = paddle.squeeze(seq2id_seq_emd, axis=-2)

        # 路况序列
        status_seq = paddle.slice(input=seq_input, axes=[2], starts=[1], ends=[2])
        status_seq_int = paddle.cast(status_seq, dtype="int64")
        status_seq_emd = self.emb_status(status_seq_int)
        status_seq_emd = paddle.squeeze(status_seq_emd, axis=-2)

        # status_road_level
        status_road_level_seq = paddle.slice(input=seq_input, axes=[2], starts=[2], ends=[3])
        status_road_level_seq_int = paddle.cast(status_road_level_seq, dtype="int64")
        status_road_level_seq_emd = self.emb_status_road_level(status_road_level_seq_int)
        status_road_level_seq_emd = paddle.squeeze(status_road_level_seq_emd, axis=-2)

        # status_width_seq
        status_width_seq = paddle.slice(input=seq_input, axes=[2], starts=[3], ends=[4])
        status_width_seq_int = paddle.cast(status_width_seq, dtype="int64")
        status_width_seq_emd = self.emb_status_width(status_width_seq_int)
        status_width_seq_emd = paddle.squeeze(status_width_seq_emd, axis=-2)

        # status_path_class_seq
        status_path_class_seq = paddle.slice(input=seq_input, axes=[2], starts=[4], ends=[5])
        status_path_class_seq_int = paddle.cast(status_path_class_seq, dtype="int64")
        status_path_class_seq_emd = self.emb_status_path_class(status_path_class_seq_int)
        status_path_class_seq_emd = paddle.squeeze(status_path_class_seq_emd, axis=-2)

        # status_pred_status_seq
        status_pred_status_seq = paddle.slice(input=seq_input, axes=[2], starts=[5], ends=[6])
        status_pred_status_seq_int = paddle.cast(status_pred_status_seq, dtype="int64")
        status_pred_status_seq_emd = self.emb_status_pred_status(status_pred_status_seq_int)
        status_pred_status_seq_emd = paddle.squeeze(status_pred_status_seq_emd, axis=-2)

        #拼接wide dep seq特征
        wide_input = paddle.concat(
            [key_features_1, key_features_2, key_features_3, key_features_4, key_features_5, key_features_6, key_features_7,
                tm_wday_one_hot, tm_hour_one_hot, tm_mins_one_hot, length_bin_one_hot], axis=1)

        deep_input = paddle.slice(input=inputs, axes=[1], starts=[0], ends=[69])
        deep_input = paddle.cast(deep_input, dtype="float32")
        deep_input_other = paddle.slice(input=inputs, axes=[1], starts=[73], ends=[76])
        deep_input_other = paddle.cast(deep_input_other, dtype="float32")
        deep_input = paddle.concat(
            [deep_input, deep_input_other, tm_wday_embedding, tm_hour_embedding, tm_mins_embedding, length_bin_embedding], axis=1)
        
        seq_input_normal = paddle.concat(
            [seq2id_seq_emd, status_seq_emd, status_road_level_seq_emd, status_width_seq_emd, status_path_class_seq_emd, status_pred_status_seq_emd], axis=-1)

        return wide_input, deep_input, seq_input_normal

    # @paddle.jit.to_static
    def forward(self, inputs, seq_input):
        wide_features, deep_features, seq_features = self.format_data(inputs, seq_input)
        # test = [wide_features, deep_features, seq_features]
        # for i, fea in enumerate(test):
        #     print("index:", i, fea.dtype, fea.shape)
        deep_out = self.deep_layer(deep_features)
        wide_out = self.wide_layer(wide_features)
        seq_out = self.seq_layer(seq_features)

        # print("wide_out", wide_out.numpy())
        self.save_wide_out = wide_out.detach()

        com_concat = paddle.concat([wide_out, deep_out, seq_out], axis=1)
        com_hidden1 = self.com_fc_1(com_concat)
        com_hidden1 = self.com_bn_1(com_hidden1)
        # com_hidden1 = F.relu(com_hidden1)
        com_hidden1 = self.com_relu_1(com_hidden1)

        com_hidden2 = self.com_fc_2(com_hidden1)
        com_hidden2 = self.com_bn_2(com_hidden2)
        # com_hidden2 = F.relu(com_hidden2)
        com_hidden2 = self.com_relu_1(com_hidden2)

        com_out = self.com_fc_3(com_hidden2)
        return com_out