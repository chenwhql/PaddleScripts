import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.utils import weight_norm

class CNNEncoder(nn.Layer):
    """
    A `CNNEncoder` takes as input a sequence of vectors and returns a
    single vector, a combination of multiple convolution layers and max pooling layers.
    The input to this encoder is of shape `(batch_size, num_tokens, emb_dim)`, 
    and the output is of shape `(batch_size, ouput_dim)` or `(batch_size, len(ngram_filter_sizes) * num_filter)`.
    The CNN has one convolution layer for each ngram filter size. Each convolution operation gives
    out a vector of size num_filter. The number of times a convolution layer will be used
    is `num_tokens - ngram_size + 1`. The corresponding maxpooling layer aggregates all these
    outputs from the convolution layer and outputs the max.
    This operation is repeated for every ngram size passed, and consequently the dimensionality of
    the output after maxpooling is `len(ngram_filter_sizes) * num_filter`.  This then gets
    (optionally) projected down to a lower dimensional output, specified by `output_dim`.
    We then use a fully connected layer to project in back to the desired output_dim.  For more
    details, refer to `A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural
    Networks for Sentence Classification <https://arxiv.org/abs/1510.03820>`__ , 
    Zhang and Wallace 2016, particularly Figure 1.
    Args:
        emb_dim(int):
            The dimension of each vector in the input sequence.
        num_filter(int):
            This is the output dim for each convolutional layer, which is the number of "filters"
            learned by that layer.
        ngram_filter_sizes(Tuple[int], optinal):
            This specifies both the number of convolutional layers we will create and their sizes.  The
            default of `(2, 3, 4, 5)` will have four convolutional layers, corresponding to encoding
            ngrams of size 2 to 5 with some number of filters.
        conv_layer_activation(Layer, optional):
            Activation to use after the convolution layers.
            Defaults to `paddle.nn.Tanh()`.
        output_dim(int, optional):
            After doing convolutions and pooling, we'll project the collected features into a vector of
            this size.  If this value is `None`, we will just return the result of the max pooling,
            giving an output of shape `len(ngram_filter_sizes) * num_filter`.
            Defaults to `None`.
    Example:
        .. code-block::
            import paddle
            import paddle.nn as nn
            import paddlenlp as nlp
            class CNNModel(nn.Layer):
                def __init__(self,
                            vocab_size,
                            num_classes,
                            emb_dim=128,
                            padding_idx=0,
                            num_filter=128,
                            ngram_filter_sizes=(3, ),
                            fc_hidden_size=96):
                    super().__init__()
                    self.embedder = nn.Embedding(
                        vocab_size, emb_dim, padding_idx=padding_idx)
                    self.encoder = nlp.seq2vec.CNNEncoder(
                        emb_dim=emb_dim,
                        num_filter=num_filter,
                        ngram_filter_sizes=ngram_filter_sizes)
                    self.fc = nn.Linear(self.encoder.get_output_dim(), fc_hidden_size)
                    self.output_layer = nn.Linear(fc_hidden_size, num_classes)
                def forward(self, text):
                    # Shape: (batch_size, num_tokens, embedding_dim)
                    embedded_text = self.embedder(text)
                    # Shape: (batch_size, len(ngram_filter_sizes)*num_filter)
                    encoder_out = self.encoder(embedded_text)
                    encoder_out = paddle.tanh(encoder_out)
                    # Shape: (batch_size, fc_hidden_size)
                    fc_out = self.fc(encoder_out)
                    # Shape: (batch_size, num_classes)
                    logits = self.output_layer(fc_out)
                    return logits
            model = CNNModel(vocab_size=100, num_classes=2)
            text = paddle.randint(low=1, high=10, shape=[1,10], dtype='int32')
            logits = model(text)
    """

    def __init__(self,
                 emb_dim,
                 num_filter,
                 ngram_filter_sizes=(2, 3, 4, 5),
                 conv_layer_activation=nn.Tanh(),
                 output_dim=None,
                 **kwargs):
        super().__init__()
        self._emb_dim = emb_dim
        self._num_filter = num_filter
        self._ngram_filter_sizes = ngram_filter_sizes
        self._activation = conv_layer_activation
        self._output_dim = output_dim

        self.convs = paddle.nn.LayerList([
            nn.Conv2D(in_channels=1,
                      out_channels=self._num_filter,
                      kernel_size=(i, self._emb_dim),
                      **kwargs) for i in self._ngram_filter_sizes
        ])

        maxpool_output_dim = self._num_filter * len(self._ngram_filter_sizes)
        if self._output_dim:
            self.projection_layer = nn.Linear(maxpool_output_dim,
                                              self._output_dim)
        else:
            self.projection_layer = None
            self._output_dim = maxpool_output_dim

    def get_input_dim(self):
        r"""
        Returns the dimension of the vector input for each element in the sequence input
        to a `CNNEncoder`. This is not the shape of the input tensor, but the
        last element of that shape.
        """
        return self._emb_dim

    def get_output_dim(self):
        r"""
        Returns the dimension of the final vector output by this `CNNEncoder`.  This is not
        the shape of the returned tensor, but the last element of that shape.
        """
        return self._output_dim

    def forward(self, inputs, mask=None):
        r"""
        The combination of multiple convolution layers and max pooling layers.
        Args:
            inputs (Tensor): 
                Shape as `(batch_size, num_tokens, emb_dim)` and dtype as `float32` or `float64`.
                Tensor containing the features of the input sequence. 
            mask (Tensor, optional): 
                Shape shoule be same as `inputs` and dtype as `int32`, `int64`, `float32` or `float64`. 
                Its each elements identify whether the corresponding input token is padding or not. 
                If True, not padding token. If False, padding token. 
                Defaults to `None`.
        Returns:
            Tensor:
                Returns tensor `result`.
                If output_dim is None, the result shape is of `(batch_size, output_dim)` and 
                dtype is `float`; If not, the result shape is of `(batch_size, len(ngram_filter_sizes) * num_filter)`.
        """
        if mask is not None:
            inputs = inputs * mask

        # Shape: (batch_size, 1, num_tokens, emb_dim) = (N, C, H, W)
        inputs = inputs.unsqueeze(1)

        # If output_dim is None, result shape of (batch_size, len(ngram_filter_sizes) * num_filter));
        # else, result shape of (batch_size, output_dim).
        convs_out = [
            self._activation(conv(inputs)).squeeze(3) for conv in self.convs
        ]
        maxpool_out = [
            F.adaptive_max_pool1d(t, output_size=1).squeeze(2)
            for t in convs_out
        ]
        result = paddle.concat(maxpool_out, axis=1)

        if self.projection_layer is not None:
            result = self.projection_layer(result)
        return result