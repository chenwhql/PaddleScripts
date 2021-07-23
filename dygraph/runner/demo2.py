import numpy as np
import paddlehub as hub
import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear

with fluid.dygraph.guard():
    ernie = hub.Module(name="ernie", version="1.2.0")
    ernie_layer = fluid.dygraph.StaticModelRunner(ernie.params_path)
    dataset = hub.dataset.ChnSentiCorp()

    reader = hub.reader.ClassifyReader(
        dataset=dataset,
        vocab_path=ernie.get_vocab_path(),
        max_seq_len=512,
        sp_model_path=ernie.get_spm_path(),
        word_dict_path=ernie.get_word_dict_path())
    train_reader = reader.data_generator(
        batch_size=16,
        phase= 'train')

    for data_id, data in enumerate(train_reader()):
        input_ids = np.array(data[0][0]).astype(np.int64)
        position_ids = np.array(data[0][1]).astype(np.int64)
        segment_ids = np.array(data[0][2]).astype(np.int64)
        input_mask = np.array(data[0][3]).astype(np.float32)
        labels = np.array(data[0][4]).astype(np.int64)
        # 问题一：多输入时提示错误
        # 问题二：本地修改static_runner的代码，解决上述问题后，输入类型与定义一致，但是提示类型错误
        # 问题三：要求输入的数据的顺序与load_inference_model的feed_list不一致
        # feed_list为 0:input_ids 1:position_ids 2:segment_ids 3:input_mask
        # static_model_runner为 0:position_ids 1:input_mask 2:input_ids 3:segment_ids
        # 真正顺序为 0:segment_ids 1:input_ids 2:input_mask 3:position_ids
        # pooled_output, sequence_output = ernie_layer([segment_ids, input_ids, input_mask, position_ids])
        pooled_output, sequence_output = ernie_layer(segment_ids, input_ids, input_mask, position_ids)
        # pooled_output, sequence_output = ernie_layer(position_ids, input_mask, input_ids, segment_ids)