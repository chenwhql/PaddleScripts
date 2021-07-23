class BaseBatchCreator(object):
    """
        Base batch creator 
    """
    def __init__(self, batch_size):
        self.batch = []
        self._batch_size = batch_size

    def append(self, info):
        """
            append info
        """
        self.batch.append(info)
        if len(self.batch) == self._batch_size:
            result = self.batch
            #print('BATCH LEN', len(result))
            self.batch = []
            return result

def do_negative_sample(self, line):
    samples = []
    splits = line.strip().split('\t')
    #splits = line.strip().split('-****-')
    pos_sample = splits[1]
    neg_samples = splits[2].split('\x02')
    cid = splits[0].split('_')[-1]
    if cid in self.cid_kv:
        cid = self.cid_kv[cid]
    else:
        cid = self.cid_kv['default_group']
    #print(cid)
    #print('NEG LEN:', len(neg_samples))
    neg_idx = range(0, len(neg_samples))
    if len(neg_samples) > self._flags.neg_num:
        if self._flags.neg_num < 4:
            neg_idx = range(0, self._flags.neg_num)
        else:
            neg_idx = [0, 1, 2] + random.sample(neg_idx, self._flags.neg_num - 3)
    for k in neg_idx:
        samples.append(self.parse_dfm_pairwise_one_line_delete_cnn([pos_sample, neg_samples[k], cid]))
    return samples

def _prepare_train_input(self, data):
        """
        convert batced input lines into batched model input.
        """
        batch_data = [d for line in data for d in self.do_negative_sample(line)]
        pos_idx_list, pos_val_list, neg_idx_list, neg_val_list, cid_list = zip(*batch_data)
        return ('pos_feat_idx', np.array(pos_idx_list).astype("int64")), \
               ('pos_feat_val', np.array(pos_val_list).astype("float32")), \
               ('neg_feat_idx', np.array(neg_idx_list).astype("int64")), \
               ('neg_feat_val', np.array(neg_val_list).astype("float32")), \
               ('city_id', np.array(cid_list).astype("int64"))

def parse_batch(self, data_gen):
        """
        parse batch data
        """
        def __batch_gen():
            batch_creator = BaseBatchCreator(self._flags.batch_size)
            for line in data_gen():
                batch = batch_creator.append(line)
                if batch is not None:
                    yield batch 
            if not self._flags.drop_last_batch and len(batch_creator.batch) != 0:
                yield batch_creator.batch
        
        data_reader = __batch_gen
        #if not use pyreader, use stack(...) for multi device
        output_batch = []
        i = 0
        for data in data_reader():
            data_inputs = self._prepare_train_input(data)
            yield data_inputs

def get_sample_reader(input_names, encoding_str='utf-8'):
    """
    return pyreader object.
    """
    def _data_generator(): 
        for fname in _flags.file_list.split(','):
            with codecs.open(fname, 'r', encoding=encoding_str) as fp:
                for line in fp:
                    if not line.strip():
                        continue
                    yield line
    def _batch_reader():
        for batch in parse_batch(_data_generator):
            sample_list = [value for key, value in batch if key in input_names]
            yield sample_lis
    return _batch_reader
        
py_reader = fluid.io.PyReader(feed_list=self.input_layers, 
                                          capacity=FLAGS.py_reader_capacity, 
                                          use_double_buffer=FLAGS.py_reader_use_double_buffer,
                                          iterable=FLAGS.py_reader_iterable)
sample_reader = get_sample_reader()
py_reader.decorate_batch_generator(sample_reader, places)
        