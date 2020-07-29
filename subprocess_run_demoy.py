import os
import numpy as np
import multiprocessing

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def _read_into_queue(reader, queue):
    try:
        for sample in reader():
            if sample is None:
                raise ValueError("sample has None")
            queue.put(sample)
        queue.put(None)
    except:
        queue.put("")
        six.reraise(*sys.exc_info())

def queue_reader(readers):
    queue = multiprocessing.Queue(10)
    for reader in readers:
        p = multiprocessing.Process(
            target=_read_into_queue, args=(reader, queue))
        p.start()

    reader_num = len(readers)
    finish_num = 0
    while finish_num < reader_num:
        sample = queue.get()
        if sample is None:
            finish_num += 1
        elif sample == "":
            raise ValueError("multiprocess reader raises an exception")
        else:
            yield sample

def reader():
    def __impl__():
      import paddle
      import paddle.fluid as fluid

      place = fluid.CUDAPlace(0)
      exe = fluid.Executor(place)
      
      [inference_program, feed_target_names, fetch_targets] = (
          fluid.io.load_inference_model(dirname="./save_load/fc.example.model", executor=exe))

      tensor_img = np.array(np.random.random((1, 784)), dtype=np.float32)
      results = exe.run(inference_program,
                        feed={feed_target_names[0]: tensor_img},
                        fetch_list=fetch_targets)
      yield results[0]

    return __impl__

if __name__=="__main__":
    train_reader = queue_reader([reader(), reader(), reader()])
    for data in train_reader:
        print(data)