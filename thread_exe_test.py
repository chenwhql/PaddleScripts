
import numpy as np
import threading
import paddle.fluid as fluid
import queue

class ExecutorManager(object):
    def __init__(self, executor_count, device_type):
        self.executor_queue = queue.Queue(executor_count)

        for i in range(executor_count):
            # executor is thread safe,
            # supports single/multiple-GPU running,
            # and single/multiple-CPU running,
            # if CPU device, only create one for all thread
            if device_type == "cpu":
                if self.executor_queue.empty():
                    place = fluid.CPUPlace()
                    executor = fluid.Executor(place)
                    self._temp_executor = executor
                else:
                    executor = self._temp_executor
            else:
                device_id = gpu_device_ids[i]
                place = fluid.CUDAPlace(device_id)
                executor = fluid.Executor(place)

            self.executor_queue.put(executor)

    def get_executor(self):
        return self.executor_queue.get()

    def return_executor(self, executor):
        return self.executor_queue.put(executor)

executor_manager = ExecutorManager(3, "cpu")
place = fluid.CPUPlace()
exe = fluid.Executor(place)

scopes = []
for i in range(3):
    scope = fluid.Scope()
    path = "./save_load/fc.example.model"
    with fluid.scope_guard(scope):
        [inference_program, feed, fetch] = (
            fluid.io.load_inference_model(dirname=path, executor=exe))
    scopes.append(scope)

# scope = fluid.global_scope()
# path = "./save_load/fc.example.model"
# with fluid.scope_guard(scope):
#     [inference_program, feed, fetch] = (
#         fluid.io.load_inference_model(dirname=path, executor=exe))

def __thread_main__(scope):
# def __thread_main__():
    executor = executor_manager.get_executor()
    if executor is not None:
        try:
            with fluid.scope_guard(scope):
                x = np.random.random((1, 784)).astype('float32')
                outputs = executor.run(inference_program,
                                      feed={feed[0]: x},
                                      fetch_list=fetch,
                                      return_numpy=False)
                print(outputs)
        finally:
            executor_manager.return_executor(executor)

for i in range(3):
    thread = threading.Thread(target=__thread_main__, args=(scopes[i],))
    # thread = threading.Thread(target=__thread_main__)
    # thread.daemon = True
    thread.start()
