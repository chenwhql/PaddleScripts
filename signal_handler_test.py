import paddle
import signal
import os

paddle.disable_signal_handler()
os.kill(os.getpid(), signal.SIGSEGV)