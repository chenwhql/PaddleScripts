from multiprocessing import Pool
import paddle.fluid

def f(x):
    return x*x

p = Pool(1)
x = [1,2,3,4,5,6]
y = p.map(f, x)
print y
p.terminate()