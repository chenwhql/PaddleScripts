import tensorflow as tf

class Calc(tf.Module):

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def mul(self, x):
        return x * x * 1.

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float64)])
    def add(self, x):
        return x + x + 1.


to_export = Calc()
tf.saved_model.save(to_export, './tmp/calc')

loaded_calc = tf.saved_model.load('./tmp/calc')

print(loaded_calc.add(tf.constant(2.0)))
print(loaded_calc.mul(tf.constant(2.0)))
