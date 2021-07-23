import tensorflow as tf

class Calc(tf.Module):
    def add(self, x):
        return x + x + 1.


to_export = Calc()
tf.saved_model.save(to_export, './calc_nf')

loaded_calc = tf.saved_model.load('./calc_nf')

print(loaded_calc)
# print(loaded_calc.add(tf.constant(2.0)))
