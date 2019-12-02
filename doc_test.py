import paddle.fluid as fluid

x = fluid.data( name='x', shape=[-1, 10], dtype='float32' )
trans = fluid.layers.fc( x, 100 )
cost = fluid.layers.reduce_mean( trans )
optimizer = fluid.optimizer.DecayedAdagradOptimizer(learning_rate=0.2)
optimizer.minimize(cost)