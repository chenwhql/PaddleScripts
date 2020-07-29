import paddle.fluid as fluid
            
env = fluid.dygraph.ParallelEnvironment()
print("The trainer endpoint are %s" % env.trainer_endpoints)