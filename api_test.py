import paddle
import paddle.fluid as fluid

paddle.enable_static()

input = fluid.data(name="input",
                    shape=[2, 192, 64, 64],
                    dtype='float32')
rois = fluid.data(name="rois",
                  shape=[-1, 4],
                  dtype='float32',
                  lod_level=1)
trans = fluid.data(name="trans",
                    shape=[2, 384, 64, 64],
                    dtype='float32')
x = fluid.layers.deformable_roi_pooling(input=input,
                                        rois=rois,
                                        trans=trans,
                                        no_trans=False,
                                        spatial_scale=1.0,
                                        group_size=(1, 1),
                                        pooled_height=8,
                                        pooled_width=8,
                                        part_size=(8, 8),
                                        sample_per_part=4,
                                        trans_std=0.1,
                                        position_sensitive=False)