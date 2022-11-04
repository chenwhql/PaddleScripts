import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.optimizer as optim

import parser
import argparse

class CausalConv1d(nn.Conv1D):
    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1, **kwargs):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding=dilation*(kernel_size-1),
            dilation=dilation,
            **kwargs)
        
    def forward(self, input):
        out = super(CausalConv1d, self).forward(input)
        return out[:, :, :-self._padding]

class WavenetLayer(nn.Layer):
    def __init__(self, residual_channels, skip_channels, cond_channels, kernel_size=2, dilation=1):
        super(WavenetLayer, self).__init__()

        self.causal = CausalConv1d(residual_channels, 2 * residual_channels, kernel_size, dilation=dilation)
        
        self.condition = nn.Conv1D(cond_channels, 2 * residual_channels, kernel_size=1)
        
        self.residual = nn.Conv1D(residual_channels, residual_channels, kernel_size=1)
        
        self.skip = nn.Conv1D(residual_channels, skip_channels, kernel_size=1)
                

    def _condition(self, x, c, f):
        c = f(c)
        x = x + c
        return x
    
    def forward(self, x, c=None):
        
        x = self.causal(x)
        if c is not None:
            x = self._condition(x, c, self.condition)

        gate, output = x.chunk(2, 1)
        m1= paddle.nn.Sigmoid()   ################
        gate = m1(gate)
        m2= paddle.nn.Tanh()
        output = m2(output)
       
        x = gate * output

        residual = self.residual(x)
        skip = self.skip(x)

        return residual, skip


class WaveNet_x_xy(nn.Layer):

    def __init__(self, blocks=1, layers=10, kernel_size=2, skip_channels=128,
                 residual_channels=128, latent_d=129, shift_input=True):
        super(WaveNet_x_xy, self).__init__()

        self.blocks = blocks
        self.layer_num = layers
        self.kernel_size = kernel_size
        self.skip_channels = skip_channels
        self.residual_channels = residual_channels
        self.cond_channels = latent_d
        self.classes = 256
        self.shift_input = shift_input

        layers = []
        for _ in range(self.blocks):
            for i in range(self.layer_num):
                dilation = 2 ** i
                layers.append(WavenetLayer(self.residual_channels, self.skip_channels, self.cond_channels,
                                           self.kernel_size, dilation))
        self.layers = nn.LayerList(layers)

        self.first_conv = CausalConv1d(1, self.residual_channels, kernel_size=self.kernel_size)
        
        self.skip_conv = nn.Conv1D(self.residual_channels, self.skip_channels, kernel_size=1)

        self.condition = nn.Conv1D(self.cond_channels, self.skip_channels, kernel_size=1)
        
        self.fc = nn.Conv1D(self.skip_channels, self.skip_channels, kernel_size=1)
        
        self.logits = nn.Conv1D(self.skip_channels, self.classes, kernel_size=1)

        
    def _condition(self, x, c, f):
        c = f(c)
        x = x + c
        return x

    def shift_right(self, x):
        #x = F.pad(x, (1, 0))     #torch 实现
        x = F.pad(x,pad=[1,0],data_format='NCL') #paddle 实现
        return x[:, :, :-1]
    
    def forward(self, x, c=None,RB_attr=True):
        x = x.unsqueeze(1)
        x = x / 255.0 - 0.5

        #if self.shift_input:
        #    x = self.shift_right(x)

        residual = self.first_conv(x)
        skip = self.skip_conv(residual)
        #########------------
        #### add RB to net 
        #if( RB_attr ):
        #    for layer in self.layers:  #1  * 10
        #        r, s = layer(residual, c)
        #        residual = residual + r
        #        skip = skip + s
        #####-----------------
        #
        skip = F.relu(skip)
        skip = self.fc(skip)
        
        if c is not None:
            skip = self._condition(skip, c, self.condition)
        skip = F.relu(skip)
        skip = self.logits(skip)

        return skip

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--rb', action='store_true', help='Enable RB layer')
    
    args = parser.parse_args()
        
    paddle.set_device("gpu:0")
    
    ############
    
    decorder = WaveNet_x_xy(3)
    
    ################
    info = \
    '''
      
      paddle 2.1
      cuda 10.2
    
    Please NOTE: device: 0, GPU Compute Capability: 6.1, 
    Driver API Version: 11.1, Runtime API Version: 10.2
    device: 0, cuDNN Version: 7.6.
    '''
    print( 'env.. ','\r\n',info)
    
    import pickle
    from pathlib import Path
    fp = Path(r'samples.npy')
    
    from pathlib import Path
    fp1 = Path(r'decoder_c.npy') 
 
    ##########################
    samples = paddle.to_tensor(pickle.loads(fp.read_bytes() )  )#paddle.uniform( [2,      16000] )
    cin =     paddle.to_tensor(pickle.loads(fp1.read_bytes() )  )#paddle.uniform( [2, 129, 16000] )
    
    ##optime....
    opt_dec = optim.Adam(parameters =decorder.parameters() ,#不加list() 速度快了一点点
                learning_rate=1e-3)
    
    #---train
    decorder.train()
    
    import time
    start = time.time()
    
    for  ep in    range(1001):
        deco = decorder( samples,c = cin,RB_attr=args.rb)
        loss = F.cross_entropy(deco,paddle.cast(samples, paddle.int64),axis=1).unsqueeze(0).mean()
        if(ep % 100 == 0):
            strs = "Yes" if args.rb else "No"
            #print(f" {strs} RB take {time.time()-start} s        /100it ")
            start = time.time()
            print(loss)
        opt_dec.clear_grad()    
                
        loss.backward()     
        opt_dec.step()    
        samples *=  0.0096
        cin *= 0.0095     
 