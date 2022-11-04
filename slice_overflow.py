import paddle

def get_coord_features(points, batchsize, rows, cols,
                        layers):  # [B, num_points*2, 4]
    num_points = points.shape[1] // 2
    points = points.reshape([-1, paddle.shape(points)[2]])  # [B*num_points*2, 4] [96, 4]
    # print("!!!!!!", points.shape)
    points, points_order = paddle.split(points, [3, 1], axis=1) # [18432, 512]
    # [B*num_points*2, 3],  [B*num_points*2, 1]
    invalid_points = paddle.max(points, axis=1, keepdim=False) < 0
    row_array = paddle.arange(
        start=0, end=rows, step=1, dtype='float32') # 64
    col_array = paddle.arange(
        start=0, end=cols, step=1, dtype='float32') # 512
    layer_array = paddle.arange(
        start=0, end=layers, step=1, dtype='float32') # 512

    coord_rows, coord_cols, coord_layers = paddle.meshgrid(
        row_array, col_array,
        layer_array)  # len is 3 [rows, cols, layers]
    
    coords = paddle.unsqueeze(
        paddle.stack(
            [coord_rows, coord_cols, coord_layers], axis=0),
        axis=0).tile(  # [B*num_points*2, 3, rows, cols, layers]
            [paddle.shape(points)[0], 1, 1, 1,
                1])  # [768, 3, 512, 512, 12] # repeat # [48, 3, 512, 512, 64]
    

    add_xy = (points * 1.0).reshape([points.shape[0], points.shape[1], 1, 1, 1])
    # add_xy = (points*1.0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    # [B*num_points*2, 3, 1, 1, 1]
    #所有的坐标组合，减去point的数值，只有point对应位置为0，其他相近的也小 [B*num_points*2, 3, rows, cols, layers]
    coords = coords - add_xy
    coords = coords * coords  # [192, 3, 512, 512, 12] 取平方 # [48, 3, 64, 512, 512]
    
    coords[:, 0] += coords[:, 1] + coords[:, 2]
    coords = coords[:, :1]  # [B*num_points*2, 1, rows, cols, layers]

    # [B*2, num_points, 1, rows, cols, layers]
    coords = coords.reshape([-1, num_points, 1, rows, cols, layers])
    # [B*2, 1, 512, 512, 12] 所有point中最小的
    coords = paddle.min(coords, axis=1)
    coords = coords.reshape([-1, 2, rows, cols, layers])
    #  [B, 2, rows, cols, layers]


    coords = (coords <=(2 * 1.0)**2).astype(
                    'float32')  # 只取较小的数值对应的特征


    return coords

points = paddle.to_tensor([[[256., 230.,  51.,  0. ],
         [267., 239.,  44., 100.],
         [257., 234.,  56., 100.],
         [270., 250.,  40., 100.],
         [252., 232.,  51., 100.],
         [260., 228.,  46., 100.],
         [256., 233.,  50., 100.],
         [260., 228.,  46., 100.],
         [268., 241.,  40., 100.],
         [251., 265.,  61., 100.],
         [256., 232.,  51., 100.],
         [253., 233.,  53., 100.],
         [266., 248.,  42., 100.],
         [278., 240.,  42., 100.],
         [249., 268.,  62., 100.],
         [260., 257.,  61., 100.],
         [266., 241.,  42., 100.],
         [258., 227.,  52., 100.],
         [-1. , -1. , -1. , -1. ],
         [-1. , -1. , -1. , -1. ],
         [-1. , -1. , -1. , -1. ],
         [-1. , -1. , -1. , -1. ],
         [-1. , -1. , -1. , -1. ],
         [-1. , -1. , -1. , -1. ],
         [256., 257.,  59., 100.],
         [266., 230.,  48., 100.],
         [257., 230.,  43., 100.],
         [256., 235.,  49., 100.],
         [256., 225.,  43., 100.],
         [375., 200.,  36., 100.],
         [280., 209.,  52., 100.],
         [256., 261.,  60., 100.],
         [ 47.,  19.,  8. , 100.],
         [256., 236.,  46., 100.],
         [256., 243.,  61., 100.],
         [252., 230.,  43., 100.],
         [-1. , -1. , -1. , -1. ],
         [-1. , -1. , -1. , -1. ],
         [-1. , -1. , -1. , -1. ],
         [-1. , -1. , -1. , -1. ],
         [-1. , -1. , -1. , -1. ],
         [-1. , -1. , -1. , -1. ],
         [-1. , -1. , -1. , -1. ],
         [-1. , -1. , -1. , -1. ],
         [-1. , -1. , -1. , -1. ],
         [-1. , -1. , -1. , -1. ],
         [-1. , -1. , -1. , -1. ],
         [-1. , -1. , -1. , -1. ]]])

batch_size, rows, cols,layers = paddle.to_tensor([1]), \
                                paddle.to_tensor([512]), paddle.to_tensor([512]), paddle.to_tensor([64])

c = get_coord_features(points, batch_size, rows, cols, layers)