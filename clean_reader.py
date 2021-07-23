from paddle_serving_client import Client
import numpy as np
import time
import sys

client = Client()
client.load_client_config(sys.argv[1])
client.connect(["127.0.0.1:9393"])

fetch_list = ['embedding_0.tmp_0', 'embedding_1.tmp_0', 'embedding_2.tmp_0', 'layer_norm_6.tmp_2', 'layer_norm_12.tmp_2', 'layer_norm_18.tmp_2', 'layer_norm_24.tmp_2', 'dropout_7.tmp_0', 'dropout_16.tmp_0', 'dropout_25.tmp_0', 'dropout_34.tmp_0']

src_ids = [1, 3770, 2366, 3, 193, 12043, 2, 65, 179, 106, 286, 609, 3117, 184, 21, 497, 193, 1914, 1058, 82, 12043, 2]
sent_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

src_ids.extend([0]*(300 - len(src_ids)))
sent_ids.extend([0]*(300 - len(sent_ids)))

src_ids = np.array(src_ids).reshape((len(src_ids), 1))
sent_ids = np.array(sent_ids).reshape((len(sent_ids), 1))

print(len(src_ids))
print(len(sent_ids))
start = time.time()
feed_map = {"src_ids": src_ids, "sent_ids": sent_ids}
for i in range(1):
    print(feed_map)
    fetch_map_list = client.predict(feed=feed_map, fetch=fetch_list[:1])
end = time.time()
print((end - start))
#print(fetch_map_list)
