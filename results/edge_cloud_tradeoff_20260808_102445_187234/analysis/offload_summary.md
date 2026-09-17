| policy | accuracy | f1_macro | mean_latency_ms | p50_latency_ms | p95_latency_ms | p99_latency_ms | throughput_samples_per_s | offloaded_fraction | bandwidth_kb_per_sample | edge_model | cloud_model | hybrid_secondary_model |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| edge_only | 0.7861111111111111 | 0.7800811285810376 | 0.0175252471533086 | 0.0168958753347396 | 0.0208802502602338 | 0.0266927499324083 | 57060.5362225209 | 0.0 | 0.0 | linear | cnn_plus | cnn |
| cloud_only | 0.6416666666666667 | 0.6241571755303719 | 63.313494533962675 | 63.313236236572266 | 63.31536865234375 | 63.31974411010742 | 15.79442119505154 | 1.0 | 0.375 | linear | cnn_plus | cnn |
| hybrid | 0.3611111111111111 | 0.3100426948687818 | 0.0447932864228884 | 0.0436197519302368 | 0.051333125680685 | 0.0646353736519813 | 22324.774086882375 | 1.0 | 0.0 | linear | cnn_plus | cnn |
