| policy | accuracy | f1_macro | mean_latency_ms | p50_latency_ms | p95_latency_ms | p99_latency_ms | throughput_samples_per_s | offloaded_fraction | bandwidth_kb_per_sample | edge_model | cloud_model | hybrid_secondary_model |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| edge_only | 0.8416666666666667 | 0.839386648404466 | 0.0117182614902655 | 0.0112213306128978 | 0.0145781524479389 | 0.0172290317714214 | 85336.89070095505 | 0.0 | 0.0 | linear | cnn_plus | cnn |
| cloud_only | 0.7027777777777777 | 0.7049480392811225 | 42.16048083835178 | 42.16028594970703 | 42.16180419921875 | 42.16322708129883 | 23.718894569398227 | 1.0 | 0.375 | linear | cnn_plus | cnn |
| hybrid | 0.4166666666666667 | 0.346603786353295 | 0.0270055139230357 | 0.0261250101029872 | 0.0300390794873237 | 0.0435571745038032 | 37029.47490093862 | 1.0 | 0.0 | linear | cnn_plus | cnn |
