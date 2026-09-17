| policy | accuracy | f1_macro | mean_latency_ms | p50_latency_ms | p95_latency_ms | p99_latency_ms | throughput_samples_per_s | offloaded_fraction | bandwidth_kb_per_sample | edge_model | cloud_model | hybrid_secondary_model |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| edge_only | 0.8416666666666667 | 0.839386648404466 | 0.0113209250072638 | 0.0108724376186728 | 0.0153516251593828 | 0.0172864999622106 | 88332.0046160868 | 0.0 | 0.0 | linear | cnn_plus | cnn |
| cloud_only | 0.7027777777777777 | 0.7049480392811225 | 42.160487450493704 | 42.16032409667969 | 42.16188430786133 | 42.1635856628418 | 23.7188908495006 | 1.0 | 0.375 | linear | cnn_plus | cnn |
| hybrid | 0.4166666666666667 | 0.346603786353295 | 0.0279503220485316 | 0.0262708123773336 | 0.0369453132152557 | 0.0493906289339065 | 35777.76307062389 | 1.0 | 0.0 | linear | cnn_plus | cnn |
