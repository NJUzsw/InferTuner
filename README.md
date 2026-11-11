# InferTuner

InferTuner scales the LLM inference function of a stream processing pipeline so that events meet an end-to-end latency SLO with as few GPUs as possible. At the end of every monitoring window, it estimates the throughput and p99 latency of every candidate parallelism *p* and bundle size *b* with offline performance models, selects (*p*, *b*), and sizes the lookup cache *s* of the lookup join.

This repository contains the algorithms, the offline profiling and training of the performance models, the engine server, and the Flink functions that InferTuner relies on. The rest of our Flink job, the controller that runs the rescaling, and our deployment scripts are engineering work tied to our environment and are not included. 

## Requirements

- Python >= 3.9, `pip install -r requirements.txt`
- For profiling, training, and the engine server: a GPU and `pip install -r requirements-ml.txt`
- For the Flink functions: Apache Flink 1.17 and Java 11

## Quick Start

```bash
python3 -m unittest discover -s tests
```

## Codes of Performance Modeling (Section V)

Offline, once per LLM and GPU type:

```bash
# operator times under the PyTorch profiler, calibration factors, and output lengths, for b in {1, 2, 4, ..., 64}
python3 training/profile_operators.py --model /path/to/llm --prompts prompts.txt --out profiles/my-llm
# one GBDT per operator type
python3 training/train_operator_models.py --profiles profiles/my-llm --out models/my-llm
# output length predictor (BGE-m3 + PCA + MLP)
python3 training/train_length_predictor.py --data profiles/my-llm/lengths.csv --encoder /path/to/bge-m3 \
    --llm-tokenizer /path/to/llm --out models/length-my-llm
```

Online, `infertuner/estimator.py` turns the predicted output lengths and iteration times into the throughput and p99 latency of every candidate (*p*, *b*) from the events of a window.

## Codes of GPU Scaling (Section IV)

`infertuner/gpu_scaler.py` contains the configuration optimization (`config_opt`) and Algorithm 1 (`ScalingTrigger`). At the end of each window:

```python
from infertuner import Config, Estimator, Event, ScalingTrigger
from infertuner.operator_model import ModelSpec, OperatorTimeModel

cfg = Config(slo=6.0, p_max=4)
estimator = Estimator(OperatorTimeModel.load("models/my-llm", ModelSpec(36, 72)))
trigger = ScalingTrigger(p=2, b=8, p_max=cfg.p_max, delta=cfg.delta)

events = [Event(arrival, prompt_len, predicted_output_len, l_other)
          for arrival, prompt_len, predicted_output_len, l_other in window_events]
cands = estimator.estimate_all(events, cfg.parallelisms, cfg.bundle_sizes, cfg.window)
decision = trigger.decide(cands, lam=arrival_rate, slo=cfg.slo)   # decision.p, decision.b, decision.rescale
```

## Codes of Lookup Cache Scaling (Section VI)

`infertuner/cache_scaler.py` computes the target hit rate and the cache capacity from the access counts of a window. The Flink version is in `examples/flink` (`CacheCapacityEstimator.java`, `LookupCache.java`).

## Engine Server

`engine/server.py` holds one model replica on one GPU and runs each bundle as one batch with Hugging Face Transformers. It also predicts output lengths.

```bash
CUDA_VISIBLE_DEVICES=0 python3 engine/server.py --model /path/to/llm --port 8000 \
    --length-model models/length-my-llm --encoder /path/to/bge-m3
```

## Flink Example

`examples/flink` contains the Flink functions that InferTuner relies on:

- `Bundler.java`: groups the events of each instance into bundles of *b*. A new *b* arrives on a broadcast stream from `ControlSource.java` and takes effect from the next bundle, without a restart.
- `LookupJoinFunction.java`, `LookupCache.java`, `CacheCapacityEstimator.java`: the lookup join resizes its cache at the end of each window.
- `InferenceFunction.java`, `EngineClient.java`: each instance passes its bundles to its engine server through Flink Async I/O.
- `Reporter.java`: the measurements that the controller needs. Connect it to your controller.

They are taken from our Flink job and are not a complete pipeline.

## License

Apache License 2.0. See `LICENSE`.
