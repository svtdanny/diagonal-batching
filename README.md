# Diagonal Batching for Reccurent transformers

Diagonal batching is a scheduling scheme that unlocks parallelism
across segments in RMTs while preserving exact recurrence. This approach eliminates the sequential constraint, enabling efficient GPU inference even for single long-context inputs.

Currently, we are refactoring our code. Last stable version with code we measured performance on in paper is available under tag `v0.0.1-paper`.
Note, that current master branch can have outdated usage examples. 

TODO:
1. Refactor main implementation to make code clean and user frendly
2. Refactor generation - make it cleaner
3. Add best available checkpoint for ARMT models
4. Update to latest ARMT repo, and stop patching it


## Code setup

1. install dependencies from `requirements.txt` file
2. Setup repository from ARMT paper: \
    a. `git clone https://github.com/RodkinIvan/associative-recurrent-memory-transformer.git` \
    b. `cd associative-recurrent-memory-transformer` \
    c. `git checkout 64ea88a428cc72904399fdfc3438e3213ca186c4` \
    d. `git apply ../group_amt.patch`


## Code of diagonal-batching method

1. Inference Diagonal batching operation -> `linear_grouped_sliced_forward.py` and `linear_grouped_forward.py`
2. Training Diagonal batching operation -> `linear_grouped_training.py`
3. Cutlass with optimized output -> `cutlass_emit_pytorch_mocked.py`
4. FastExecutor -> `fast_executor.py`
5. UniversalGroupedExecutor (sutable for llama, gpt models) -> `universal_executor.py`
6. AmortizedExecutor -> `executor.py`

## Usage example

`usage_llama1b_training.py` - simple training example \
`usage_llama1b.ipynb` - interactive comparision of torch model, armt implementation and grouped batching algorithm \
`usage_universal_executor.ipynb` - example of using UniversalGroupedExecutor with both llama and gpt2 model

## Scripts to reproduce results from paper (graphs, tables)
`paper_experiments/measure_flops.ipynb` - Individual operation scaling \
`paper_experiments/llamas_batch_scaling.ipynb` - Llama scaling with batch size \
`paper_experiments/ideal_grouped_scaling.ipynb` - reproduce Ideal/Even Load baseline in paper \
`usage_llama1b.ipynb` - Performance comparision of torch model, armt implementation and grouped batching algorithm


## How to reproduce BABILong evaluation and training

1. Install additional dependencies - clone BABILong repo and prepare data: \
    a. `git clone https://github.com/booydar/babilong.git` \
    b. `unzip ./babilong/data/tasks_1-20_v1-2.zip`
2. `run_eval_bl_fast_trained.py` - example of evaluation on BABILong for ARMT and ARMT with Diagonal batching (trained on BABILong train set)
3. `calc_babilong_scores.ipynb` - extract and plot tables with BABILong scores and inference time for ARMT and ARMT with Diagonal batching
4. `train_babilong_example.ipynb` - example of finetuning ARMT with Diagonal batching on BABILong
5. `run_eval_bl_fast_finetuned.py` - example of evaluation on BABILong for ARMT with Diagonal batching after additional finetuning
