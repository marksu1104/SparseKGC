#!/bin/bash

export PYTHONPATH=`pwd`
echo $PYTHONPATH

if [[ -z "$PYTHON_BIN" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    else
        PYTHON_BIN="python3"
    fi
fi

source $1
exp=$2
gpu=$3
ARGS=${@:4}

if [[ "$exp" != "--process_data" && "$model" != "conve" ]]; then
    echo "SKIP_UNSUPPORTED_MODEL: model=$model exp=$exp"
    exit 0
fi

group_examples_by_query_flag=''
if [[ $group_examples_by_query = *"True"* ]]; then
    group_examples_by_query_flag="--group_examples_by_query"
fi
relation_only_flag=''
if [[ $relation_only = *"True"* ]]; then
    relation_only_flag="--relation_only"
fi
use_action_space_bucketing_flag=''
if [[ $use_action_space_bucketing = *"True"* ]]; then
    use_action_space_bucketing_flag='--use_action_space_bucketing'
fi

cmd="$PYTHON_BIN -u -m src.experiments \
    --data_dir $data_dir \
    $exp \
    --model $model \
    --bandwidth $bandwidth \
    --entity_dim $entity_dim \
    --relation_dim $relation_dim \
    --history_dim $history_dim \
    --history_num_layers $history_num_layers \
    --num_rollouts $num_rollouts \
    --num_rollout_steps $num_rollout_steps \
    --bucket_interval $bucket_interval \
    --num_epochs $num_epochs \
    --num_wait_epochs $num_wait_epochs \
    --num_peek_epochs $num_peek_epochs \
    --batch_size $batch_size \
    --train_batch_size $train_batch_size \
    --dev_batch_size $dev_batch_size \
    --margin $margin \
    --learning_rate $learning_rate \
    --baseline $baseline \
    --grad_norm $grad_norm \
    --emb_dropout_rate $emb_dropout_rate \
    --ff_dropout_rate $ff_dropout_rate \
    --action_dropout_rate $action_dropout_rate \
    --action_dropout_anneal_interval $action_dropout_anneal_interval \
    $relation_only_flag \
    --beta $beta \
    --beam_size $beam_size \
    --num_paths_per_entity $num_paths_per_entity \
    --emb_2D_d1 $emb_2D_d1 \
    --emb_2D_d2 $emb_2D_d2 \
    $group_examples_by_query_flag \
    $use_action_space_bucketing_flag \
    --gpu $gpu \
    $ARGS"

echo "Executing $cmd"

start_ts=$(date +%s)
$cmd
status=$?
end_ts=$(date +%s)
seconds=$((end_ts - start_ts))
if [[ -n "$SPARSEKGC_OUTPUT_DIR" ]]; then
    timing_dir="$SPARSEKGC_OUTPUT_DIR/dackgr"
else
    timing_dir="timings"
fi
mkdir -p "$timing_dir"
timing_file="$timing_dir/dackgr_timings.csv"
if [[ ! -f "$timing_file" ]]; then
    echo "timestamp,baseline,model,dataset,status,seconds,log_file,command" > "$timing_file"
fi
if [[ $status -eq 0 ]]; then
    status_label="ok"
else
    status_label="failed"
fi
dataset_name=$(basename "$data_dir")
command_csv=$(printf '%s' "$cmd" | sed 's/"/""/g')
printf '"%s","DacKGR","%s","%s","%s","%s","","%s"\n' "$(date --iso-8601=seconds)" "$model" "$dataset_name" "$status_label" "$seconds" "$command_csv" >> "$timing_file"
echo "RUNTIME_STD baseline=DacKGR model=$model dataset=$dataset_name status=$status_label seconds=$seconds"
exit $status
