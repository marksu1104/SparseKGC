#!/bin/bash

export PYTHONPATH=`pwd`
echo $PYTHONPATH

<<<<<<< HEAD
if [[ -z "$PYTHON_BIN" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    else
        PYTHON_BIN="python3"
    fi
fi

=======
>>>>>>> 39bcf0d3ffe720aac1329c1ab0ffaf4df7a52c4f
source $1
exp=$2
gpu=$3
ARGS=${@:4}

<<<<<<< HEAD
if [[ "$model" != "conve" ]]; then
    echo "SKIP_NON_CONVE: model=$model (只保留 conve)"
    exit 0
fi

=======
>>>>>>> 39bcf0d3ffe720aac1329c1ab0ffaf4df7a52c4f
add_reversed_training_edges_flag=''
if [[ $add_reversed_training_edges = *"True"* ]]; then
    add_reversed_training_edges_flag="--add_reversed_training_edges"
fi
group_examples_by_query_flag=''
if [[ $group_examples_by_query = *"True"* ]]; then
    group_examples_by_query_flag="--group_examples_by_query"
fi

<<<<<<< HEAD
cmd="$PYTHON_BIN -u -m src.experiments \
=======
cmd="python -m src.experiments \
>>>>>>> 39bcf0d3ffe720aac1329c1ab0ffaf4df7a52c4f
    --data_dir $data_dir \
    $exp \
    --model $model \
    --entity_dim $entity_dim \
    --relation_dim $relation_dim \
    --num_rollouts $num_rollouts \
    --bucket_interval $bucket_interval \
    --num_epochs $num_epochs \
    --num_wait_epochs $num_wait_epochs \
    --batch_size $batch_size \
    --train_batch_size $train_batch_size \
    --dev_batch_size $dev_batch_size \
    --num_negative_samples $num_negative_samples \
    --margin $margin \
    --learning_rate $learning_rate \
    --grad_norm $grad_norm \
    --emb_dropout_rate $emb_dropout_rate \
    --beam_size $beam_size \
    --emb_2D_d1 $emb_2D_d1 \
    --emb_2D_d2 $emb_2D_d2 \
    $group_examples_by_query_flag \
    $add_reversed_training_edges_flag \
    --gpu $gpu \
    $ARGS"

echo "Executing $cmd"

<<<<<<< HEAD
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
=======
$cmd
>>>>>>> 39bcf0d3ffe720aac1329c1ab0ffaf4df7a52c4f
