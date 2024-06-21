#!/bin/bash
. $(dirname $0)/path.sh || exit 1;

export CUDA_VISIBLE_DEVICES=$1

# Experimantal settings
# (* set any config file *)
CONF_DIR=${UPC_IWSLT_ROOT}/config/trials/de_lna_hubert_2.5e-4_ablation
CONF_NAME=no-covost.yaml
export WANDB_NOTES="no CoVoST + FT"

# Environment variables of Weights & Biases
# (* basically you don't have to change *)
export WANDB_ENTITY=sst-team
export WANDB_PROJECT=iwslt2023
export WANDB_NAME=`basename $CONF_DIR`_${CONF_NAME%.*}

# GPU Settings
# (* basically you don't have to change here *)
# to adjust the update_freq according to the number of available GPUs
base_update_freq=24  # default: 24/bs32, 96/bs8
n_gpus=`echo "$1" | awk '{n=split($0,arr,",");print n}'`

# Default settings (may need to be adjusted)
keep_interval_updates=8
patience=8

# Perform in-domain FT when ft=true
ft=${2:-false}
if ${ft}; then
  finetune_from_model=${SAVE_DIR}/${WANDB_NAME}/ckpts/checkpoint_best.pt
  base_update_freq=8
  max_update=3_000
  save_interval_updates=200
  validate_interval_updates=200
  export WANDB_NAME=${WANDB_NAME}/ft
fi

mkdir -p ${SAVE_DIR}/${WANDB_NAME}; cp ${BASH_SOURCE[0]} ${SAVE_DIR}/${WANDB_NAME}
log=${SAVE_DIR}/${WANDB_NAME}/train.log
touch ${SAVE_DIR}/${WANDB_NAME}/${HOSTNAME}:${1}_$(date "+%y%m%d-%H%M%S")
cmd="fairseq-hydra-train
  --config-dir $CONF_DIR
  --config-name $CONF_NAME
  dataset.num_workers=$(($(eval nproc) / 4))
  optimization.update_freq=[$(( $base_update_freq / $n_gpus ))]
  dataset.disable_validation=false
  dataset.skip_invalid_size_inputs_valid_test=true
  checkpoint.keep_best_checkpoints=1
  checkpoint.no_last_checkpoints=false
  checkpoint.keep_interval_updates=$keep_interval_updates
  checkpoint.patience=$patience"
if ${ft}; then
cmd=${cmd}" checkpoint.finetune_from_model=$finetune_from_model
  optimization.max_update=$max_update
  checkpoint.save_interval_updates=$save_interval_updates
  dataset.validate_interval_updates=$validate_interval_updates
  dataset.train_subset=train_mustc"
fi
cmd=""${cmd}" 2>&1 | tee -a $log"
eval $cmd
