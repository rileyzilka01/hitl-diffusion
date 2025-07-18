# use the same command as training except the script
# for example:
# bash scripts/eval_policy.sh hitl hitl_block 0001 0 0 false



DEBUG=False

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed=${4}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

gpu_id=${5}
server=${6}


cd hitl-diffusion

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}
python eval.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            server=${server} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt}



                                