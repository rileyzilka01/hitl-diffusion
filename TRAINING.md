# Start the tmux server

tmux new -s hitl_train

1. `conda activate hitl`
2. in scripts/train.sh there is an example script to run training
3. leaving the session: `ctrl+b` then `d`
4. `ctrl+b` then `c` for new window for eval
5. in scripts/eval.sh there is an example script to run eval

### To get back into tmux sesssion

1. `tmux attach -t hitl_train`