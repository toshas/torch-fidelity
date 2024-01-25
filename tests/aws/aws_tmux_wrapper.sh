#!/bin/bash

tmux new-session -d -s fid -n fid
tmux send-keys -t fid:fid "(sleep 4h && sudo shutdown -h now) &" Enter
tmux send-keys -t fid:fid "cd ~/torch-fidelity/tests" Enter
tmux send-keys -t fid:fid "bash aws/aws_enable_swap.sh && free -h" Enter
tmux send-keys -t fid:fid "bash run_tests.sh 2>&1 | tee log_$(date '+%Y-%m-%d_%H-%M-%S').txt" Enter
