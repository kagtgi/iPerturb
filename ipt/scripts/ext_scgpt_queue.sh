#!/bin/bash
# waits for the biolord queue, then runs scGPT (as-is, eval only) on both lines, one process at a time
export MALLOC_ARENA_MAX=2
while [ ! -f ~/ext/biolord/queue_done.txt ]; do sleep 20; done
mkdir -p ~/ext/scgpt; cd ~/ext/scgpt
for line in rpe1 K562_essential; do
  /usr/bin/time -v ~/scgpt-env/bin/python ~/ext_scgpt_run.py --line $line > run_$line.log 2>&1
done
echo QUEUE_DONE > queue_done.txt
