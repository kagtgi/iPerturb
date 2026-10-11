#!/bin/bash
# waits for the STATE queue, then runs biolord on both lines, one process at a time (VM, biolord-env)
export MALLOC_ARENA_MAX=2
while [ ! -f ~/ext/state/queue_done.txt ]; do sleep 20; done
mkdir -p ~/ext/biolord; cd ~/ext/biolord
for line in rpe1 K562_essential; do
  /usr/bin/time -v ~/biolord-env/bin/python ~/ext_biolord_run.py --line $line > run_$line.log 2>&1
done
echo QUEUE_DONE > queue_done.txt
