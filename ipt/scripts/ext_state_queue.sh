#!/bin/bash
# Sequential STATE (released ST-HVG-Replogle checkpoints, evaluated as-is, no fine-tuning). Runs on the VM.
export MALLOC_ARENA_MAX=2
cd ~/ext/state
B=/home/Win/state_ckpt/ST-HVG-Replogle
for variant in fewshot zeroshot; do
  for spec in "rpe1 rpe1" "K562_essential k562"; do
    set -- $spec; line=$1; k=$2
    d=$B/$variant/$k
    out=pred_${variant}_$line.h5ad
    /usr/bin/time -v ~/state-env/bin/state tx infer --model-dir $d --checkpoint $d/checkpoints/best.ckpt \
      --adata ctrl_$line.h5ad --pert-col gene --embed-key X_hvg --tsv perts_$line.tsv --output $out > infer_${variant}_$line.log 2>&1
    ~/state-env/bin/python ~/ext_state_post.py --line $line --h5ad $out --var-dims $d/var_dims.pkl \
      --targets-json targets_$line.json --out ~/ext/state_$variant/$line > post_${variant}_$line.log 2>&1
    rm -f $out
  done
done
echo QUEUE_DONE > queue_done.txt
