#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodelist=c-node08
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1200G
#SBATCH --gres=gpu:h200:7
#SBATCH --cpus-per-task=84
#SBATCH --time=24:00:00
#SBATCH --job-name=cbk256_ft
#SBATCH --output=/mnt/beegfs/siyuan/workspace/helios-echo/results/stage1_cbk256_ft_run1.log

# K-means-codebook history finetune, the FramePack-faithful experiment: teach the
# model to read quantized history so we can tell whether training recovers the
# quality that the training-free codebook projection visibly costs.
#
# One node x the 7 temperature-normal cards (c-node08 GPU1 runs hot at 77C). Each node keeps one card
# out: c-node08 GPU1 and mc-node02 GPU5 run hot (77C / 66C) and are in
# Prohibited compute mode anyway, so Slurm's gres count is 7 per node.
# Global batch: bs2 x accum2 x 7 = 28. Stage A ran at 32 (bs2 x accum1 x 16);
# 32 is not reachable at an integer accumulation with 14 processes, so this run
# is at 28 and the difference is recorded here rather than hidden.
#
# The preflight refuses pinghe (standing rule) and requires real headroom per
# card. It does NOT refuse every foreign process the way the Stage A launcher
# did: jiongze holds about 7GB on c-node08, which leaves ~136GB free and is
# harmless, whereas an unconditional refusal would block this run for no reason.
MIN_FREE_MIB=120000

echo "NODE=$SLURM_JOB_NODELIST min_free=${MIN_FREE_MIB}MiB"
# An explicit --mem is mandatory here: DefMemPerNode=UNLIMITED makes a job
# without it request the whole node's memory, which is what left the earlier
# two-node submission pending behind two 64G CPU metrics jobs.

bash -c '
  owners=$(/usr/bin/nvidia-smi --query-compute-apps=pid --format=csv,noheader \
    | /usr/bin/xargs -r -n1 /bin/ps -o user= -p 2>/dev/null | /usr/bin/sort -u | /usr/bin/tr "\n" ",")
  case ",$owners" in
    *,pinghe,*)
      echo "PREFLIGHT_ABORT: pinghe owns GPU processes on $(hostname) (owners=$owners)"
      exit 42
      ;;
  esac
  tight=$(/usr/bin/nvidia-smi --query-gpu=index,compute_mode,memory.free --format=csv,noheader,nounits \
    | /usr/bin/awk -F", " -v m='"$MIN_FREE_MIB"' "\$2 == \"Default\" && (\$3+0) < (m+0) {print \$1\":\"\$3}" | /usr/bin/tr "\n" " ")
  if [ -n "$tight" ]; then
    echo "PREFLIGHT_ABORT: $(hostname) Default cards below ${MIN_FREE_MIB}MiB free: $tight (owners=$owners)"
    exit 43
  fi
  echo "PREFLIGHT_OK host=$(hostname) owners=$owners"
  cd /mnt/beegfs/siyuan/workspace/helios-echo
  export WANDB_MODE=offline
  export PYTHONPATH=.
  export NCCL_DEBUG=WARN
  echo "RANK_NODE=$SLURM_NODEID host=$(hostname)"
  /mnt/beegfs/yuheng/miniconda3/envs/helios/bin/accelerate launch \
    --num_processes 7 \
    --mixed_precision bf16 train_helios.py \
    --config scripts/training/configs/stage1_lora_cbk256_ft.yaml
'
rc=$?
echo "EXIT_CODE=$rc"
exit $rc
