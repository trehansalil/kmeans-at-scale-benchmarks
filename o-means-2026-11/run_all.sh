#!/bin/zsh
# Sequential driver for E1-E3 on the Mac mini. ONE job at a time, each under ../common/guard.py
# (wall cap, RSS cap, memory floor, heartbeat stall). Stops at the first failed or killed job and at a
# global time budget. Finished jobs leave run/<name>.done and are skipped on rerun.
# Prints ALL_DONE or ALL_STOPPED <reason> as its last line; guard events go to logs/all.guard.log.
cd "$(dirname "$0")" || exit 1
export OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
PY=.venv/bin/python
BUDGET_MIN=${BUDGET_MIN:-210}
T0=$(date +%s)

run() {
  name=$1; cap=$2; shift 2
  [ -f run/$name.done ] && { echo "skip $name (done)"; return 0; }
  el=$(( ($(date +%s) - T0) / 60 ))
  if [ $el -ge $BUDGET_MIN ]; then echo "ALL_STOPPED budget ${BUDGET_MIN}min reached before $name"; exit 2; fi
  echo "$(date +%T) job $name cap=${cap}min elapsed=${el}min"
  $PY ../common/guard.py --name $name --max-min $cap --max-rss-gb 8 --stall-min 20 \
      --beat run/$name.beat --done run/$name.done --log logs/all.guard.log -- "$@" >> logs/$name.out 2>&1 \
    || { echo "ALL_STOPPED at $name (see logs/all.guard.log)"; exit 1; }
}

E=experiments
for K in 16384 4096; do
  run e2_sift_k$K 50 $PY $E/e2_equal_epochs.py --dataset sift --n 400000 --k $K --seeds 3 --Ts 1,2,3,5 \
      --methods lloyd25,minibatch,emeans --out results/e2_sift_k$K.csv --beat run/e2_sift_k$K.beat --threads 4 --tag e2k$K
  run e1_sift_k$K 40 $PY $E/e1_fidelity.py --dataset sift --n 400000 --k $K --seeds 3 --Ts 1,2,3,5 \
      --out results/e1_sift_k$K.csv --beat run/e1_sift_k$K.beat --threads 4
  run e3_sift_k$K 40 $PY $E/e3_wallclock.py --dataset sift --n 400000 --k $K --seeds 3 --no-lloyd \
      --out results/e3_sift_k$K.csv --beat run/e3_sift_k$K.beat --threads 4
done
run e2_hard16k_k4096 40 $PY $E/e2_equal_epochs.py --dataset mix_hard16k --n 400000 --k 4096 --seeds 3 --Ts 1,2,3,5 \
    --methods lloyd25,minibatch,emeans --out results/e2_hard16k_k4096.csv --beat run/e2_hard16k.beat --threads 4 --tag e2hard
run e1_hard16k_k4096 30 $PY $E/e1_fidelity.py --dataset mix_hard16k --n 400000 --k 4096 --seeds 3 --Ts 1,2,3,5 \
    --out results/e1_hard16k_k4096.csv --beat run/e1_hard16k.beat --threads 4
run e2_init_k500 15 $PY $E/e2_equal_epochs.py --dataset mix_hard --n 100000 --k 500 --seeds 10 --Ts 1 \
    --methods lloyd25,lloyd25_kpp --out results/e2_init_k500.csv --beat run/e2_init.beat --threads 4 --tag init
echo ALL_DONE
