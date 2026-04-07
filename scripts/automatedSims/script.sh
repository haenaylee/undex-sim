#!/bin/bash
#SBATCH -J underwaterF
#SBATCH -A fuge-prj-jrl
#SBATCH -p standard
#SBATCH -t 06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

module load ansys

#Metadata echoes so the automated Python script can parse Node and JobID from Slurm output
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_NODELIST=${SLURM_JOB_NODELIST}"

#Slurm resource logger; starts at job launch
#Logs to: ~/slurm_monitor_logs/<JOBID>.csv
LOGDIR="$HOME/slurm_monitor_logs"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/${SLURM_JOB_ID}.csv"

#Write CSV header once
if [ ! -f "$LOGFILE" ]; then
  echo "epoch,iso,maxrss,avecpu" > "$LOGFILE"
fi

monitor_loop() {
  while true; do
    #Query sstat for the .batch step
    line=$(sstat -j "${SLURM_JOB_ID}.batch" -P --format=MaxRSS,AveCPU 2>/dev/null | tail -n 1)

    #Expect "MaxRSS|AveCPU"
    if echo "$line" | grep -q "|"; then
      maxrss=$(echo "$line" | cut -d"|" -f1)
      avecpu=$(echo "$line" | cut -d"|" -f2)

      #Only log if AveCPU looks like a valid Slurm time
      if echo "$avecpu" | grep -Eq '^[0-9]+(-[0-9]{1,2})?:[0-9]{2}:[0-9]{2}$'; then
        epoch=$(date +%s)
        iso=$(date -Is)
        echo "${epoch},${iso},${maxrss},${avecpu}" >> "$LOGFILE"
      fi
    fi

    sleep 5
  done
}

#Start logger in background
monitor_loop &
MON_PID=$!

#Ensure logger stops when job exits
cleanup() {
  kill $MON_PID 2>/dev/null
}
trap cleanup EXIT

#Run LS-DYNA
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
lsdyna i=input.k memory=1300000000      #large memory, works for element size = 0.2 cm
