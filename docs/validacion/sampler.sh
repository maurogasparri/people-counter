#!/bin/bash
# Soak sampler: muestrea sistema cada INTERVAL s, N veces -> CSV.
OUT="$1"; N="${2:-720}"; INTERVAL="${3:-5}"
CG=/sys/fs/cgroup/system.slice/people-counter.service
echo "epoch,iso,temp_c,throttled,arm_hz,load1,load5,load15,cpu_total_j,cpu_idle_j,mem_total_b,mem_avail_b,mem_free_b,buffcache_b,swap_total_b,swap_free_b,svc_mem_cur_b,svc_mem_peak_b,cg_high,cg_max,cg_oom,cg_oomkill" > "$OUT"
for ((i=0;i<N;i++)); do
  EP=$(date +%s); ISO=$(date -Iseconds)
  T=$(vcgencmd measure_temp | sed "s/temp=//;s/'C//")
  TH=$(vcgencmd get_throttled | sed 's/throttled=//')
  ARM=$(vcgencmd measure_clock arm | sed 's/.*=//')
  read L1 L5 L15 _ < /proc/loadavg
  read _ U N_ S IDL IOW IRQ SIRQ ST _ _ < <(grep '^cpu ' /proc/stat)
  TOTAL=$((U+N_+S+IDL+IOW+IRQ+SIRQ+ST)); CIDLE=$((IDL+IOW))
  read MT MA MF BC ST_ SF_ < <(awk '/^MemTotal:/{mt=$2*1024}/^MemAvailable:/{ma=$2*1024}/^MemFree:/{mf=$2*1024}/^Buffers:/{b=$2*1024}/^Cached:/{c=$2*1024}/^SwapTotal:/{st=$2*1024}/^SwapFree:/{sf=$2*1024}END{print mt,ma,mf,b+c,st,sf}' /proc/meminfo)
  SC=$(cat $CG/memory.current 2>/dev/null||echo); SP=$(cat $CG/memory.peak 2>/dev/null||echo)
  EV=$(cat $CG/memory.events 2>/dev/null)
  HI=$(echo "$EV"|awk '/^high/{print $2}'); MX=$(echo "$EV"|awk '/^max/{print $2}')
  OM=$(echo "$EV"|awk '/^oom /{print $2}'); OK=$(echo "$EV"|awk '/^oom_kill/{print $2}')
  echo "$EP,$ISO,$T,$TH,$ARM,$L1,$L5,$L15,$TOTAL,$CIDLE,$MT,$MA,$MF,$BC,$ST_,$SF_,$SC,$SP,$HI,$MX,$OM,$OK" >> "$OUT"
  sleep "$INTERVAL"
done
