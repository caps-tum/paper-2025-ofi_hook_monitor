#!/usr/bin/bash
! getopt --test > /dev/null 
if [[ ${PIPESTATUS[0]} -ne 4 ]]; then
    echo "'getopt --test' failed in this environment."
    exit 1
fi

LONGOPTS=basedir:,mpirun:,mpi-args:,binary:,binary-args:,iterations:,verbose,
OPTIONS=
! PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@")
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    # e.g. return value is 1
    #  then getopt has complained about wrong arguments to stdout
    exit 2
fi

BASE_PWD="$(pwd)/data"
ITERATIONS=10
eval set -- "$PARSED"
while true; do
    case "$1" in
    --basedir)
      BASE_PWD="$2"
      shift 2 ;;
    --mpirun)
      MPIRUN="$2"
      shift 2 ;;
    --mpi-args)
      MPI_ARGS="$2"
      shift 2 ;;
    --binary)
      BINARY="$2"
      shift 2 ;;
    --binary-args)
	    BINARY_ARGS="$2"
      shift 2 ;;
    --iterations)
      ITERATIONS="$2"
      shift 2 ;;
    --verbose)
      VERBOSE=1
      shift 1 ;;
    --)
      shift 
      break 
      ;;
    *)
      echo "Programming error, remaining args: $@"
      exit 3
      ;;
    esac
done

profiles=(
  ""

  "-x FI_HOOK=ofi_hook_monitor\
   -x FI_HOOK_MONITOR_BASEPATH=/dev/shm/ofi_$USER\
   -x FI_HOOK_MONITOR_TICK_MAX=20000\
   -x FI_HOOK_MONITOR_LINGER=1"
 
   "-x FI_HOOK=ofi_hook_monitor\
    -x FI_HOOK_MONITOR_BASEPATH=/dev/shm/ofi_$USER\
    -x FI_HOOK_MONITOR_TICK_MAX=100\
    -x FI_HOOK_MONITOR_LINGER=1"
  )

if [[ $VERBOSE == 1 ]]; then
  echo "BASE_PWD   : $BASE_PWD"
  echo "MPIRUN     : $MPIRUN"
  echo "MPI_ARGS   : $MPI_ARGS"
  echo "BINARY     : $BINARY"
  echo "BINARY_ARGS: $BINARY_ARGS"
  echo "ITERATIONS : $ITERATIONS"
  echo "VERBOSE    : $VERBOSE"
  for profile in "${profiles[@]}"; do
    echo "profile: $profile"
  done
fi


BASEDIR="measurements_$(hostname)_$(basename $BINARY)_$(date +%y-%m-%dT%H%M)"
OUTDIR="$BASE_PWD"

# if $OUTDIR/$BASEDIR exists, try $OUTDIR/$BASEDIR-$i until not exists
_basedir="$BASEDIR"
i=1
while true; do
  if [[ -d "$OUTDIR/$_basedir" ]]; then
        _basedir="$BASEDIR-$i"
    i=$((i+1))
  else break
  fi
done

BASEDIR="$_basedir"
BASEPATH="$OUTDIR/$BASEDIR"

echo "Will write to: $BASEDIR"
mkdir -p "$BASEPATH"

for i in "${!profiles[@]}"; do
  mkdir -p "$BASEPATH/profile_$i"
done 

START_GLOBAL=$(($(date +%s%N)/1000000))

ITERATION_MARKERS=()
for i in $(seq 1 "$ITERATIONS"); do
  for j in "${!profiles[@]}"; do
    profile="${profiles[$j]}"
  	printf "\r[%-2s/%s]" "$i" "$ITERATIONS"
  	ITERATION_MARKERS+=( $(($(date +%s%N)/1000000)) )
  	$MPIRUN --report-pid - $MPI_ARGS $profile \
  	  $BINARY $BINARY_ARGS \
      > "$BASEPATH/profile_$j/measurement_$i.dat"
  done
done
END_GLOBAL=$(($(date +%s%N)/1000000))

echo ""

read -d '' meta_info << EOF
Measurement Start Timestamp: \`$START_GLOBAL\`
Measurement End Timestamp: \`$END_GLOBAL\`
Host: $(hostname)
MPI Path: \`$MPIRUN\`
MPI Arguments: \`$MPI_ARGS\`
Binary: \`$BINARY\`
Binary Arguments: \`$BINARY_ARGS\`
Iterations: \`$ITERATIONS\`
Iteration Markers: \`${ITERATION_MARKERS[@]}\`
SLURM_JOB_ID: \`$SLURM_JOB_ID\`
EOF
echo "$meta_info" > "$BASEPATH/meta.md"

echo "Written data to: $BASEDIR"

