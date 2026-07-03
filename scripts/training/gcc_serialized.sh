#!/bin/bash
# Serialized + retrying gcc wrapper to defeat the beegfs concurrent-header-read race.
#
# Triton/torch JIT-compile small launchers (cuda_utils.c, __triton_launcher.c) at first kernel
# launch by invoking gcc, which reads the Python C headers from the conda env on beegfs
# (-I .../envs/helios/include/python3.11). When all 8 ranks on a node compile simultaneously,
# beegfs intermittently returns ENOENT for sub-headers ("fatal error: cpython/tupleobject.h:
# No such file") -> gcc exits 1 -> --kill-on-bad-exit kills the whole job. Triton's _build honors
# the CC env var (triton/runtime/build.py), so we point CC here: a per-node flock serializes the
# gcc calls (one beegfs header reader at a time) and a single retry absorbs any residual hiccup.
LOCK="${HELIOS_GCC_LOCK:-/tmp/helios_gcc_${USER:-u}.lock}"
exec 9>"$LOCK"
flock 9
/usr/bin/gcc "$@" || { sleep 1; /usr/bin/gcc "$@"; }
rc=$?
exit $rc
