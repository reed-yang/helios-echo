# Postmortem: cluster-wide INVAL after Slurm partition-defaults reconfigure

Date: 2026-07-23 02:00–02:15 (+0000) | Impact: all 8 GPU nodes unschedulable ~15 min; zero running jobs harmed (held queue unaffected) | Operator: Claude (authorized by siyuan for the config change)

## Timeline

- 02:00:30 backup slurm.conf; add `DefMemPerGPU=250000 DefCpuPerGPU=16` to the gpu partition line on mc-node01 (primary slurmctld); `scontrol reconfigure`.
- 02:00:45 all 8 nodes → `INVAL` "Low socket*core*thread count, Low CPUs"; slurmctld log: `_slurm_rpc_node_registration ...: Invalid argument` for every node.
- 02:04 rollback (restore backup + reconfigure) — INVAL sticky, `state=resume/undrain/idle` all rejected (`Invalid node state specified`): INVALID_REG clears only via a successful re-registration.
- 02:07 `systemctl restart slurmd` on mc-node01 → node drops to plain `drained` → `resume` works. Same procedure applied to the other 7 nodes via ssh+sudo.
- 02:12 all nodes back to idle/mixed. Recovery complete.

## Root cause

~~Deployment-order error~~ **RULED OUT by attempt 2** (02:35): with slurm.conf byte-identically synced to ALL 8 nodes BEFORE the single reconfigure, every node still went INVAL with the same "Low socket*core*thread count, Low CPUs" registration rejection. Conclusion: adding `DefMemPerGPU`/`DefCpuPerGPU` to the partition line itself trips a node-registration validation failure on **Slurm 24.05.7** (a version where reconfigure was reimplemented "restartless"). Second rollback via the proven path (restore everywhere + slurmd restarts + resume) recovered in ~4 min. Off-line investigation delegated (see logs/research/slurm-defcpupergpu-inval-investigation.md when it lands); do NOT retry these partition params on 24.05.7 without that verdict. Alternative default-injection mechanisms to evaluate: DefMemPerCPU, job_submit.lua, cli_filter, or a Slurm upgrade.

## Correct procedure (for re-applying the change)

1. Edit slurm.conf on the controller; 2. **sync the file to ALL nodes** (scp/pdsh, incl. backup controller mc-node02); 3. `scontrol reconfigure`; 4. verify `sinfo` states + `JobDefaults` on the partition; 5. functional test: 1-GPU job without `--mem` shows `AllocTRES mem=250000M,cpu=16`.

## Ruled out

- Config corruption: `diff` backup vs edited = exactly the intended one-line change.
- The parameters themselves: rejection happened at registration validation, not parameter parsing; rollback with the SAME reconfigure mechanism recovered once configs matched.
- slurmdbd "Unable to connect to database" errors in the log: pre-existing/parallel accounting issue (mc-node01:6819), unrelated to scheduling; still present after recovery — separate item for infra.

## Collateral / separate findings

- smoke run4 (running on mc-node01 during the slurmd restarts) died with CUDA unknown error — plausibly collateral, BUT:
- **mc-node01 GPU2 is unhealthy independent of tonight**: `nvidia-smi` cannot get its device handle (0000:4B:00.0 Unknown Error); latest kernel GPU events are nvidia_uvm unclean-unload slab leaks at **Jul 22 20:08** — hours BEFORE any of tonight's operations. Node likely needs a driver reload/reboot; this may be why it sat fully idle.
- Trainer detail discovered by run4: the trainer validates a config snapshot stored in output_dir against the current config ("Key ... missing in existing config") — scratch run dirs must be cleaned when the config schema changes between attempts.

## Lessons

1. Non-configless Slurm: config sync to every node BEFORE reconfigure — always.
2. A "fully idle" node is a signal to investigate, not just an opportunity (GPU2 was likely why).
3. Cluster-level changes at night: schedule a validation job immediately after, and check `sinfo -R` within a minute of any reconfigure.
