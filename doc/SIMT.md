# SIMT mode: one work-item per SIMD lane

Status: implemented for kernels with uniform control flow (`src/normalization/SIMT.cpp`). Verified in
the emulator and on a Raspberry Pi 3B through the vc4 DRM backend.

## Why

VC4C runs every OpenCL work-item on a whole QPU, and uses the QPU's 16 SIMD lanes only for the components
of vector types within that work-item. A kernel written with scalar types (`float`, `int`) therefore
does useful work in 1 of 16 lanes. clpeak on a Raspberry Pi 3B (V3D at 300 MHz) shows the effect:

| | `float` | `float16` |
|---|---|---|
| single-precision compute | 0.61 GFLOPS | 6.12 GFLOPS |
| global memory bandwidth | 0.21 GB/s | 1.33 GB/s |

GPUs usually map one work-item to one SIMD lane ("SIMT"), so scalar code fills the hardware. Doing the
same on the QPU makes scalar kernels up to 16× faster. Measured with SIMT mode (see Results): `float`
compute 6.60 GFLOPS (10.8×), scalar `float` bandwidth 8.9× higher.

## Execution model

In SIMT mode a **work-group runs on one QPU, one work-item per SIMD lane**:

- a work-group has at most 16 work-items (`CL_KERNEL_WORK_GROUP_SIZE` = 16) and is 1-dimensional;
- up to 12 work-groups run in parallel, one per QPU. A single launch runs many work-groups: QPU *q* runs
  the work-groups *q*, *q* + 12, *q* + 24, … one after the other (see "Work-group loop");
- for a local size below 16, the lanes beyond the local size are inactive. They compute garbage, but
  never store it (see below).

With one work-group per QPU, the only inactive lanes are a suffix of every QPU, whose length is the
same for all QPUs and already in the uniforms (the local size).

Planned: schedule 16-work-item *chunks* instead of work-groups. Since SIMT kernels have no barriers
or `__local` memory, it doesn't matter which QPU runs which chunk of which work-group. VC4CL would
deal all (group, chunk) pairs round-robin across the QPUs, so one large work-group uses all 12 QPUs
and many small ones run side by side, and work-groups could have up to 255 work-items (the 8-bit
local-size field). The kernel then needs the active-lane count `min(16, local_size − first_local_id)`
instead of `local_size`.

`CL_DEVICE_MAX_WORK_GROUP_SIZE` stays 12, so applications that size their work-groups from the device
limit keep working for non-SIMT kernels. SIMT kernels report 16 through `CL_KERNEL_WORK_GROUP_SIZE`.

## Hardware behavior relevant for SIMT

Measured with hand-assembled QPU programs submitted through the vc4 compute ioctls:
`vc4-compute/tests/qpu_probe.py`, whose assembler reproduces VC4C's encodings of 15 reference
instructions exactly.

| Question | Result |
|---|---|
| Per-lane conditional writes to *registers* | Work per lane (VC4C already relies on this, e.g. `or.ifzc rb2, r0, r0`). |
| Per-lane conditional write to `vpm` | **Not masked.** All 16 lanes are written, even with condition *never*. Masked lanes receive a stale value: the result of the previous instruction. |
| Per-lane conditional write to `tmu0_s` (address of a load) | **Not masked.** A request is issued for all 16 lanes. Masked lanes read from undefined addresses and return garbage. |
| `tmu0_s` write with all lanes masked | Still issues a request for all lanes, so it doesn't hang. All values are garbage. |
| VPM-to-memory DMA (VDW) of part of a row | **Works exactly.** Start column and width (`VPMBase` column bits, `Depth`) write only the selected words; memory around them is untouched. |
| Branches | Condition on "all lanes" or "any lane" (`ifallzc`, `ifanyz`, …). |

**Per-lane conditions only work for register writes.** Writes to peripheral registers (VPM, TMU, and
presumably the other I/O registers) always act on all 16 lanes.

This matters for safety, not just correctness. The QPUs have no MMU, so a load issued by an inactive
lane with a garbage address can read any physical address, including peripheral registers. A VPM row
written with stale lanes and stored by DMA overwrites memory.

How the implementation stays safe with these rules:

- **Loads.** All addresses are contiguous across the lanes (`base + lane × element size`), so an
  inactive lane reads at most 15 elements past the work-group's last element. That's ordinary RAM
  next to the buffer, never a peripheral, and the values are discarded. This holds for TMU loads and
  for the 16-word VPM DMA reads VC4C uses for buffers that the kernel also writes. Such a read may
  overlap the next work-group's first elements while another QPU writes them, but only inactive lanes
  see those words.
- **Stores.** A store writes the whole VPM row, then DMAs only the first *local size* words of it
  (`Depth` computed at run time from the local-size uniform). Inactive lanes never reach memory.

## Implementation

### Which kernels use SIMT mode

`vectorizeWorkItems` (`src/normalization/SIMT.cpp`) runs early in normalization, before the
work-item functions are intrinsified. It converts a kernel only if:

- all its locals are scalars of at most 32 bits or pointers (no vector types, no 64-bit values);
- it uses no barriers, atomics, mutexes, semaphores, async copies/prefetches, `get_global_linear_id`,
  `__local` pointer parameters or stack allocations (private arrays);
- every memory access reads or writes a single element; no `memcpy`/`memset`-style accesses;
- every work-item dependent address has a lane stride equal to the element size (`p[gid + c]`,
  `p[get_local_id(0)]`, …), i.e. the 16 lanes access 16 consecutive elements;
- no value is written through a work-group uniform address (all lanes would race for one location);
- no branch depends on a work-item dependent value (**uniform control flow**). Per-lane `select`s
  are fine;
- work-item IDs are only queried for dimension 0, with a constant argument;
- a `reqd_work_group_size`, if given, is 1-dimensional and at most 16.

Otherwise the kernel is compiled as before, and the log says why (INFO level, e.g.
`SIMT: not using SIMT mode for kernel 'bounded': divergent branch`).

### Switching SIMT mode on and off

SIMT mode is on by default. It can be switched off, or back on, like an optimization pass:

| Where | Off | On |
|---|---|---|
| Whole process (VC4C library, e.g. inside VC4CL) | environment variable `VC4C_NO_SIMT=1` | (default) |
| One program in VC4CL | build option `--fno-simt` in `clBuildProgram`/`clCompileProgram` | build option `--fsimt` |
| `vc4c` command line | `--fno-simt` | `--fsimt` |

The options take precedence over the environment variable, so `--fsimt` uses SIMT mode even with
`VC4C_NO_SIMT` set. VC4CL removes these two options before passing the build options to the
front-end compiler (clang doesn't know them). `CL_PROGRAM_BUILD_OPTIONS` still returns the options as
given. `--fsimt` only allows SIMT mode: kernels that don't qualify are still compiled the normal way.

### The conversion

1. **Analysis.** A fixed-point pass finds the *varying* (work-item dependent) locals, starting from
   `get_local_id(0)` / `get_global_id(0)` (lane stride 1), and tracks each one's lane stride through
   moves, `add`, `sub`, `shl` and `mul` by constants, and `zext`/`sext` of the IDs themselves (the
   local ID is an 8-bit value, which can't wrap around since it's below 16). Loaded values, and values computed under flags
   set from varying values, are varying with unknown stride.
2. **Vectorization.** Every varying non-pointer local is replaced by a 16-element vector of the same
   element type, so the existing vector code generation does the work. ID calls become
   `id_vector = id_of_first_lane + elem_num`.
3. **Memory accesses.** Varying pointers stay scalar (their per-lane values are still computed). An
   access through one is changed to access a `<16 x T>` at lane 0's address; TMU loads then use
   `lane 0 address + elem_num × 4` and stores write one VPM row.
4. **Metadata.** `mergedWorkItemsFactor` is set to 16, which VC4CL already reads from the kernel
   header.

Other changes for SIMT kernels (`mergedWorkItemsFactor > 1`):

| Where | What |
|---|---|
| `KernelMetaData.h` | Maximum work-group size = merge factor (one QPU per group); maximum instances = 12 QPUs |
| `optimization/ControlFlow.cpp` (`addWorkGroupLoop`) | SIMT work-group loop instead of the normal one (see below) |
| `periphery/VPM.cpp` (`lowerWriteRAM`) | A 16-wide store DMAs `local_size & 0xFF` words instead of 16 |
| `normalization/MemoryAccess.cpp` | The check for memory dependencies between work-items accounts for the 16 work-items per QPU. Without it every SIMT kernel got a work-group barrier |

### Work-group loop

The normal work-group loop runs all work-groups on all QPUs in lockstep: every iteration ends with a
barrier, and the group IDs are counted up in the kernel. In SIMT mode the QPUs run different
work-groups independently, so the loop is different:

- VC4CL writes one block of UNIFORMs per work-group a QPU runs, all of a QPU's blocks one after
  another. Each block ends with a *next group* flag (new kernel-header bit `NextGroupFlagUsed`, bit 17
  of `KernelUniforms`): 1 if another block follows, 0 for the QPU's last work-group.
- At the end of the kernel code, the kernel reads the flag and, if set, branches back to its start,
  which reads the next block. No group-ID arithmetic in the kernel, so 2- and 3-dimensional NDRanges
  need nothing special.
- The branch is marked as a work-group loop, like the normal one, so that analyses and loop
  optimizations don't treat it as a loop of the kernel code.

Without the loop, every launch could run only one work-group per QPU. Each launch costs about 0.13 ms
through the DRM backend (submit, wait, interrupt), which dominated: clpeak's bandwidth kernel was no
faster in SIMT mode than without it. To bound the UNIFORM memory, a launch holds at most 64K UNIFORM
words (e.g. about 10,900 work-groups of a kernel with 6 UNIFORMs); larger NDRanges use several
launches. Kernels compiled without the flag (e.g. with the `loop-work-groups` pass disabled) still
run, one work-group per QPU and launch.

### VC4CL changes

- `Kernel.cpp`: `CL_KERNEL_WORK_GROUP_SIZE` and the local-size checks and heuristics use the merge
  factor (16) for SIMT kernels. Local sizes above 16 or multi-dimensional local sizes are rejected
  with `CL_INVALID_WORK_GROUP_SIZE`. Without a given local size, VC4CL picks the largest divisor of the
  global size up to 16 (as before, OpenCL 1.2 requires the local size to divide the global size).
- `executor.cpp`: for SIMT kernels, every QPU runs its own work-groups, all with local ID start 0,
  through the work-group loop above. Without the loop flag, the groups are launched in batches of up
  to 12.

## Testing

Emulator (`VC4CL_EMULATOR=1`) and hardware (vc4 DRM backend), 1-D kernels `y = a·x + b` (float), clamp of a sum (int with selects),
and an ID kernel writing `group_id·1000 + local_id + 100000·local_size`, each with a guard region after
the output buffer:

- local sizes 16, 13, 12, 8, 5, 1 and NULL; 1 to 12,000 work-groups (the latter in two launches): all
  correct, guard region untouched;
- local size 17: rejected with `CL_INVALID_WORK_GROUP_SIZE`;
- `y[i] = y[i] * 2 + c` (read and write of the same buffer, through VPM DMA): correct for the same
  local sizes;
- a kernel with `if (i < n)`: falls back to the normal mode and stays correct;
- `VC4C_NO_SIMT=1`: same results in the normal mode.

## Results

Raspberry Pi 3B, V3D at 300 MHz, vc4 DRM backend.

clpeak (local size chosen by clpeak):

| | before | SIMT |
|---|---|---|
| `float` compute | 0.61 GFLOPS | 6.60 GFLOPS (10.8×) |
| `int` compute | 0.18 GIOPS | 2.01 GIOPS (11×) |
| `float16` compute (not SIMT, for reference) | 6.12 GFLOPS | 6.18 GFLOPS |

clpeak's scalar global-bandwidth kernel (16 loads per work-item, `A[id + i * local_size]`), timed
separately on 12 MB because clpeak's own bandwidth test couldn't allocate its 2 × 64 MB buffers on this
system:

| | GB/s |
|---|---|
| without SIMT, local size 12 | 0.103 |
| SIMT, local size 12 | 0.799 |
| SIMT, local size 16 | 0.915 (8.9×) |

Scalar `float` code now slightly beats `float16`. The bandwidth is still far below DRAM speed: a QPU
runs a single thread, so memory latency is only hidden by issuing several loads before waiting for
them, which VC4C does in few cases.

## Future work: divergent branches

Divergent control flow (for example the bounds check `if (gid < n) …`, or loops with a per-work-item
trip count) needs execution masks:

- every block runs with a mask of active lanes;
- `if`/`else` run both sides with complementary masks;
- loops run until no lane is active;
- loads in masked regions first give every inactive lane a safe address with a conditional register
  write, then write `tmu0_s` unconditionally. When no lane is active, branch around the load: a fully
  masked request is still issued;
- stores in masked regions DMA only the active lanes. A prefix of active lanes (the bounds check)
  needs one shorter DMA. Any other mask needs one 1-word DMA per active lane, with the start column
  set to that lane: correct for any address pattern, but up to 16 DMAs per store. A read-modify-write
  of whole rows would be faster, but races with other work-items writing the same row.

Other extensions: 2- and 3-dimensional work-groups (needs the local-ID packing in VC4CL's
`executor.cpp` and the emulator's `buildUniforms` to handle per-lane x/y/z), non-contiguous accesses
(per-lane TMU addresses work as is; stores need the per-lane DMA above).
