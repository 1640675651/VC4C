# SIMT mode: one work-item per SIMD lane

Status: implemented (`src/normalization/SIMT.cpp`), including divergent branches and divergent loops,
with work-groups of up to 192 work-items. Verified in the emulator and on a Raspberry Pi 3B through
the vc4 DRM backend.

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
compute 9.36 GFLOPS (15×), scalar `float` bandwidth 8.9× higher.

## Execution model

In SIMT mode **a QPU runs 16 work-items at a time, one per SIMD lane**:

- a work-group has at most 192 work-items (12 QPUs × 16 lanes), in any number of dimensions;
- the work-groups are split into *chunks* of up to 16 consecutive work-items in x of one row (same
  local IDs in y and z; the last chunk of a row may be shorter). VC4CL deals all chunks of all work-groups round-robin across the 12 QPUs: QPU *q* runs
  the chunks *q*, *q* + 12, *q* + 24, … one after the other within a single launch (see "Work-group
  loop"). One large work-group uses all QPUs, and many small ones run side by side;
- SIMT kernels have no barriers or `__local` memory, so it doesn't matter which QPU runs which chunk,
  or when;
- in a chunk shorter than 16, the lanes beyond the work-group's end are inactive: they compute
  garbage, but never store it (see below). Every QPU knows its active lanes from the uniforms:
  `min(16, local_size − first_local_id)`.

### Modes and work-group size limits

The chunk scheduling works for any kernel whose work-items never interact (no barriers, no `__local`
memory), also for kernels which don't qualify for SIMT mode, in both modes: their chunks are a single
work-item, in any dimension. Only kernels with barriers or `__local` memory still need all work-items
of a work-group running at the same time, one per QPU, so at most 12 per work-group.

The limits are the same in both modes, the mode is chosen per process, before anything is compiled:

| | SIMT mode (default) | classic mode (`VC4CL_NO_SIMT=1`) |
|---|---|---|
| `CL_DEVICE_MAX_WORK_GROUP_SIZE` | 192 | 192 |
| `CL_DEVICE_MAX_WORK_ITEM_SIZES` | 192, 192, 192 | 192, 192, 192 |
| `CL_KERNEL_WORK_GROUP_SIZE`, SIMT kernels | 192 | (no SIMT kernels) |
| `CL_KERNEL_WORK_GROUP_SIZE`, other kernels with independent work-items | 192 | 192 |
| `CL_KERNEL_WORK_GROUP_SIZE`, kernels with barriers or `__local` memory, with work-item loops | 192 | 192 |
| `CL_KERNEL_WORK_GROUP_SIZE`, other kernels with barriers or `__local` memory (64-bit values kept across barriers, or loops disabled) | 12 | 12 |

A kernel may report a lower limit than the device, as OpenCL allows. In classic mode, VC4CL compiles
every kernel with `--fno-simt`. (Until 2026-10-06, classic mode reported 12 for the device and for all
kernels.) Kernels with barriers or `__local` memory get work-item loops if supported (see roadmap item
3), so their QPUs loop over up to 16 work-items each.

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
| `tmu0_s` write with some lanes at address 0 | **Does not disable those lanes, and the other lanes' values come back wrong** (inferred: the kernel's stores addressed with them went astray, and loading a valid address instead fixed it). Found with OpenCL-CTS, see "Known bugs" in the roadmap. |
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
- **Never address 0.** Lanes that load nothing useful must still read a valid address, never 0 (see
  the table). VC4C now uses the first lane's address for them.

## Implementation

### Which kernels use SIMT mode

`vectorizeWorkItems` (`src/normalization/SIMT.cpp`) runs early in normalization, before the
work-item functions are intrinsified. It converts a kernel only if:

- all its locals are scalars of at most 32 bits or pointers (no vector types, no 64-bit values);
- it uses no atomics, mutexes, semaphores, async copies/prefetches, `get_global_linear_id`,
  `printf` (which writes one record per QPU) or stack allocations (private arrays). Barriers and
  `__local` memory (parameters or variables) are supported through the work-item loops (roadmap item
  3, step 5), i.e. unless they are disabled;
- every memory access reads or writes a single element; no `memcpy`/`memset`-style accesses;
- every work-item dependent address has a lane stride equal to the element size (`p[gid + c]`,
  `p[get_local_id(0)]`, …), i.e. the 16 lanes access 16 consecutive elements. The analysis knows
  that the local ID in x is a multiple of 16 in lane 0 (the chunks start at multiples of 16), so
  with `tid = get_local_id(0) + 16 * k * get_local_id(1)`, `tid / 16` (and `tid >> 4`) is the same
  in all lanes and `tid % 16` (`tid - tid / 16 * 16`) has stride 1, as in CLBlast's copies to
  `__local` memory. Values computed only from such values are the same in all lanes too;
- no value is written through a work-group uniform address (all lanes would race for one location);
- branches depending on a work-item dependent value (**divergent branches**) start a region with a
  single entry, laid out between its entry and its merge point; loops in it need to be consecutive
  blocks with the only back edge from the last one (see "Divergent branches" below). Uniform
  branches and loops outside such regions are unrestricted, per-lane `select`s are fine;
- stores inside divergent regions write 32-bit values;
- work-item IDs are queried with a constant dimension (the IDs in y and z are the same for all
  lanes of a chunk);
- a `reqd_work_group_size`, if given, is at most 192 work-items (any dimensions);
- barriers are not in divergent code.

Otherwise the kernel is compiled as before, and the log says why (INFO level, e.g.
`SIMT: not using SIMT mode for kernel 'bounded': divergent branch`).

### Divergent branches

A conditional branch on a work-item dependent value can go different ways for the lanes of a QPU.
The branch and the blocks up to its merge point (its immediate post-dominator) form a *divergent
region*. The region is linearized: its blocks run one after the other, each only for the lanes that
reach it.

- **Masks.** Every block of the region gets a mask of active lanes (a `<16 x bool>`). The branch
  block and every block inside the region add their lanes to the masks of their successors, ANDed
  with the branch condition or its inverse, instead of branching. The first mask is the entry mask:
  the lanes below the local size.
- **Skipping.** Each block starts with a uniform branch past it if its mask is empty. This also
  keeps loads safe: a block runs only if at least one lane is active, and all addresses are
  contiguous across the lanes, so every load address is within 15 elements of an active lane's
  address.
- **Values.** A value written in a block and used in another block (typically the moves of a
  phi-node) is written to a temporary inside the block. At the end of the block the temporary is
  moved into the value for the active lanes only (a conditional move). Values only used inside the
  block need nothing. The analysis treats such values as work-item dependent (the phi-node of a
  divergent if-else differs between the lanes even if both sides are uniform).
- **Stores.** A VPM row write can't be masked. If all lanes of the work-group are active, the store
  runs as before. Otherwise a loop over the 16 lanes stores every active lane's element with its own
  single-word DMA. Bounds checks only take that path in the one QPU where the active lanes end.
- **Loads** need no change.

Example: `if (i < n) y[i] = x[i] * 3;` becomes

```
mask_then = entry_mask & (i < n)         // instead of the branch
if (none of mask_then) goto end          // uniform branch
load x (16 lanes), multiply
if (mask_then == entry_mask) store 16 lanes (DMA width: local size)
else for each lane in mask_then: store 1 word
end:
```

**Loops.** A region may contain loops. If the divergent branch is inside a loop and its lanes may
leave the loop in different iterations (a loop with a work-item dependent trip count, a `break`
depending on the work-item), the region grows to contain the whole loop: its entry is then the block
before the loop. Inside the region:

- the lanes taking the loop's back edge are collected in a *continue mask*;
- at the end of the loop (its latch), the continue mask becomes the header's mask, the masks of the
  loop's other blocks are reset, and a uniform branch repeats the loop while any lane is left;
- lanes leaving the loop add themselves to the mask of the exit block, accumulated over the
  iterations. Values written in the loop follow the rule above, so a lane that left keeps its final
  values;
- loads and stores in the loop follow the rules above. An address changing in a divergent loop
  (`p[i + k * n]` with a per-lane loop counter `k`) isn't contiguous across the lanes, so such kernels
  fall back to the normal mode.

Not supported (the kernel falls back to the normal mode): control flow where a region isn't laid out
between its entry and merge point, loops with several back edges or entered in the middle, and
overlapping regions.

### Vector kernels

A kernel using vector types of a single width N (2, 4 or 8, 32-bit elements) runs 16 / N work-items per
QPU, each in N adjacent SIMD lanes: lane = work-item × N + element. (16-element vectors already use
all lanes.) The kernel header's merge factor is 16 / N.

| Value | Layout |
|---|---|
| vector of a work-item | its N lanes |
| scalar of a work-item | repeated in its N lanes |
| uniform scalar | all lanes, as before |
| uniform vector | repeated for every work-item |

The front-end has already turned vector element accesses into rotations and conditional moves on the
SIMD lanes when this pass runs. They are adapted:

- the element number (`elem_num`) the front-end uses for element accesses becomes the element index
  within the work-item (`elem_num & (N − 1)`), so inserting element k sets element k of every work-item;
- replicating element 0 (splats, element extraction) replicates the first lane of every work-item
  within its lanes, and so does taking element 0 of a vector;
- rotations are kept, since the lanes read afterwards never cross the boundary between work-items.
  Other uses of rotations (real shuffles) make the kernel fall back to the normal mode;
- loads and stores of a work-item's own vector (`p[gid]` with `floatN* p`) access 16 consecutive
  elements, like scalar SIMT. A scalar load of consecutive elements is expanded (work-item w's value
  moves from lane w to its lanes), the load of one element of consecutive vectors (a vector load split
  up by the front-end) is replicated from the work-item's first lane, and a scalar store is compacted
  (lane w × N to lane w) and stores 16 / N elements.

Not supported in vector kernels yet (the kernel falls back): mixed vector widths, vector parameters,
non-splat vector constants, calls other than the work-item functions, divergent control flow, vector
stores to uniform addresses and scalar stores to every N-th element.

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
| `KernelMetaData.h` | Maximum work-group size = 192 for all kernels (used for the value ranges of the work-item IDs, only known to be smaller after normalization); maximum instances = 12 QPUs |
| `optimization/ControlFlow.cpp` (`addWorkGroupLoop`) | Loop over independent work-items instead of the normal one, for SIMT kernels and all kernels without barriers and `__local` memory (see below) |
| `tools/Emulator.cpp` (`buildUniforms`) | Deals the chunks of all work-groups across the QPUs like VC4CL |
| `periphery/VPM.cpp` (`lowerWriteRAM`) | A 16-wide store DMAs `min(16, local_size − first_local_id)` words instead of 16 |
| `normalization/MemoryAccess.cpp` | The check for memory dependencies between work-items accounts for the 16 work-items per QPU. Without it every SIMT kernel got a work-group barrier |

### Work-group loop

The normal work-group loop runs all work-groups on all QPUs in lockstep: every iteration ends with a
barrier, and the group IDs are counted up in the kernel. Kernels with independent work-items (all
SIMT kernels, and all other kernels without barriers and `__local` memory) let the QPUs run chunks of
different work-groups independently, so their loop is different:

- VC4CL writes one block of UNIFORMs per chunk a QPU runs (with the chunk's group IDs and first local
  ID), all of a QPU's blocks one after another. Each block ends with a *next group* flag (new
  kernel-header bit `NextGroupFlagUsed`, bit 17 of `KernelUniforms`): 1 if another block follows, 0
  for the QPU's last chunk.
- At the end of the kernel code, the kernel reads the flag and, if set, branches back to its start,
  which reads the next block. No group-ID arithmetic in the kernel, so 2- and 3-dimensional NDRanges
  need nothing special.
- The branch is marked as a work-group loop, like the normal one, so that analyses and loop
  optimizations don't treat it as a loop of the kernel code.

Without the loop, every launch could run only one work-group per QPU. Each launch costs about 0.13 ms
through the DRM backend (submit, wait, interrupt), which dominated: clpeak's bandwidth kernel was no
faster in SIMT mode than without it. To bound the UNIFORM memory, a launch holds at most 64K UNIFORM
words (e.g. about 10,900 chunks of a kernel with 6 UNIFORMs); larger NDRanges use several launches. Kernels compiled without the flag (e.g. with the `loop-work-groups` pass disabled) still
run, one work-group per QPU and launch.

### VC4CL changes

- `common.cpp`, `Device.cpp`: SIMT or classic mode (`VC4CL_NO_SIMT`) and the device limits, see
  "Modes and work-group size limits".
- `Kernel.cpp`: `CL_KERNEL_WORK_GROUP_SIZE`, the local-size checks and the heuristics use 192 for SIMT
  kernels with the work-group loop (16 without it). Larger local sizes are rejected with
  `CL_INVALID_WORK_GROUP_SIZE`, multi-dimensional ones only for SIMT kernels without the loop. Without a given local size, VC4CL picks the largest
  divisor of the global size up to that limit (OpenCL 1.2 requires the local size to divide the
  global size).
- `executor.cpp`: for kernels with independent work-items, the chunks (16 work-items in SIMT mode,
  otherwise 1) of all work-groups are dealt across the QPUs through the work-group loop above. SIMT
  kernels without the loop flag run one work-group per QPU, launched in batches of up to 12.
- `Program.cpp`: the `--fsimt`/`--fno-simt` build options, and `--fno-simt` in classic mode.

## Testing

Emulator (`VC4CL_EMULATOR=1`) and hardware (vc4 DRM backend), 1-D kernels `y = a·x + b` (float), clamp of a sum (int with selects),
and an ID kernel writing `group_id·1000 + local_id + 100000·local_size`, each with a guard region after
the output buffer:

- local sizes 192, 100, 48, 32, 16, 13, 12, 8, 5, 1 and NULL; 1 to 12,000 work-groups (the latter in
  two launches): all correct, guard region untouched;
- local size 193: rejected with `CL_INVALID_WORK_GROUP_SIZE`; in classic mode, local size 16 too;
- `y[i] = y[i] * 2 + c` (read and write of the same buffer, through VPM DMA): correct for the same
  local sizes;
- divergent kernels (bounds check, data-dependent if-else with a phi-node, nested if with a select,
  early return, a loop LLVM turns into a formula), with n below, inside and above the range and
  negative: all correct, guard region untouched;
- divergent loops (collatz with two exit conditions, a store inside a divergent loop, a divergent
  loop with a divergent `if` and a `break` inside a bounds check, a uniform loop around a divergent
  `if`), local sizes 1 to 192 and up to 1,024 work-groups: all correct;
- a reduction kernel with `__local` memory and a barrier (not SIMT): correct, reports 12; a kernel
  with a `__local` array but no barrier: not SIMT, reports 12;
- a 2-dimensional ID kernel (built with `--fno-simt`, so its work-items run one per chunk) with
  work-groups of 8×8, 16×12, 12×2, 16×4 and 6×6: all correct;
- vector kernels (widths 2, 4 and 8): vector loads and stores, splats of per-work-item scalars, uniform
  vectors built from parameters, element extraction and horizontal sums, vectors built from scalar
  loads, a loop of `mad`s like clpeak's, a swizzle LLVM turns into element accesses: bit-identical to
  the normal mode for local sizes 192, 16, 12, 5 and NULL;
- `VC4C_NO_SIMT=1`: same results in the normal mode.

## Results

Raspberry Pi 3B, V3D at 300 MHz, vc4 DRM backend.

clpeak (local size chosen by clpeak from `CL_DEVICE_MAX_WORK_GROUP_SIZE`):

| | before | scalar SIMT, device limit 192 | with vector SIMT |
|---|---|---|---|
| `float` | 0.61 | 9.29 | 9.36 GFLOPS (15×) |
| `float2` | 1.19 | 1.16 | 8.95 GFLOPS (7.5×) |
| `float4` | 2.27 | 2.23 | 8.24 GFLOPS (3.6×) |
| `float8` | 3.98 | 4.06 | 5.75 GFLOPS (1.4×) |
| `float16` (not SIMT) | 6.12 | 6.25 | 6.35 GFLOPS |
| `int` | 0.18 | 2.82 | 2.81 GIOPS (16×) |
| `int2` | 0.36 | 0.35 | 2.76 GIOPS (7.7×) |
| `int4` | 0.69 | 0.68 | 2.58 GIOPS (3.7×) |
| `int8` | 0.85 | 0.79 | 1.88 GIOPS (2.2×) |
| `int16` (not SIMT) | 1.43 | 1.35 | 1.35 GIOPS |

All widths except 16 now use all SIMD lanes. The wider vectors gain less, since their work-items
replicate and rotate more lanes for the horizontal sum at the end and run fewer work-items per QPU.

clpeak's scalar global-bandwidth kernel (16 loads per work-item, `A[id + i * local_size]`), timed
separately on 12 MB because clpeak's own bandwidth test couldn't allocate its 2 × 64 MB buffers on this
system:

| | GB/s |
|---|---|
| without SIMT, local size 12 | 0.103 |
| SIMT, local size 12 | 0.799 |
| SIMT, local size 16 | 0.915 (8.9×) |

Divergent kernels (393,216 work-items, 300,000 of them taking the branch):

| kernel | normal mode, local size 12 | SIMT, local size 16 |
|---|---|---|
| `if (i < n) y[i] = x[i] * 3` | 82.5 ms | 14.7 ms (5.6×) |
| data-dependent if-else, `y[i] = v` after the merge | 251.5 ms | 23.6 ms (10.6×) |
| `if (i >= n) return; y[i] = x[i] + 1` | 82.5 ms | 17.9 ms (4.6×) |

Divergent loops (196,608 work-items):

| kernel | normal mode, local size 12 | SIMT, local size 192 |
|---|---|---|
| collatz, iteration count per work-item (up to 100) | 484.5 ms | 57.9 ms (8.4×) |
| uniform loop (100 iterations) around a data-dependent if-else | 387.5 ms | 30.1 ms (12.9×) |
| store inside a loop with a per-work-item trip count (0 to 7) | 108.5 ms | 39.8 ms (2.7×) |

Scalar `float` code is now faster than `float16`. The bandwidth is still far below DRAM speed: a QPU
runs a single thread, so memory latency is only hidden by issuing several loads before waiting for
them, which VC4C does in few cases.

### CLBlast GEMM

CLBlast's `Xgemm` kernel (single precision) with parameters chosen for SIMT mode, 256 × 256 × 256,
kernel time only (`cts-results/clblast/xgemm_time.c`, results checked against the host):

| `Xgemm` parameters | mode | time | GFLOPS |
|---|---|---|---|
| no `__local` memory (`SA = SB = 0`), `STRM = STRN = 1`, 16 × 4 work-items, 32 × 16 tile | SIMT | 32 ms | 1.04 |
| the same | classic | 445 ms | 0.075 |
| tiles in `__local` memory (`SA = SB = 1`), otherwise the same | SIMT, looping over chunks | 460 ms | 0.073 |

Without `__local` memory, every work-item loads its elements of A (consecutive across the lanes,
through the TMU) and of B (one element per chunk, the same for all lanes) directly, and the kernel
has no barriers: 14× faster than classic mode. With `__local` memory, the tiles go through RAM by
VPM DMA (SIMT kernels keep `__local` memory in RAM, see roadmap item 3), and the values kept across
the two barriers per tile are stored to RAM, so the kernel is as slow as classic mode. CLBlast's
parameter database got a `Broadcom` entry with the first parameter set, and a minimum size of 16 for
the indirect `Xgemm` path (default 896, so the tests used only `XgemmDirect`). CLBlast's SGEMM test
then passes all 234 cases (278 skipped: CBLAS has no error codes to compare with). The whole
`clblast_client_xgemm` run of 256³ takes 320 ms: CLBlast's helper kernels which pad and transpose the
matrices for `Xgemm` take most of the time (they have barriers and `__local` memory and run with
work-item loops).

## Caveats

Current state (2026-10-03). Kernels that don't qualify for SIMT mode are compiled and run exactly as
before, so most of these only matter for SIMT kernels.

### Which kernels benefit

- **Divergent control flow with some limits.** Bounds checks, early returns, if-else (also nested)
  and loops with work-item dependent exits run in SIMT mode. Kernels fall back to the normal mode for
  irregular control flow (a region not laid out between its entry and merge point, loops entered in
  the middle or with several back edges), for addresses changing inside a divergent loop, and for
  stores of 8- or 16-bit values in divergent code.
- Values written inside divergent code are moved to their final location under the block's mask at
  the end of every block, and every block checks its mask first: a few extra instructions per block
  and iteration.
- Stores in divergent code where not all lanes are active (the QPU where a bounds check ends, or
  data-dependent conditions) store lane by lane: up to 16 single-word DMAs instead of one.
- **No atomics, private arrays, vector types, 64-bit values or non-contiguous accesses**
  (`p[2 * gid]`, `p[idx[gid]]`, transposes): these kernels fall back too. Kernels with barriers or
  `__local` memory run in SIMT mode if they qualify otherwise, looping over the chunks of their
  work-group; all values differing between the lanes which are kept across a barrier are stored to
  and loaded from RAM at every barrier, so kernels with many such values and barriers in loops
  spend time on that.
- Kernels whose work-item IDs only come from dimension 0. IDs of dimensions 1 and 2 are uniform
  within a chunk and allowed (tested with 2- and 3-dimensional NDRanges, also with global work
  offsets: OpenCL-CTS `global_work_offsets`).

### Work-group sizes and OpenCL conformance

- SIMT kernels accept work-groups of any dimensions, up to 192 work-items. A chunk only covers one
  row in x, so narrow work-groups waste lanes: 18 × 10 runs 10 chunks of 16 and 10 of 2 work-items,
  4 × 4 uses 4 of 16 lanes.
- **Kernels keeping 64-bit values across barriers report `CL_KERNEL_WORK_GROUP_SIZE` 12, below the
  device limit of 192** (see roadmap item 3). OpenCL allows this, but applications which use the
  device limit as their local size without checking the kernel's limit fail with
  `CL_INVALID_WORK_GROUP_SIZE` for such kernels. Classic mode has the same limits. All other kernels
  accept 192 in either mode, also if they don't qualify for SIMT mode.
- Without a given local size, VC4CL picks the largest divisor of the global size up to 192: a global
  size of 100 gives a single work-group of 100 (7 chunks, the last with 4 active lanes).

### Memory and safety

- Inactive lanes (beyond the local size) still issue loads: up to 15 elements (60 bytes) past the last
  element a work-group accesses. That's RAM next to the buffer, the values are discarded, and no
  peripheral can be reached this way, but tools that check buffer bounds would see these reads.
- Every store takes the GPU-wide VPM mutex, as in the normal mode, so stores of the 12 QPUs are
  serialized.
- Memory bandwidth is limited by latency, not by SIMT: each QPU runs one thread, and VC4C rarely
  issues several loads before waiting for the first one.

### Run-time behavior

- A single launch now runs up to about 10,000 chunks (64K UNIFORM words). While a compute job
  runs, the vc4 compute driver holds back OpenGL jobs, so long kernels can make the desktop stutter
  (the normal-mode work-group loop has the same effect).
- The launch timeout is 1 s per chunk of the launch, at least 30 s, capped at the driver maximum of
  10 minutes. A hanging SIMT kernel with many chunks may take up to 10 minutes to be reset.
- The UNIFORM buffer grows with the number of chunks per launch: up to 2 × 256 KB of contiguous GPU
  memory per launch.

### Compatibility and tools

- Binaries from this VC4C need this VC4CL: older VC4CL versions don't write the "next group" flag
  (`NextGroupFlagUsed`, bit 17), so SIMT kernels would read garbage. Older binaries (without the flag
  or merge factor) run unchanged on the new VC4CL. Install both together.
- VC4C's own emulator (`tools/Emulator.cpp`, used by VC4C's tests) deals chunks across the QPUs like
  VC4CL. This code is untested: VC4CL's emulator mode (`VC4CL_EMULATOR=1`, used for all tests here)
  builds its own UNIFORMs, and VC4C's tests aren't built on this system.
- VC4CL's debug memory dumps and performance counters haven't been tested with SIMT kernels.

### Not SIMT related, but seen during testing

- clpeak's global-bandwidth test fails to allocate its 2 × 64 MB buffers (`clCreateBuffer` returns
  -5) on this system. VC4CL reports half of the CMA area as the maximum allocation size, and under
  memory pressure the kernel can't free 64 MB of contiguous memory. Reporting a quarter would let
  clpeak pick 2 × 32 MB.

## Roadmap to a conformant SIMT implementation

Goal: the OpenCL conformance tests (OpenCL-CTS) pass in SIMT mode exactly as in classic mode, and as
many kernels as possible run in SIMT mode. Kernels which don't qualify fall back to the classic code,
so most items below are about coverage and speed; correctness depends on the fallback being exact,
and on the compiler being correct in both modes (see step 1).

### 1. Verification first

- **Run OpenCL-CTS** (results of all suites run so far: VC4CL's `CONFORMANCE.md`;
  github.com/KhronosGroup/OpenCL-CTS; VC4CL already contains workarounds for
  its 1.2 version, `cl12_trunk`) in both modes and compare the pass lists. Start with `basic`,
  `compiler`, `api`, `vectors`, `relationals`, `commonfns`, `integer_ops` and `geometrics`; the
  `math_brute_force` and `conversions` suites take hours to days on this GPU. Build only the
  selected test directories, single job (`make -j1`).
  Status: `test_basic` (current CTS main) passes all 64 applicable subtests of 112 in both modes,
  with 48 skipped (images, 64-bit integers, OpenCL 2.0). It found two SIMT bugs: a work-group uniform
  value stored at a per-work-item address (`out[id] = sum`) was written by the first lane only
  (`constant`), and SIMT kernels rejected multi-dimensional work-groups (`global_work_offsets`). It
  also found 6 failures in both modes, all fixed: out-of-range dimensions of `get_local_size()` and
  `get_local_id()` (both `work_item_functions_out_of_range` variants), VC4CL's rectangular buffer
  copies (`bufferreadwriterect`), three loop vectorizer bugs (`vload_local`), and TMU loads with lanes
  at address 0 (`vstore_local`, `vstore_private`, see "Known bugs").
  `test_api` passes all 95 applicable subtests of 164 in both modes, with 69 skipped. In classic mode,
  the 3 `work_group_suggested_local_size` subtests failed while classic mode reported a maximum
  work-group size of 12: their "odd sizes" case needs an odd non-prime work-group size (counting 1 as
  prime) below the device maximum, and there is none below 17. They pass since classic mode reports
  192 (roadmap item 3, step 1). The first run had 22 failures in classic mode and 20 in SIMT mode,
  plus a hang in both; all are fixed:
  - VC4C rejected kernels with a `reqd_work_group_size` above 12 work-items at compile time
    (`null_required_work_group_size`, `kernel_required_group_size`). OpenCL requires them to compile;
    only launching an unsupported size fails. The number of instances (stack frames, spill areas) is
    now capped at the number of QPUs.
  - VC4CL reported image and sampler limits without image support, which must be 0 (8 `min_max_*`
    subtests), and the whole GPU memory as `CL_DEVICE_LOCAL_MEM_SIZE`, which programs then failed to
    allocate (`min_max_local_mem_size`; now 32 KB, the full profile minimum).
  - `CL_KERNEL_LOCAL_MEM_SIZE` didn't include the `__local` arguments (`kernel_local_memory_size`),
    and a `__local` argument of size 0 returned the wrong error (`local_arg_size_zero`).
  - Missing argument validation (`negative_*`): command-queue properties (invalid bits, out-of-order
    execution, which the device doesn't support), context properties (invalid values, duplicate
    names) and device types.
  - Events in a wait list were neither checked for validity nor for belonging to the command-queue's
    context. A user event of another context in the wait list of a marker blocked the queue forever
    (`negative_enqueue_marker_with_wait_list`).
  `test_printf`: `printf` is now implemented (VC4C `normalization/Printf.cpp` writes records into a
  hidden buffer parameter, VC4CL `printf.cpp` formats them after the kernel finished). All 15
  applicable subtests pass in both modes, 7 are skipped (double, half, 64-bit integers). The CTS
  expects an RTZ device to round toward zero also when formatting floats (VC4CL does) and when
  converting float literals (it parses them with `strtof` in RTZ mode). clang rounds literals to
  nearest and ignores `#pragma STDC FENV_ROUND` for SPIR, so VC4C now rewrites the float literals of
  the OpenCL C source into the hexadecimal literals of their values rounded toward zero before
  running clang (`precompilation/FloatLiterals.cpp`), which C99 allows. Before, 3 subtests failed in
  the last digit of some values.
  `test_relationals` passes all 17 subtests in both modes after the register allocation fixes (see
  "Known bugs"); `test_commonfns` fails only `mix` and `mixf` (RTZ results against the CTS's absolute
  error bound).
- **Check results against independently computed values**, not only SIMT against classic mode:
  both modes share the front-end and most of the backend, so a compiler bug produces the same wrong
  result in both (see the `switch` bug below, which a mode comparison didn't catch).
- **Check which mode every test kernel actually used** (the INFO log line), since results alone
  don't show a silent fallback.
- **Run VC4C's own test suite** (`testing/`) with its emulator. Its `buildUniforms` now deals chunks
  across the QPUs like VC4CL, untested so far.

### 2. Known bugs (both modes)

- Fixed: **unused lanes of TMU loads read address 0, which corrupted memory.** VC4C groups loads of
  neighboring addresses into one TMU load of several lanes (`optimization/Memory.cpp`) and loads with
  a dynamic element count (`periphery/TMU.cpp`). It set the address of the unused lanes to 0, assuming
  the TMU skips them; VC4C's emulator does. On the hardware, the values loaded by the used lanes came
  back wrong (inferred from the A/B test below; the values themselves weren't captured). In OpenCL-CTS `test_basic` `vstore_private` and `vstore_local`, `offsets[tid]` was such
  a grouped load (two loads of the same address, lanes 2 to 15 at address 0), and the kernel computed
  the address of its final DMA store from it:
  - the result of some work-items never arrived (in single launches always for work-item 16, never for
    0 to 15; with 12 QPUs in about 1 of 10 runs, the vectors of one QPU program);
  - the misdirected DMA writes went to other RAM: GPU timeouts, and on the Raspberry Pi corrupted page
    tables, crashed processes and hung systems, also after runs that seemed to pass.

  Now the unused lanes load the first lane's address, which is valid and on a cache line loaded
  anyway. Verified on the GPU with an A/B test of the same kernel: lost results with address 0, correct
  with the first lane's address. The CTS kernel that failed in 2 of 20 runs passed 20 of 20, and
  `vstore_private` and `vstore_local` pass in both modes. Since the emulator can't show this, it was
  found on the hardware: single work-items launched with a global offset (failing deterministically),
  kernel variants with one part removed at a time, and `--fno-group-memory` (VC4CL now passes VC4C's
  `--f…` optimization options from the build options to VC4C). The host-checked harness and the logs
  (`vs_check.c`, `exp.sh`, `experiments.log`) are in `cts-results/tools/` next to this repository.

- Fixed: **irregular integer types were widened without masking.** LLVM narrows values to types like
  `i2` (e.g. for `switch (v & 3)`), which the front-end extends to `i8`. The truncation masked with
  `0xFF` instead of `0x3`, and negative constants were sign-extended, so `switch (v & 3)` took the
  default case for most values. Now the values of such types are kept zero-extended to their actual
  width (`llvm/BitcodeReader.cpp`): truncations, `add`/`sub`/`mul`/`shl` and constants are masked,
  `ashr`/`sdiv`/`srem`, signed comparisons and `sext` first sign-extend from the actual width.
- Fixed: **missing register interferences.** The interference graph (`analysis/InterferenceGraph.cpp`)
  walks every block backwards from its live-out locals and only added edges when a local became live
  (at a read). Two locals live at the end of a block because different successors need them, but
  never read while the other one is live, never interfered: e.g. a kernel argument read in some cases
  of a `switch` and the switch's result, written before the computed branch. They could get the same
  register, overwriting the argument. Now a local written while other locals are live interferes with
  all of them (the classic rule). Since VC4C iterates hash maps keyed by pointers, the allocation order
  depends on memory addresses, so this showed up only for some compilations (about 3 % of the second
  compilation in a process for the test kernel). `VC4C_SINGLE_THREADED` (new) runs the compiler's
  per-kernel stages one kernel at a time, to rule out data races in such cases.

- Fixed: **two register-allocation fixups broke conditionally written values.** When the register
  allocation fails, VC4C retries after rewriting the code (`asm/RegisterFixes.cpp`). Which fixups run
  depends on the (nondeterministic) allocation order, so the bugs showed up only for some
  compilations: in OpenCL-CTS `test_relationals` `shuffle_function_call`, about 10 % of the
  compilations of one of its kernels returned zeros for some elements, deterministically per binary.
  The CTS test failed in about 1 of 3 runs, in different cases (`float4 to float2`,
  `uint4 to uint2`) since every run compiles its kernels anew.
  - `rematerializeConstants` moves a constant write next to its single reader. It didn't check that
    the constant is the only write of the local: a vector assembled from a zero initialization and
    16 conditional per-element writes (the TMU address offsets of a grouped load) got the
    initialization moved after the element writes, so all lanes loaded the first address. Now only
    locals with a single write are moved.
  - `groupScalarLocals` spills locals into the elements of a vector register, inserting each value
    with a flag-setting element selection. It inserted the spill code right after the write, where
    the flags could be live: several conditional instructions reusing one flag setter (merged by the
    optimizer) then used the spill's flags, and elements were zeroed. Now the spill code goes to the
    first position where the flags are dead, and the local isn't spilled if there is none in the
    block.

  Found by comparing the binaries of repeated compilations (all different) on the GPU, each run
  several times against a reference output: a wrong binary fails every time. Verified with 60
  compilations, all correct (before: 5 of 70 wrong). The harness (`shuffle/shb_check.c`, with
  `BINARY`, `REF`, `GLOBAL` and `LOCAL`) is in `cts-results/tools/`.

### 3. 192 work-items per work-group for every kernel, in both modes

Goal: every kernel accepts work-groups of up to 192 work-items, in SIMT and in classic mode, so the
device limits are the same in both modes and no kernel reports less than the device. Today kernels
with barriers or `__local` memory accept only 12 (one work-item per QPU, all running at the same
time), also in SIMT mode, where the device reports 192 (see "Work-group sizes and OpenCL
conformance"). OpenCL allows a lower per-kernel limit, but applications commonly size their
work-groups from the device limit or hard-code 64 to 256 work-items (e.g. CLBlast's default
parameters), and then fail with `CL_INVALID_WORK_GROUP_SIZE`. With the same limits, `VC4CL_NO_SIMT` becomes a pure performance and debugging switch, and both modes
can be compared with identical launches.

Desktop GPUs keep all work-items of a work-group resident at the same time (as warps or wavefronts
with their own registers in a large register file, up to 256 KB per compute unit) and implement
barriers in hardware. A QPU runs a single thread with 4 KB of registers and has no barrier other than
the semaphores between QPUs, so the compiler has to do in software what that hardware does: run the
work-items of a QPU one after another and switch between them at barriers, as CPU implementations
(e.g. pocl) do.

**Work-item loops.** For a kernel with barriers, VC4C splits the kernel into regions at the barriers
and wraps every region in a loop over the work-items assigned to the QPU (in SIMT mode: over the
chunks of 16 lanes), with the local ID as the loop variable. Between two regions, the QPUs of the
work-group synchronize through the existing semaphore barrier, once per region instead of once per
work-item. OpenCL requires all work-items of a work-group to reach the same barriers in the same
order, so the same region sequence is valid for all of them. 192 work-items on 12 QPUs are 16
work-items per QPU in classic mode, or a single chunk per QPU in SIMT mode.

- **Values live across a barrier** are stored per work-item: in registers if they fit (e.g. 16
  work-items with 3 values each), otherwise in a context buffer in RAM, written and read through DMA
  at the region boundaries. RAM always has room (192 × the live bytes), so no kernel needs a lower
  limit; kernels with many such values get slower with large work-groups.
  `CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE` keeps recommending the efficient sizes.
- **Private arrays and stack frames** are needed per work-item, i.e. per loop iteration.
- **Barriers in loops and branches** (e.g. the reduction loop `for (s = …) { …; barrier(…); }`) need
  regions which span loop boundaries: the loop around the barrier becomes a loop around the region
  sequence. This is the most complex part.
- **`__local` memory** stays per work-group. In classic mode one work-group runs at a time, so the
  existing VPM placement (or the RAM buffer for `__local` arguments) is unchanged. In SIMT mode up to
  12 work-groups run at the same time, so every one needs its own region: VC4C addresses `__local`
  memory relative to a per-work-group base address passed as a UNIFORM, instead of fixed VPM
  addresses. VC4CL decides at every launch, like a GPU dispatcher: it places the regions in the VPM
  (about 2–3 KB are left after VC4C's scratch area, of the 4 KB user programs can address because of
  hardware erratum HW-2253) if enough work-groups fit, limiting the number of work-groups running at
  the same time (occupancy); otherwise in a buffer in RAM, accessed through DMA (the TMU cache isn't
  coherent with other QPUs' writes). Local memory stays reported as `CL_GLOBAL`, since the VPM is too
  small for the advertised sizes.
- **Work-groups of up to 16 work-items in SIMT mode** are a special case: a single chunk on a single
  QPU, where `barrier()` only has to wait for outstanding stores, since the work-items are lanes of
  one instruction stream. Typical barrier code is divergent (`if (lid < s) tmp[lid] += tmp[lid + s]`),
  which is supported.

**Steps:**

1. **Done: classic mode, kernels without barriers or `__local` memory** accept 192 work-items, running
   the work-group in chunks of one work-item per QPU. VC4CL's chunk scheduling already did this; only
   the limits changed. Since a kernel can't report more than the device, the classic-mode device
   limits went to 192 at the same time (earlier than planned in step 5), so classic mode now has the
   same limits as SIMT mode, including the limit of 12 for kernels with barriers or `__local` memory.
   The `work_group_suggested_local_size` subtests of `test_api` now pass in classic mode too, and
   `test_basic` still passes.
2. **Done: work-item loops for kernels whose barriers are all at the top level** (executed exactly once
   on every path through the kernel), in both modes (kernels with barriers never use SIMT mode), with
   the values live across barriers in registers (`normalization/WorkItemLoops.cpp`). VC4C records the
   new limit (192) in the kernel's metadata (`max_work_group_size`), VC4CL then runs a work-group on
   `min(local size, 12)` QPUs. Details:
   - A QPU's work-items `q, q + Q, …` (`Q` QPUs) get their local IDs from a vector computed once per
     work-group (lane `k` for the `k`-th work-item); the work-item functions read the current
     work-item's IDs instead of the UNIFORM.
   - Values kept per work-item: locals live at a barrier (block-level liveness, store addresses count
     as reads) which may differ between the work-items, i.e. not computed only from literals,
     parameters and work-group uniform work-item functions in blocks executed by all work-items.
     Each takes one vector register, the work-item's value in its lane.
   - All barriers of such kernels synchronize the QPUs, not the work-items, also the barrier the
     work-group loop inserts at the end of a work-group.
   - Kernels with `__local` memory, but without barriers, are a single region: the whole kernel
     loops over the QPU's work-items.
   - Not yet supported, so these kernels keep the limit of 12: private arrays, values other than scalars of at most 32 bits or pointers kept across a barrier,
     more than 16 such values, and calls of `async_work_group_copy`, `wait_group_events`,
     `prefetch` and work-group functions (which insert their own barriers between the work-items).
   - Cost: about 170 to 300 more instructions per kernel (loop setup, saving and restoring the values
     at every barrier), and the loop overhead per work-item and region.
   - Verified with host-checked kernels (one barrier; three barriers with branches and a `return`
     after the last one) for work-groups of 1 to 192 work-items in 1, 2 and 3 dimensions, in the
     emulator and on the GPU in both modes.
   - It found two VC4C bugs affecting other kernels too: `PropagateMoves` inserted a register (e.g.
     the replication register r5) into an instruction unpacking its input, which only register-file A
     can do, so the unpack was dropped (`get_local_id(0)` returned all packed IDs); and the NOP
     replacement of the instruction reordering moved a write of a local above earlier reads of it (for
     locals written several times, e.g. loop counters). Both are fixed (`optimization/Eliminator.cpp`,
     `optimization/Reordering.cpp`); the pass also avoids the latter pattern itself.
   - CTS `test_basic`, `test_api` and `test_atomics` pass as before in both modes.
3. **Done: barriers in loops and (work-group uniform) branches.** A region starts at the start of the
   kernel or after a barrier and ends at the next barrier reached or at the end of the kernel, so with
   a barrier in a loop, the region after it ends at the same barrier again (next iteration) or at a
   later one. Since barriers are in work-group uniform control flow (required by OpenCL), all
   work-items of a QPU end a region at the same barrier. Where a barrier (or the end of the kernel)
   ends several regions, the loop over the work-items continues with the region the QPU is in, kept
   in a (QPU-uniform) variable. Values written in several places (e.g. loop counters, after phi
   elimination) are kept per work-item, so every work-item continues with its own copy.
   - The loop over the work-items enters the user's loop in the middle (irreducible control flow),
     where VC4C's dominator tree analysis got stuck: it now uses the iterative algorithm by Cooper,
     Harvey and Kennedy (`analysis/DominatorTree.cpp`), the post-dominator tree is unchanged.
   - Verified with a tree reduction (barrier in the loop, any work-group size) and barriers in a
     uniform branch (both outcomes), for work-groups of up to 192 work-items, in the emulator and on
     the GPU in both modes.
4. **Done: private memory and any number of values per work-item** (2026-10-07). Every work-item
   has its own stack frame: the stack allocations use the work-item's local linear ID instead of the
   QPU number (`KernelMetaData::workItemLoopFrameIndex`) and stay in RAM, VC4CL reserves one frame per
   work-item of the work-group. Of the values kept across barriers, up to 8 scalars (those kept
   across the most barriers) stay in the lanes of vector registers, all others (also vectors) are
   stored in private variables in the work-item's frame at the end of a region and loaded at the
   start of the next one. `async_work_group_copy` and `wait_group_events` need nothing special: the
   copy is done by the work-items with local ID 0 in x, and the wait is a barrier. Private variables
   of scalar or vector types which are only loaded and stored as a whole (clang keeps some, e.g. with
   life-time markers) become ordinary values before both passes
   (`promoteSimpleStackAllocations`). Kernels with a required work-group size of at most 12 don't
   loop. Verified with host-checked kernels (a private array, 24 values, `float4`/`float2`/`uchar`/
   `short`/`bool` values kept across a barrier, `async_work_group_copy` in both directions) for
   work-groups of 1 to 192 work-items, on the GPU in both modes. Only kernels keeping 64-bit values
   across barriers keep the limit of 12.
5. **Done: SIMT mode for kernels with barriers or `__local` memory** (2026-10-07). The SIMT
   conversion accepts them if the work-item loops are enabled, and the QPUs then loop over the
   *chunks* of their work-group instead of single work-items (the same chunks of up to 16
   work-items in x of a row as for independent work-items): QPU *q* runs the chunks *q*, *q* + *Q*,
   ... of the work-group, a barrier synchronizes the QPUs after all their chunks reached it. The
   values kept across barriers which differ between the lanes are vectors, so they are kept in RAM
   (per chunk), uniform ones (e.g. loop counters) in registers. The kernel code reads the local IDs
   of the current chunk instead of the UNIFORM (also for the active lanes of stores and divergent
   code). VC4CL runs one work-group at a time, on `min(chunks, 12)` QPUs, like in classic mode, so
   `__local` memory stays per program; in SIMT kernels it is always in RAM, since the VPM can't
   address 16 consecutive elements at any offset. Barriers in divergent code are rejected (OpenCL
   requires them in work-group uniform control flow). Verified with a 2D tiled matrix
   multiplication (tiles of 1×1 to 12×12) and a neighbour exchange through `__local` memory
   (work-groups of 1 to 192, several chunks per row) on the GPU, and with CLBlast's `Xgemm`
   (`SA = SB = 1`, see "CLBlast GEMM" in the results).
   - It found a bug of the register allocation fixups affecting all SIMT kernels: grouping
     parameters or scalars into the lanes of a vector register kept them only in the first lane
     where VC4C thought only that lane was used, but in SIMT mode they are combined with the values
     of all lanes (e.g. a base address with the per-lane offsets). Now they are replicated to all
     lanes again (`asm/RegisterFixes.cpp`).
   - With 16 values kept in registers per work-item, a classic-mode kernel (16 × 4 tiled matrix
     multiplication) read from a wrong address under the resulting register pressure (cause not
     found; such a read can freeze the system). The limit is 8 now.

   Not done: running several work-groups at the same time with per-work-group `__local` regions
   (in the VPM if they fit, otherwise in RAM), which would use more QPUs for work-groups of fewer
   than 12 chunks.
6. (Done with step 1: the classic-mode device limits are 192.)

### 4. Memory accesses

- **Non-contiguous loads** (`p[idx[gid]]`, `p[2 * gid]`, transposes): the TMU accepts a separate
  address per lane (VC4C's `customAddressCalculation`). Inactive lanes need a safe address (e.g. the
  address of an active lane), since TMU requests can't be masked.
- **Non-contiguous stores:** one single-word DMA per active lane, as the masked store loop already
  does; faster variants for strided patterns (VDW with a memory stride).
- **Addresses changing in divergent loops:** use per-lane addresses (above) instead of lane 0's.
- **8- and 16-bit stores in divergent code**, 64-bit types, private arrays (one stack frame per
  lane), atomics (serialize the active lanes).
- **Faster partial stores:** a prefix of active lanes (the end of a bounds check) needs one shorter
  DMA instead of the lane loop.

### 5. Control flow and work-item functions

- **Computed branches** (`switch` lowered to a jump table) in divergent code: convert to a chain of
  conditional edges, or handle as a multi-way edge in the region linearization.
- **Irregular control flow** (regions not laid out between their entry and merge point, loops
  entered in the middle, overlapping regions): reorder blocks before linearizing, or fall back as
  today.
- **Packing rows of narrow work-groups:** multi-dimensional work-groups work, but a chunk is one row
  in x, so a local size in x below 16 wastes lanes. Several rows per chunk need per-lane local IDs in
  x, y and z (VC4CL passes the chunk's first local ID; the lane's ID needs to be unflattened), and
  so do `get_global_linear_id` and `get_local_linear_id`.

### 6. Vector kernels

- Divergent control flow (the entry mask per work-item: lanes `w × N … w × N + N − 1`),
  mixed vector widths in one kernel, real shuffles (rotations within each work-item's lanes),
  vector parameters (tile them after loading), non-splat vector constants (repeat per work-item),
  calls to VC4C intrinsics working on vectors (`dot`, `length`, … need the per-work-item layout).
- Use the QPU's per-quad replication for N = 4 instead of 3 rotations.

### 7. Conformance details

- Device limits: `CL_DEVICE_MAX_WORK_GROUP_SIZE` 192 with lower per-kernel limits is allowed, but the
  goal is 192 for every kernel in both modes (item 3); check
  `CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE` (16 for SIMT kernels, 16 / N for vector kernels, 1
  otherwise) and `CL_DEVICE_MAX_WORK_ITEM_SIZES` against the CTS expectations.
- Inactive lanes read up to 60 bytes past the accessed elements. Harmless on this hardware (RAM, no
  MMU), but a buffer at the very end of the GPU memory could read beyond it.
- Long launches: one compute job runs up to about 10,000 chunks and blocks OpenGL meanwhile; a hang
  takes up to the driver's 10-minute limit to be reset. Consider shorter launches (time-sliced) for
  interactive systems.

### 8. Performance (not required for conformance)

- Memory latency: issue several TMU loads before waiting for the first (a QPU runs a single
  thread, so this is the only way to hide DRAM latency).
- Stores of the 12 QPUs are serialized by the GPU-wide VPM mutex; per-QPU VPM areas would allow
  concurrent DMA setup.
- clpeak's global-bandwidth test: report a maximum allocation size of a quarter of the CMA area, so
  it can allocate its buffers under memory pressure.
