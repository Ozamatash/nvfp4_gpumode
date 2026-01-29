Utilities for SM100
cutlass.utils.sm100.compute_epilogue_tile_shape(
cta_tile_shape: cutlass.cute.typing.Shape,
use_2cta_instrs: bool,
layout_d: LayoutEnum,
elem_ty_d: Type[cutlass.cutlass_dsl.Numeric],
*,
layout_c: LayoutEnum | None = None,
elem_ty_c: Type[cutlass.cutlass_dsl.Numeric] | None = None,
loc=None,
ip=None,
) → cutlass.cute.typing.Tile
Attempts to compute a reasonable epilogue tile based on block tile shape or allows the user to provide one.

Parameters
:
cta_tile_shape (cute.Shape) – A tuple or list representing the dimensions of the CTA tile, where cta_tile_shape[0] corresponds to the height (M) and cta_tile_shape[1] corresponds to the width (N) of the tile.
use_2cta_instrs (bool) – A flag indicating whether the configuration is for a 2SM setup.
layout_d (LayoutEnum) – The layout enum of the output tensor D.
elem_ty_d (Type[Numeric]) – The element type of output tensor D.
layout_c (LayoutEnum, optional) – The layout enum of the input tensor C. Defaults to None.
elem_ty_c (Union[Type[Numeric], None], optional) – The element type for input tensor C. Defaults to None.
Returns
:
Returns epilog tiler, which is used in subsequent epilog partitions.
Return type
:
cute.Tile
Raises
:
ValueError – If the computed tile cute.size does not meet minimum requirements based on CTA dimensions.
cutlass.utils.sm100.get_smem_store_op(
layout_d: LayoutEnum,
elem_ty_d: Type[cutlass.cutlass_dsl.Numeric],
elem_ty_acc: Type[cutlass.cutlass_dsl.Numeric],
tiled_tmem_load: TiledCopy,
*,
loc=None,
ip=None,
) → CopyAtom
Selects the largest vectorized smem store atom available subject to constraint of gmem layout and chosen TMEM_LOAD’s thread-value ownership.

Parameters
:
layout_d (LayoutEnum) – The layout enum of the output tensor D.
elem_ty_d (Type[Numeric]) – The element type for output tensor D.
elem_ty_acc (Type[Numeric]) – The element type for accumulator.
tiled_tmem_load (cute.TiledCopy) – An instance of TiledCopy that represents the tmem load operation.
Returns
:
Either SmemStoreMatrix or SimtSyncCopy, based on the input parameters.
Return type
:
cute.CopyAtom
cutlass.utils.sm100.get_tmem_load_op(
cta_tile_shape: cutlass.cute.typing.Shape,
layout_d: LayoutEnum,
elem_ty_d: Type[cutlass.cutlass_dsl.Numeric],
elem_ty_acc: Type[cutlass.cutlass_dsl.Numeric],
epi_tile: cutlass.cute.typing.Tile,
use_2cta_instrs: bool,
*,
loc=None,
ip=None,
) → CopyAtom
Finds a performant TMEM_LOAD copy op for the selected epilogue tile (epi_tile), element types, and tcgen05.mma instruction used.

Parameters
:
cta_tile_shape (cute.Shape) – A tuple or list representing the dimensions of the CTA tile.
layout_d (LayoutEnum) – The layout enum of the output tensor D.
elem_ty_d (Type[Numeric]) – The element type for output tensor D.
elem_ty_acc (Type[Numeric]) – The element type for accumulation.
epi_tile (cute.Tile) – The epilogue tile configuration.
use_2cta_instrs (bool) – A flag indicating whether the configuration is for 2 SMs.
Returns
:
An instance of Sm100TmemLoad with the computed configuration.
Return type
:
cute.CopyAtom
Raises
:
ValueError – If the function cannot handle the given combination of accumulation and dimension types, or if it cannot determine the appropriate configuration based on the input parameters.
cutlass.utils.sm100.get_num_tmem_alloc_cols(
tmem_tensors: cutlass.cute.typing.Tensor | List[cutlass.cute.typing.Tensor],
rounding=True,
) → int
Get the total number of TMEM allocation columns for the given TMEM tensors.

Parameters
:
tmem_tensors (Union[cute.Tensor, List[cute.Tensor]]) – The TMEM tensors to get the number of allocation columns for.
rounding (bool) – Whether to round up the number of allocation columns to the nearest power of 2.
Returns
:
The total number of TMEM allocation columns.
Return type
:
int
Raises
:
ValueError – If the number of TMEM allocation columns exceeds the maximum capacity of 512 or is less than 32.
cutlass.utils.sm100.make_smem_layout_a(
tiled_mma: TiledMma,
mma_tiler_mnk: cutlass.cute.typing.Tile,
a_dtype: Type[cutlass.cutlass_dsl.Numeric],
num_stages: int,
*,
loc=None,
ip=None,
) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout
This function helps with:

Get the partitioned shape of the A tensor based on the tiled_mma & MMA tiler.
Select the heuristic SMEM layout atom based on the A tensor’s majorness, the data type, and the major mode size.
cute.Tile the SMEM layout atom to the MMA tile shape.
Stage the SMEM layout based on the number of stages.
Parameters
:
tiled_mma (cute.TiledMma) – The tiled MMA used to partition tensor A
mma_tiler_mnk (cute.cute.Tile) – The MMA tile shape
a_dtype (Type[Numeric]) – The element type for tensor A
num_stages (int) – The number of pipeline stages for tensor A
Returns
:
SMEM layout for tensor A
Return type
:
Union[cute.Layout, cute.ComposedLayout]
cutlass.utils.sm100.make_smem_layout_b(
tiled_mma: TiledMma,
mma_tiler_mnk: cutlass.cute.typing.Tile,
b_dtype: Type[cutlass.cutlass_dsl.Numeric],
num_stages: int,
*,
loc=None,
ip=None,
) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout
This function helps:

Get the partitioned shape of the B tensor based on the tiled_mma & MMA tiler.
Select the heuristic SMEM layout atom based on the B tensor’s majorness, the data type, and the major mode size.
cute.Tile the SMEM layout atom to the MMA tile shape.
Stage the SMEM layout based on the number of stages.
Parameters
:
tiled_mma (cute.TiledMma) – The tiled MMA which is used to partition the B tensor.
mma_tiler_mnk (cute.cute.Tile) – The MMA tile shape.
b_dtype (Type[Numeric]) – The element type for the B tensor.
num_stages (int) – The stage of the B tensor.
Returns
:
SMEM layout for the B tensor.
Return type
:
Union[cute.Layout, cute.ComposedLayout]
cutlass.utils.sm100.make_smem_layout_epi(
epi_dtype: Type[cutlass.cutlass_dsl.Numeric],
epi_layout: LayoutEnum,
epi_tile: cutlass.cute.typing.Tile,
epi_stage: int,
*,
loc=None,
ip=None,
) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout
This function helps:

Select the heuristic SMEM layout atom based on the epilog tile shape, the epilog tensor’s majorness, and the element type.
cute.Tile the SMEM layout atom to the epilog tile shape.
Stage the SMEM layout based on the number of stages.
Parameters
:
epi_dtype (Type[Numeric]) – The element type for the epilog tensor.
epi_layout (LayoutEnum) – The layout enum for the epilog tensor.
epi_tile (cute.cute.Tile) – The epilogue tile shape.
epi_stage (int) – The stage of the epilog tensor.
Returns
:
SMEM layout for epilog tensors (usually C & D which are processed in the epilog)
Return type
:
Union[cute.Layout, cute.ComposedLayout]
cutlass.utils.sm100.make_trivial_tiled_mma(
ab_dtype: Type[cutlass.cutlass_dsl.Numeric],
a_leading_mode: OperandMajorMode,
b_leading_mode: OperandMajorMode,
acc_dtype: Type[cutlass.cutlass_dsl.Numeric],
cta_group: CtaGroup,
mma_tiler_mn: Tuple[int, int],
a_source: OperandSource = cutlass._mlir.dialects.cute.MmaFragKind.smem_desc,
*,
loc=None,
ip=None,
) → TiledMma
Make a tiled MMA atom with given data type, leading dimension, cta group and mma tile shape. By default, the MMA atom is created with SMEM operand source for A.

Parameters
:
ab_dtype (type[Numeric]) – Data type of operands A and B.
a_leading_mode (tcgen05.OperandMajorMode) – Leading dimension of operand A (1 for K, 0 for M/N).
b_leading_mode (tcgen05.OperandMajorMode) – Leading dimension of operand B (1 for K, 0 for M/N).
acc_dtype (type[Numeric]) – Data type of the accumulator.
cta_group (tcgen05.CtaGroup) – The CTA group to use.
mma_tiler_mn (Tuple[int, int]) – The shape (M, N, K) of the MMA tiler.
a_source (cutlass.cute.nvgpu.tcgen05.OperandSource) – The source of operand A (SMEM by default or TMEM).
Returns
:
A tiled MMA atom.
Return type
:
cute.TiledMma
Raises
:
TypeError – If the data type is not supported.
cutlass.utils.sm100.make_blockscaled_trivial_tiled_mma(
ab_dtype: Type[cutlass.cutlass_dsl.Numeric],
a_leading_mode: OperandMajorMode,
b_leading_mode: OperandMajorMode,
sf_dtype: Type[cutlass.cutlass_dsl.Numeric],
sf_vec_size: int,
cta_group: CtaGroup,
mma_tiler_mn: Tuple[int, int],
a_source: OperandSource = cutlass._mlir.dialects.cute.MmaFragKind.smem_desc,
*,
loc=None,
ip=None,
) → TiledMma
Make a BlockScaled tiled MMA atom with given data type, leading dimension, cta group and mma tile shape. By default, the MMA atom is created with SMEM operand source for A.

Parameters
:
ab_dtype (type[Numeric]) – Data type of operands A and B.
a_leading_mode (tcgen05.OperandMajorMode) – Leading dimension of operand A (1 for K, 0 for M/N).
b_leading_mode (tcgen05.OperandMajorMode) – Leading dimension of operand B (1 for K, 0 for M/N).
sf_dtype (type[Numeric]) – Data type of the Scale Factor.
sf_vec_size (int) – The vector size of the Scale Factor.
cta_group (tcgen05.CtaGroup) – The CTA group to use.
mma_tiler_mn (Tuple[int, int]) – The shape (M, N, K) of the MMA tiler.
a_source (cutlass.cute.nvgpu.tcgen05.OperandSource) – The source of operand A (SMEM by default or TMEM).
Returns
:
A tiled MMA atom.
Return type
:
cute.TiledMma
Raises
:
TypeError – If the data type is not supported.
cutlass.utils.sm100.cluster_shape_to_tma_atom_A(
cluster_shape_mnk: cutlass.cute.typing.Shape,
atom_thr_id: cutlass.cute.typing.Layout,
*,
loc=None,
ip=None,
) → CopyBulkTensorTileG2SMulticastOp | CopyBulkTensorTileG2SOp
Select the appropriate TMA copy atom for A based on the number of SMs and the multicast flag.

Parameters
:
cluster_shape_mnk (cute.Shape) – The shape of the cluster
atom_thr_id (cute.Layout) – The thread ID of the atom
Returns
:
The appropriate TMA copy atom kind
Return type
:
cpasync.CopyBulkTensorTileG2SMulticastOp or cpasync.CopyBulkTensorTileG2SOp
Raises
:
ValueError – If the atom_sm_cnt is invalid
ValueError – If the cluster shape is not divisible by the atom SM count
cutlass.utils.sm100.cluster_shape_to_tma_atom_B(
cluster_shape_mnk: cutlass.cute.typing.Shape,
atom_thr_id: cutlass.cute.typing.Layout,
*,
loc=None,
ip=None,
) → CopyBulkTensorTileG2SMulticastOp | CopyBulkTensorTileG2SOp
Select the appropriate TMA copy atom for Bbased on the number of SMs and the multicast flag.

Parameters
:
cluster_shape_mnk (cute.Shape) – The shape of the cluster
atom_thr_id (cute.Layout) – The thread ID of the atom
Returns
:
The appropriate TMA copy atom kind
Return type
:
cpasync.CopyBulkTensorTileG2SMulticastOp or cpasync.CopyBulkTensorTileG2SOp
Raises
:
ValueError – If the atom_sm_cnt is invalid
ValueError – If the cluster shape is not divisible by the atom SM count
cutlass.utils.sm100.cluster_shape_to_tma_atom_SFB(
cluster_shape_mnk: cutlass.cute.typing.Shape,
atom_thr_id: cutlass.cute.typing.Layout,
*,
loc=None,
ip=None,
) → CopyBulkTensorTileG2SMulticastOp | CopyBulkTensorTileG2SOp
Select the appropriate TMA copy atom for SFB based on the number of SMs and the multicast flag.

Parameters
:
cluster_shape_mnk (cute.Shape) – The shape of the cluster
atom_thr_id (cute.Layout) – The thread ID of the atom
Returns
:
The appropriate TMA copy atom kind
Return type
:
cpasync.CopyBulkTensorTileG2SMulticastOp or cpasync.CopyBulkTensorTileG2SOp
Raises
:
ValueError – If the atom_sm_cnt is invalid
ValueError – If the cluster shape is not divisible by the atom SM count


class cutlass.pipeline.PipelineTmaUmma(
sync_object_full: SyncObject,
sync_object_empty: SyncObject,
num_stages: int,
producer_mask: cutlass.cutlass_dsl.Int32 | None,
consumer_mask: cutlass.cutlass_dsl.Int32 | None,
is_leader_cta: bool,
cta_group: CtaGroup,
)
Bases: PipelineAsync

PipelineTmaUmma is used for TMA producers and UMMA consumers (e.g. Blackwell mainloops).

is_leader_cta: bool
cta_group: CtaGroup
static _compute_mcast_arrival_mask(
cta_layout_vmnk: cutlass.cute.typing.Layout,
mcast_mode_mn: tuple[int, int],
)
Computes a mask for signaling arrivals to multicasting threadblocks.
static _compute_is_leader_cta(
cta_layout_vmnk: cutlass.cute.typing.Layout,
)
Computes leader threadblocks for 2CTA kernels. For 1CTA, all threadblocks are leaders.
static create(
*,
num_stages: int,
producer_group: CooperativeGroup,
consumer_group: CooperativeGroup,
tx_count: int,
barrier_storage: cutlass.cute.typing.Pointer | None = None,
cta_layout_vmnk: cutlass.cute.typing.Layout | None = None,
mcast_mode_mn: tuple[int, int] = (1, 1),
)
This helper function computes any necessary attributes and returns an instance of PipelineTmaUmma. :param barrier_storage: Pointer to the smem address for this pipeline’s mbarriers :type barrier_storage: cute.Pointer :param num_stages: Number of buffer stages for this pipeline :type num_stages: Int32 :param producer_group: CooperativeGroup for the producer agent :type producer_group: CooperativeGroup :param consumer_group: CooperativeGroup for the consumer agent :type consumer_group: CooperativeGroup :param tx_count: Number of bytes expected to be written to the transaction barrier for one stage :type tx_count: int :param cta_layout_vmnk: Layout of the cluster shape :type cta_layout_vmnk: cute.Layout | None :param mcast_mode_mn: Tuple of two integers, specifying whether mcast is enabled for the m and n modes. At least one of the two integers must be 1. :type mcast_mode_mn: tuple[int, int]
consumer_release(
state: PipelineState,
)
UMMA consumer release buffer empty, cta_group needs to be provided.
producer_acquire(
state: PipelineState,
try_acquire_token: cutlass.cutlass_dsl.Boolean | None = None,
)
TMA producer commit conditionally waits on buffer empty and sets the transaction barrier for leader threadblocks.
producer_commit(
state: PipelineState,
)
TMA producer commit is a noop since TMA instruction itself updates the transaction count.
__init__(
sync_object_full: SyncObject,
sync_object_empty: SyncObject,
num_stages: int,
producer_mask: cutlass.cutlass_dsl.Int32 | None,
consumer_mask: cutlass.cutlass_dsl.Int32 | None,
is_leader_cta: bool,
cta_group: CtaGroup,
) → None

class cutlass.pipeline.PipelineTmaStore(
sync_object_full: SyncObject,
sync_object_empty: SyncObject,
num_stages: int,
producer_mask: cutlass.cutlass_dsl.Int32 | None,
consumer_mask: cutlass.cutlass_dsl.Int32 | None,
)
Bases: PipelineAsync

PipelineTmaStore is used for synchronizing TMA stores in the epilogue. It does not use mbarriers.

static create(
*,
num_stages: int,
producer_group: CooperativeGroup,
)
This helper function computes any necessary attributes and returns an instance of PipelineTmaStore. :param num_stages: Number of buffer stages for this pipeline :type num_stages: Int32 :param producer_group: CooperativeGroup for the producer agent :type producer_group: CooperativeGroup
producer_acquire()
producer_commit()
consumer_wait()
consumer_release()
producer_tail()
Make sure the last used buffer empty signal is visible to producer. Producer tail is usually executed by producer before exit, to avoid dangling mbarrier arrive signals after kernel exit.

Parameters
:
state (PipelineState) – The pipeline state that points to next useful buffer
__init__(
sync_object_full: SyncObject,
sync_object_empty: SyncObject,
num_stages: int,
producer_mask: cutlass.cutlass_dsl.Int32 | None,
consumer_mask: cutlass.cutlass_dsl.Int32 | None,
) → None

class cutlass.pipeline.PipelineProducer(
pipeline,
state,
group: CooperativeGroup,
)
Bases: object

A class representing a producer in an asynchronous pipeline.

This class manages the producer side of an asynchronous pipeline, handling synchronization and state management for producing data. It provides methods for acquiring, committing, and advancing through pipeline stages.

Variables
:
__pipeline – The asynchronous pipeline this producer belongs to
__state – The current state of the producer in the pipeline
__group – The cooperative group this producer operates in
Examples:

pipeline = PipelineAsync.create(...)
producer, consumer = pipeline.make_participants()
for i in range(iterations):
    # Try to acquire the current buffer without blocking
    try_acquire_token = producer.try_acquire()

    # Do something else independently
    ...

    # Wait for current buffer to be empty & Move index to next stage
    # If try_acquire_token is True, return immediately
    # If try_acquire_token is False, block until buffer is empty
    handle = producer.acquire_and_advance(try_acquire_token)

    # Produce data
    handle.commit()

class ImmutableResourceHandle(
_ImmutableResourceHandle__origin: cutlass.pipeline.sm90.PipelineAsync,
_ImmutableResourceHandle__immutable_state: cutlass.pipeline.helpers.PipelineState,
)
Bases: ImmutableResourceHandle

property barrier
Get the barrier pointer for the current pipeline stage.

Returns
:
Pointer to the barrier for the current stage
Return type
:
cute.Pointer
commit()
Signal that data production is complete for the current stage.

This allows consumers to start processing the data.
__init__(
_ImmutableResourceHandle__origin: PipelineAsync,
_ImmutableResourceHandle__immutable_state: PipelineState,
) → None
__init__(
pipeline,
state,
group: CooperativeGroup,
)
Initialize a new Producer instance.

Parameters
:
pipeline (PipelineAsync) – The pipeline this producer belongs to
state (PipelineState) – Initial pipeline state
group (CooperativeGroup) – The cooperative group for synchronization
__pipeline: PipelineAsync
__state: PipelineState
__group: CooperativeGroup
reset()
Reset the count of how many handles this producer has committed.
acquire(
try_acquire_token: cutlass.cutlass_dsl.Boolean | None = None,
) → ImmutableResourceHandle
Wait for the current buffer to be empty before producing data. This is a blocking operation.

Parameters
:
try_acquire_token (Optional[Boolean]) – Optional token to try to acquire the buffer
Returns
:
A handle to the producer for committing the data
Return type
:
ImmutableResourceHandle
advance()
Move to the next pipeline stage.
acquire_and_advance(
try_acquire_token: cutlass.cutlass_dsl.Boolean | None = None,
) → ImmutableResourceHandle
Acquire the current buffer and advance to the next pipeline stage.

This method combines the acquire() and advance() operations into a single call. It first waits for the current buffer to be empty before producing data, then advances the pipeline to the next stage.

Parameters
:
try_acquire_token (Optional[Boolean]) – Token indicating whether to try non-blocking acquire. If True, returns immediately without waiting. If False or None, blocks until buffer is empty.
Returns
:
A handle to the producer that can be used to commit data to the acquired buffer stage
Return type
:
ImmutableResourceHandle
try_acquire() → cutlass.cutlass_dsl.Boolean
Attempt to acquire the current buffer without blocking.

This method tries to acquire the current buffer stage for producing data without waiting. It can be used to check buffer availability before committing to a blocking acquire operation.

Returns
:
A boolean token indicating whether the buffer was successfully acquired
Return type
:
Boolean
commit(
handle: ImmutableResourceHandle | None = None,
)
Signal that data production is complete for the current stage.

This allows consumers to start processing the data.

Parameters
:
handle (Optional[ImmutableResourceHandle]) – Optional handle to commit, defaults to None
Raises
:
AssertionError – If provided handle does not belong to this producer
tail()
Ensure all used buffers are properly synchronized before producer exit.

This should be called before the producer finishes to avoid dangling signals.
class cutlass.pipeline.PipelineConsumer(
pipeline,
state: PipelineState,
group: CooperativeGroup,
)
Bases: object

A class representing a consumer in an asynchronous pipeline.

The Consumer class manages the consumer side of an asynchronous pipeline, handling synchronization and state management for consuming data. It provides methods for waiting, releasing, and advancing through pipeline stages.

Variables
:
__pipeline – The asynchronous pipeline this consumer belongs to
__state – The current state of the consumer in the pipeline
__group – The cooperative group this consumer operates in
Examples:

pipeline = PipelineAsync.create(...)
producer, consumer = pipeline.make_participants()
for i in range(iterations):
    # Try to wait for buffer to be full
    try_wait_token = consumer.try_wait()

    # Do something else independently
    ...

    # Wait for buffer to be full & Move index to next stage
    # If try_wait_token is True, return immediately
    # If try_wait_token is False, block until buffer is full
    handle = consumer.wait_and_advance(try_wait_token)

    # Consume data
    handle.release(  )  # Signal buffer is empty

    # Alternative way to do this is:
    # handle.release()  # Signal buffer is empty

class ImmutableResourceHandle(
_ImmutableResourceHandle__origin: cutlass.pipeline.sm90.PipelineAsync,
_ImmutableResourceHandle__immutable_state: cutlass.pipeline.helpers.PipelineState,
)
Bases: ImmutableResourceHandle

release()
Signal that data production is complete for the current stage. This allows consumers to start processing the data.
__init__(
_ImmutableResourceHandle__origin: PipelineAsync,
_ImmutableResourceHandle__immutable_state: PipelineState,
) → None
__init__(
pipeline,
state: PipelineState,
group: CooperativeGroup,
)
Initialize a new Consumer instance.

Parameters
:
pipeline (PipelineAsync) – The pipeline this consumer belongs to
state (PipelineState) – Initial pipeline state
group (CooperativeGroup) – The cooperative group for synchronization
__pipeline: PipelineAsync
__group: CooperativeGroup
__state: PipelineState
reset()
Reset the count of how many handles this consumer has consumed.
wait(
try_wait_token: cutlass.cutlass_dsl.Boolean | None = None,
) → ImmutableResourceHandle
Wait for data to be ready in the current buffer. This is a blocking operation that will not return until data is available.

Parameters
:
try_wait_token (Optional[Boolean]) – Token used to attempt a non-blocking wait for the buffer. If provided and True, returns immediately if buffer is not ready.
Returns
:
An immutable handle to the consumer that can be used to release the buffer once data consumption is complete
Return type
:
ImmutableResourceHandle
advance()
Advance the consumer to the next pipeline stage.

This updates the internal state to point to the next buffer in the pipeline. Should be called after consuming data from the current buffer.
wait_and_advance(
try_wait_token: cutlass.cutlass_dsl.Boolean | None = None,
) → ImmutableResourceHandle
Atomically wait for data and advance to next pipeline stage.

This is a convenience method that combines wait() and advance() into a single atomic operation. It will block until data is available in the current buffer, then automatically advance to the next stage.

Parameters
:
try_wait_token (Optional[Boolean]) – Token used to attempt a non-blocking wait for the buffer. If provided and True, returns immediately if buffer is not ready.
Returns
:
An immutable handle to the consumer that can be used to release the buffer once data consumption is complete
Return type
:
ImmutableResourceHandle
try_wait() → cutlass.cutlass_dsl.Boolean
Non-blocking check if data is ready in the current buffer.

This method provides a way to test if data is available without blocking. Unlike wait(), this will return immediately regardless of buffer state.

Returns
:
True if data is ready to be consumed, False if the buffer is not yet ready
Return type
:
Boolean
release(
handle: ImmutableResourceHandle | None = None,
)
Signal that data consumption is complete for the current stage. This allows producers to start producing new data.
cutlass.pipeline.make_pipeline_state(
type: PipelineUserType,
stages: int,
)
Creates a pipeline state. Producers are assumed to start with an empty buffer and have a flipped phase bit of 1.
cutlass.pipeline.pipeline_init_wait(
cta_layout_vmnk: cutlass.cute.typing.Layout | None = None,
)
Fences the mbarrier init and syncs the threadblock or cluster
cutlass.pipeline.arrive(barrier_id: int, num_threads: int)
The aligned flavor of arrive is used when all threads in the CTA will execute the same instruction. See PTX documentation.
cutlass.pipeline.arrive_unaligned(barrier_id: int, num_threads: int)
The unaligned flavor of arrive can be used with an arbitrary number of threads in the CTA.
cutlass.pipeline.wait(barrier_id: int, num_threads: int)
NamedBarriers do not have a standalone wait like mbarriers, only an arrive_and_wait. If synchronizing two warps in a producer/consumer pairing, the arrive count would be 32 using mbarriers but 64 using NamedBarriers. Only threads from either the producer or consumer are counted for mbarriers, while all threads participating in the sync are counted for NamedBarriers.
cutlass.pipeline.wait_unaligned(barrier_id: int, num_threads: int)
cutlass.pipeline.arrive_and_wait(barrier_id: int, num_threads: int)
cutlass.pipeline.sync(barrier_id: int = 0)


class cutlass.pipeline.NamedBarrier(barrier_id: int, num_threads: int)
Bases: SyncObject

NamedBarrier is an abstraction for named barriers managed by hardware. There are 16 named barriers available, with barrier_ids 0-15.

See the PTX documentation.

barrier_id: int
num_threads: int
arrive() → None
The aligned flavor of arrive is used when all threads in the CTA will execute the same instruction. See PTX documentation.
arrive_unaligned() → None
The unaligned flavor of arrive can be used with an arbitrary number of threads in the CTA.
wait() → None
NamedBarriers do not have a standalone wait like mbarriers, only an arrive_and_wait. If synchronizing two warps in a producer/consumer pairing, the arrive count would be 32 using mbarriers but 64 using NamedBarriers. Only threads from either the producer or consumer are counted for mbarriers, while all threads participating in the sync are counted for NamedBarriers.
wait_unaligned() → None
arrive_and_wait() → None
arrive_and_drop() → None
sync() → None
get_barrier() → int
max() → int
__init__(barrier_id: int, num_threads: int) → None
_abc_impl = <_abc._abc_data object>
class cutlass.pipeline.PipelineOrder(
sync_object_full: SyncObject,
depth: int,
length: int,
group_id: int,
state: PipelineState,
)
Bases: object

PipelineOrder is used for managing ordered pipeline execution with multiple groups.

This class implements a pipeline ordering mechanism where work is divided into groups and stages, allowing for controlled progression through pipeline stages with proper synchronization between different groups.

The pipeline ordering works as follows: - The pipeline is divided into ‘length’ number of groups - Each group has ‘depth’ number of stages - Groups execute in a specific order with synchronization barriers - Each group waits for the previous group to complete before proceeding

Example:

# Create pipeline order with 3 groups, each with 2 stages
pipeline_order = PipelineOrder.create(
    barrier_storage=smem_ptr,      # shared memory pointer for barriers
    depth=2,                       # 2 stages per group
    length=3,                      # 3 groups total
    group_id=0,                    # current group ID (0, 1, or 2)
    producer_group=producer_warp   # cooperative group for producers
)

# In the pipeline loop
for stage in range(num_stages):
    pipeline_order.wait()          # Wait for previous group to complete
    # Process current stage
    pipeline_order.arrive()        # Signal completion to next group

sync_object_full: SyncObject
depth: int
length: int
group_id: int
state: PipelineState
static create(
barrier_storage: cutlass.cute.typing.Pointer,
depth: int,
length: int,
group_id: int,
producer_group: CooperativeGroup,
)
get_barrier_for_current_stage_idx(group_id)
arrive()
wait()
__init__(
sync_object_full: SyncObject,
depth: int,
length: int,
group_id: int,
state: PipelineState,
) → None
class cutlass.pipeline.TmaStoreFence(num_stages: int = 0)
Bases: SyncObject

TmaStoreFence is used for a multi-stage epilogue buffer.

__init__(num_stages: int = 0) → None
arrive() → None
wait() → None#
arrive_and_wait() → None
arrive_and_drop() → None
get_barrier() → None
max() → None
tail() → None
_abc_impl = <_abc._abc_data object>


class cutlass.cute.nvgpu.tcgen05.MmaMXF4NVF4Op(
sf_dtype: Type[cutlass.cute.typing.Numeric],
instruction_shape: cutlass.cute.typing.Shape,
cta_group: CtaGroup,
a_src: OperandSource,
)
Bases: BlockScaledMmaOp

MXF4NVF4 tcgen05 BlockScaled MMA Operation.

See the PTX documentation. This Operation corresponds to the .kind::mxf4nvf4 qualifier.

descriptive_name = 'tcgen05 MXF4NVF4 BlockScaled MMA Operation'
__init__(
sf_dtype: Type[cutlass.cute.typing.Numeric],
instruction_shape: cutlass.cute.typing.Shape,
cta_group: CtaGroup,
a_src: OperandSource,
) → None
class cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind(value)
Bases: Enum

Enum class for the kinds of SMEM layout atoms for SM100.

Given a swizzle kind, an SMEM layout atom is the compact layout of smallest size that can be used to construct an SMEM layout using blocked product for operand A or B such that the resulting layout is legal for both TMA and UMMA.

Note that there are other ways of creating legal layouts for operand A and B.

MN_INTER = 1
MN_SW32 = 2
MN_SW64 = 3
MN_SW128 = 4
MN_SW128_32B = 5
K_INTER = 6
K_SW32 = 7
K_SW64 = 8
K_SW128 = 9
cutlass.cute.nvgpu.tcgen05.make_smem_layout_atom(
kind: SmemLayoutAtomKind,
element_type: Type[cutlass.cute.typing.Numeric],
*,
loc=None,
ip=None,
) → cutlass.cute.typing.ComposedLayout
Makes a SMEM layout Atom.

This function creates a composed layout in unit of elements consistent with the requested layout Atom kind and element data type.

Parameters
:
kind (SmemLayoutAtomKind) – The kind of layout Atom
element_type (Type[Numeric]) – The element data type to construct the layout for
Returns
:
The SMEM layout atom
Return type
:
ComposedLayout
cutlass.cute.nvgpu.tcgen05.tile_to_mma_shape(
atom,
mma_tile_shape: cutlass.cute.typing.Shape,
order: cutlass.cute.typing.IntTuple | None = None,
*,
loc=None,
ip=None,
)
Tiles a layout to an MMA shape.
cutlass.cute.nvgpu.tcgen05.commit(
mbar_ptr: cutlass.cute.typing.Pointer,
mask=None,
cta_group: ~cutlass.cute.nvgpu.tcgen05.mma.CtaGroup = <CtaGroup.ONE>,
*,
loc=None,
ip=None,
) → None
Perform an arrive operation on a mbarrier upon completion of previous MMA operations.

Parameters
:
mbar_ptr (Pointer) – A pointer to the mbarrier in SMEM
mask (Int) – An optional multicast mask for the CTAs in the cluster to signal arrival to
cutlass.cute.nvgpu.tcgen05.is_tmem_load(atom: CopyAtom) → bool
Returns whether a CopyAtom instance is a TMEM load.
cutlass.cute.nvgpu.tcgen05.is_tmem_store(atom: CopyAtom) → bool
Returns whether a CopyAtom instance is a TMEM store.
cutlass.cute.nvgpu.tcgen05.get_tmem_copy_properties(
atom: CopyAtom,
) → Tuple[int, int, int, Pack | Unpack]
Returns the properties of a TMEM copy atom (number of data paths, bits, repetitions, and whether packing/unpacking is used).
cutlass.cute.nvgpu.tcgen05.find_tmem_tensor_col_offset(
tmem_tensor: cutlass.cute.typing.Tensor,
*,
loc=None,
ip=None,
) → cutlass.cute.typing.Int
Computes the TMEM column offset given a TMEM tensor.

Parameters
:
tmem_tensor (Tensor) – The TMEM tensor to use to compute the columns offset
Returns
:
The columns offset
Return type
:
Int
cutlass.cute.nvgpu.tcgen05.make_tmem_copy(
atom: CopyAtom,
tmem_tensor: cutlass.cute.typing.Tensor,
*,
loc=None,
ip=None,
) → TiledCopy
Makes a Tiled Copy instance from a TMEM Copy Atom and a TMEM tensor.
cutlass.cute.nvgpu.tcgen05.make_s2t_copy(
atom: CopyAtom,
tmem_tensor: cutlass.cute.typing.Tensor,
*,
loc=None,
ip=None,
) → TiledCopy
Makes a Tiled Copy instance from a TMEM Copy Atom and a TMEM tensor.
cutlass.cute.nvgpu.tcgen05.get_s2t_smem_desc_tensor(
atom: CopyAtom,
smem_tensor: cutlass.cute.typing.Tensor,
*,
loc=None,
ip=None,
) → cutlass.cute.typing.Tensor
Returns the SMEM descriptor tensor from a S2T copy atom and a SMEM tensor.

class cutlass.cute.nvgpu.tcgen05.CtaGroup(value)
Bases: Enum

An enumeration for the cta_group qualifier of the MMA.

ONE = 1
TWO = 2


class cutlass.cute.nvgpu.tcgen05.OperandMajorMode(value)
Bases: Enum

An enumeration for the majorness of the input operands of the MMA.
class cutlass.cute.nvgpu.tcgen05.OperandSource(value)
Bases: Enum

An enumeration for the source memory location of the A input operand of the MMA.


arch
The cute.arch module provides lightweight wrappers for NVVM Operation builders which implement CUDA built-in device functions such as thread_idx. It integrates seamlessly with CuTe DSL types.

These wrappers enable source location tracking through the @dsl_user_op decorator. The module includes the following functionality:

Core CUDA built-in functions such as thread_idx, warp_idx, block_dim, grid_dim, cluster_dim, and related functions
Memory barrier management functions including mbarrier_init, mbarrier_arrive, mbarrier_wait, and associated operations
Low-level shared memory (SMEM) management capabilities, with SmemAllocator as the recommended interface
Low-level tensor memory (TMEM) management capabilities, with TmemAllocator as the recommended interface
API documentation

cutlass.cute.arch.make_warp_uniform(
value: cutlass.cute.typing.Int,
*,
loc=None,
ip=None,
) → cutlass.cute.typing.Int32
Provides a compiler hint indicating that the specified value is invariant across all threads in the warp, which may enable performance optimizations.

Parameters
:
value (Int) – The integer value to be marked as warp-uniform.
Returns
:
The input value, marked as warp-uniform.
Return type
:
Int32
cutlass.cute.arch.elect_one(*, loc=None, ip=None) → IfOpRegion
Elects one thread within a warp.

with elect_one():
    # Only one thread in the warp executes the code in this context
    pass

cutlass.cute.arch.mbarrier_init(
mbar_ptr: cutlass.cute.typing.Pointer,
cnt: cutlass.cute.typing.Int,
*,
loc=None,
ip=None,
) → None
Initializes a mbarrier with the specified thread arrival count.

Parameters
:
mbar_ptr (Pointer) – A pointer to the mbarrier in SMEM
cnt (Int) – The arrival count of the mbarrier
cutlass.cute.arch.mbarrier_init_fence(*, loc=None, ip=None) → None
A fence operation that applies to the mbarrier initializations.
cutlass.cute.arch.mbarrier_arrive_and_expect_tx(
mbar_ptr: cutlass.cute.typing.Pointer,
bytes: cutlass.cute.typing.Int,
peer_cta_rank_in_cluster=None,
*,
loc=None,
ip=None,
) → None
Arrives on a mbarrier and expects a specified number of transaction bytes.

Parameters
:
mbar_ptr (Pointer) – A pointer to the mbarrier in SMEM
bytes (Int) – The number of transaction bytes
peer_cta_rank_in_cluster – An optional CTA rank in cluster. If provided, the pointer to the mbarrier is converted to a remote address in the peer CTA’s SMEM.
cutlass.cute.arch.mbarrier_expect_tx(
mbar_ptr: cutlass.cute.typing.Pointer,
bytes: cutlass.cute.typing.Int,
peer_cta_rank_in_cluster=None,
*,
loc=None,
ip=None,
) → None
Expects a specified number of transaction bytes without an arrive.

Parameters
:
mbar_ptr (Pointer) – A pointer to the mbarrier in SMEM
bytes (Int) – The number of transaction bytes
peer_cta_rank_in_cluster – An optional CTA rank in cluster. If provided, the pointer to the mbarrier is converted to a remote address in the peer CTA’s SMEM.
cutlass.cute.arch.mbarrier_wait(
mbar_ptr: cutlass.cute.typing.Pointer,
phase: cutlass.cute.typing.Int,
*,
loc=None,
ip=None,
) → None
Waits on a mbarrier with a specified phase.

Parameters
:
mbar_ptr (Pointer) – A pointer to the mbarrier in SMEM
phase (Int) – The phase to wait for (either 0 or 1)
cutlass.cute.arch.mbarrier_try_wait(
mbar_ptr: cutlass.cute.typing.Pointer,
phase: cutlass.cute.typing.Int,
*,
loc=None,
ip=None,
) → cutlass.cute.typing.Boolean
Attempts to wait on a mbarrier with a specified phase in a non-blocking fashion.

Parameters
:
mbar_ptr (Pointer) – A pointer to the mbarrier in SMEM
phase (Int) – The phase to wait for (either 0 or 1)
Returns
:
A boolean value indicating whether the wait operation was successful
Return type
:
Boolean
cutlass.cute.arch.mbarrier_conditional_try_wait(
cond,
mbar_ptr: cutlass.cute.typing.Pointer,
phase: cutlass.cute.typing.Int,
*,
loc=None,
ip=None,
) → cutlass.cute.typing.Boolean
Conditionally attempts to wait on a mbarrier with a specified phase in a non-blocking fashion.

Parameters
:
cond – A boolean predicate
mbar_ptr (Pointer) – A pointer to the mbarrier in SMEM
phase (Int) – The phase to wait for (either 0 or 1)
Returns
:
A boolean value indicating whether the wait operation was successful
Return type
:
Boolean
cutlass.cute.arch.mbarrier_arrive(
mbar_ptr: cutlass.cute.typing.Pointer,
peer_cta_rank_in_cluster: cutlass.cute.typing.Int | None = None,
*,
loc=None,
ip=None,
) → None
Arrives on an mbarrier.

Parameters
:
mbar_ptr (Pointer) – A pointer to the mbarrier in SMEM
peer_cta_rank_in_cluster – An optional CTA rank in cluster. If provided, the pointer to the mbarrier is converted to a remote address in the peer CTA’s SMEM.
cutlass.cute.arch.lane_idx(*, loc=None, ip=None) → cutlass.cute.typing.Int32
Returns the lane index of the current thread within the warp.
cutlass.cute.arch.warp_idx(*, loc=None, ip=None) → cutlass.cute.typing.Int32
Returns the warp index within a CTA.
cutlass.cute.arch.thread_idx(
*,
loc=None,
ip=None,
) → Tuple[cutlass.cute.typing.Int32, cutlass.cute.typing.Int32, cutlass.cute.typing.Int32]
Returns the thread index within a CTA.
cutlass.cute.arch.block_dim(
*,
loc=None,
ip=None,
) → Tuple[cutlass.cute.typing.Int32, cutlass.cute.typing.Int32, cutlass.cute.typing.Int32]
Returns the number of threads in each dimension of the CTA.
cutlass.cute.arch.block_idx(
*,
loc=None,
ip=None,
) → Tuple[cutlass.cute.typing.Int32, cutlass.cute.typing.Int32, cutlass.cute.typing.Int32]
Returns the CTA identifier within a grid.
cutlass.cute.arch.grid_dim(
*,
loc=None,
ip=None,
) → Tuple[cutlass.cute.typing.Int32, cutlass.cute.typing.Int32, cutlass.cute.typing.Int32]
Returns the number of CTAs in each dimension of the grid.
cutlass.cute.arch.cluster_idx(
*,
loc=None,
ip=None,
) → Tuple[cutlass.cute.typing.Int32, cutlass.cute.typing.Int32, cutlass.cute.typing.Int32]
Returns the cluster identifier within a grid.
cutlass.cute.arch.cluster_dim(
*,
loc=None,
ip=None,
) → Tuple[cutlass.cute.typing.Int32, cutlass.cute.typing.Int32, cutlass.cute.typing.Int32]
Returns the number of clusters in each dimension of the grid.
cutlass.cute.arch.cluster_size(*, loc=None, ip=None) → cutlass.cute.typing.Int32
Returns the number of CTA within the cluster.
cutlass.cute.arch.block_in_cluster_idx(
*,
loc=None,
ip=None,
) → Tuple[cutlass.cute.typing.Int32, cutlass.cute.typing.Int32, cutlass.cute.typing.Int32]
Returns the CTA index within a cluster across all dimensions.
cutlass.cute.arch.block_in_cluster_dim(
*,
loc=None,
ip=None,
) → Tuple[cutlass.cute.typing.Int32, cutlass.cute.typing.Int32, cutlass.cute.typing.Int32]
Returns the dimensions of the cluster.
cutlass.cute.arch.block_idx_in_cluster(
*,
loc=None,
ip=None,
) → cutlass.cute.typing.Int32
Returns the linearized identifier of the CTA within the cluster.
cutlass.cute.arch.barrier(
*,
barrier_id=None,
number_of_threads=None,
loc=None,
ip=None,
) → None
Creates a barrier, optionally named.
cutlass.cute.arch.barrier_arrive(
*,
barrier_id=None,
number_of_threads=None,
loc=None,
ip=None,
) → None
cutlass.cute.arch.sync_threads(*, loc=None, ip=None) → None
Synchronizes all threads within a CTA.
cutlass.cute.arch.sync_warp(
mask: cutlass.cute.typing.Int = 4294967295,
*,
loc=None,
ip=None,
) → None
Performs a warp-wide sync with an optional mask.
cutlass.cute.arch.fence_acq_rel_cta(*, loc=None, ip=None) → None
Fence operation with acquire-release semantics.

See the PTX documentation.
cutlass.cute.arch.fence_acq_rel_cluster(*, loc=None, ip=None) → None
Fence operation with acquire-release semantics.

See the PTX documentation.
cutlass.cute.arch.fence_acq_rel_gpu(*, loc=None, ip=None) → None
Fence operation with acquire-release semantics.

See the PTX documentation.
cutlass.cute.arch.fence_acq_rel_sys(*, loc=None, ip=None) → None
Fence operation with acquire-release semantics.

See the PTX documentation.
cutlass.cute.arch.cp_async_commit_group(*, loc=None, ip=None) → None
Commits all prior initiated but uncommitted cp.async instructions.

See the PTX documentation.
cutlass.cute.arch.cp_async_wait_group(n, *, loc=None, ip=None) → None
Waits till only a specified numbers of cp.async groups are pending.

See the PTX documentation.
cutlass.cute.arch.cp_async_bulk_commit_group(*, loc=None, ip=None) → None
Commits all prior initiated but uncommitted cp.async.bulk instructions.

See the PTX documentation.
cutlass.cute.arch.cp_async_bulk_wait_group(
group,
*,
read=None,
loc=None,
ip=None,
) → None
Waits till only a specified numbers of cp.async.bulk groups are pending.

See the PTX documentation.
cutlass.cute.arch.cluster_wait(*, loc=None, ip=None) → None
A cluster-wide wait operation.
cutlass.cute.arch.cluster_arrive(*, aligned=None, loc=None, ip=None) → None
A cluster-wide arrive operation.
cutlass.cute.arch.cluster_arrive_relaxed(*, aligned=None, loc=None, ip=None) → None
A cluster-wide arrive operation with relaxed semantics.
cutlass.cute.arch.vote_ballot_sync(
pred: cutlass.cute.typing.Boolean,
mask: cutlass.cute.typing.Int = 4294967295,
*,
loc=None,
ip=None,
) → cutlass.cute.typing.Int32
Performs a ballot operation across the warp.

It copies the predicate from each thread in mask into the corresponding bit position of destination register d, where the bit position corresponds to the thread’s lane id.

Parameters
:
pred (Boolean) – The predicate value for the current thread
mask (Int, optional) – A 32-bit integer mask specifying which threads participate, defaults to all threads (0xFFFFFFFF)
Returns
:
A 32-bit integer where each bit represents a thread’s predicate value
Return type
:
Int32
See the PTX documentation.
cutlass.cute.arch.vote_any_sync(
pred: cutlass.cute.typing.Boolean,
mask: cutlass.cute.typing.Int = 4294967295,
*,
loc=None,
ip=None,
) → cutlass.cute.typing.Boolean
True if source predicate is True for any non-exited threads in mask. Negate the source predicate to compute .not_all.

Parameters
:
pred (Boolean) – The predicate value for the current thread
mask (Int, optional) – A 32-bit integer mask specifying which threads participate, defaults to all threads (0xFFFFFFFF)
Returns
:
A boolean value indicating if the source predicate is True for all non-exited threads in mask
Return type
:
Boolean
See the PTX documentation.
cutlass.cute.arch.vote_all_sync(
pred: cutlass.cute.typing.Boolean,
mask: cutlass.cute.typing.Int = 4294967295,
*,
loc=None,
ip=None,
) → cutlass.cute.typing.Boolean
True if source predicate is True for all non-exited threads in mask. Negate the source predicate to compute .none.

Parameters
:
pred (Boolean) – The predicate value for the current thread
mask (Int, optional) – A 32-bit integer mask specifying which threads participate, defaults to all threads (0xFFFFFFFF)
Returns
:
A boolean value indicating if the source predicate is True for all non-exited threads in mask
Return type
:
Boolean
See the PTX documentation.
cutlass.cute.arch.vote_uni_sync(
pred: cutlass.cute.typing.Boolean,
mask: cutlass.cute.typing.Int = 4294967295,
*,
loc=None,
ip=None,
) → cutlass.cute.typing.Boolean
True f source predicate has the same value in all non-exited threads in mask. Negating the source predicate also computes .uni

Parameters
:
pred (Boolean) – The predicate value for the current thread
mask (Int, optional) – A 32-bit integer mask specifying which threads participate, defaults to all threads (0xFFFFFFFF)
Returns
:
A boolean value indicating if the source predicate is True for all non-exited threads in mask
Return type
:
Boolean
cutlass.cute.arch.popc(
value: cutlass.cute.typing.Numeric,
*,
loc=None,
ip=None,
) → cutlass.cute.typing.Numeric
Performs a population count operation.
cutlass.cute.arch.fence_proxy(
kind: cutlass._mlir.dialects.nvvm.ProxyKind,
*,
space: cutlass._mlir.dialects.nvvm.SharedSpace | None = None,
use_intrinsic=None,
loc=None,
ip=None,
) → None
cutlass.cute.arch.fmax(
a: float | cutlass.cute.typing.Float32,
b: float | cutlass.cute.typing.Float32,
*,
loc=None,
ip=None,
) → cutlass.cute.typing.Float32
cutlass.cute.arch.rcp_approx(
a: float | cutlass.cute.typing.Float32,
*,
loc=None,
ip=None,
)
cutlass.cute.arch.exp2(
a: float | cutlass.cute.typing.Float32,
*,
loc=None,
ip=None,
) → cutlass.cute.typing.Float32
cutlass.cute.arch.alloc_smem(
element_type: Type[cutlass.cute.typing.Numeric],
size_in_elems: int,
alignment: int | None = None,
*,
loc=None,
ip=None,
) → cutlass.cute.typing.Pointer
Statically allocates SMEM.

Parameters
:
element_type (Type[Numeric]) – The pointee type of the pointer.
size_in_elems (int) – The size of the allocation in terms of number of elements of the pointee type
alignment (int) – An optional pointer alignment for the allocation
Returns
:
A pointer to the start of the allocation
Return type
:
Pointer
cutlass.cute.arch.get_dyn_smem(
element_type: Type[cutlass.cute.typing.Numeric],
alignment: int | None = None,
*,
loc=None,
ip=None,
) → cutlass.cute.typing.Pointer
Retrieves a pointer to a dynamic SMEM allocation.

Parameters
:
element_type (Type[Numeric]) – The pointee type of the pointer.
alignment (int) – An optional pointer alignment, the result pointer is offset appropriately
Returns
:
A pointer to the start of the dynamic SMEM allocation with a correct alignement
Return type
:
Pointer
cutlass.cute.arch.get_dyn_smem_size(*, loc=None, ip=None) → int
Gets the size in bytes of the dynamic shared memory that was specified at kernel launch time. This can be used for bounds checking during shared memory allocation.

Returns
:
The size of dynamic shared memory in bytes
Return type
:
int
cutlass.cute.arch.retrieve_tmem_ptr(
element_type: Type[cutlass.cute.typing.Numeric],
alignment: int,
ptr_to_buffer_holding_addr: cutlass.cute.typing.Pointer,
*,
loc=None,
ip=None,
) → cutlass.cute.typing.Pointer
Retrieves a pointer to TMEM with the provided element type and alignment.

Parameters
:
element_type (Type[Numeric]) – The pointee type of the pointer.
alignment (int) – The alignment of the result pointer
ptr_to_buffer_holding_addr (Pointer) – A pointer to a SMEM buffer holding the TMEM address of the start of the allocation allocation
Returns
:
A pointer to TMEM
Return type
:
Pointer
cutlass.cute.arch.alloc_tmem(
num_columns: cutlass.cute.typing.Int,
smem_ptr_to_write_address: cutlass.cute.typing.Pointer,
is_two_cta=None,
*,
loc=None,
ip=None,
) → None
Allocates TMEM.

Parameters
:
num_columns (Int) – The number of TMEM columns to allocate
smem_ptr_to_write_address (Pointer) – A pointer to a SMEM buffer where the TMEM address is written to
is_two_cta – Optional boolean parameter for 2-CTA MMAs
cutlass.cute.arch.relinquish_tmem_alloc_permit(
is_two_cta=None,
*,
loc=None,
ip=None,
) → None
Relinquishes the right to allocate TMEM so that other CTAs potentially in a different grid can allocate.
cutlass.cute.arch.dealloc_tmem(
tmem_ptr: cutlass.cute.typing.Pointer,
num_columns: cutlass.cute.typing.Int,
is_two_cta=None,
*,
loc=None,
ip=None,
) → None
Deallocates TMEM using the provided pointer and number of columns.

Parameters
:
tmem_ptr (Pointer) – A pointer to the TMEM allocation to de-allocate
num_columns (Int) – The number of columns in the TMEM allocation
is_two_cta – Optional boolean parameter for 2-CTA MMAs
cutlass.cute.arch.prmt(src, src_reg_shifted, prmt_indices, *, loc=None, ip=None)
cutlass.cute.arch.cvt_i8_bf16_intrinsic(vec_i8, length, *, loc=None, ip=None)
Convert a vector of int8 to a vector of bfloat16.

Parameters
:
vec_i8 (1D vector of int8) – The input vector of int8.
length (int) – The length of the input vector.
Returns
:
The output 1D vector of bfloat16 with the same length as the input vector.
Return type
:
1D vector of bfloat16
cutlass.cute.arch.cvt_i4_bf16_intrinsic(vec_i4, length, *, loc=None, ip=None)
Convert a vector of int4 to a vector of bfloat16.

Parameters
:
vec_i4 (1D vector of int4) – The input vector of int4.
length (int) – The length of the input vector.
Returns
:
The output 1D vector of bfloat16 with the same length as the input vector.
Return type
:
1D vector of bfloat16
cutlass.cute.arch.cvt_f4e2m1_f16_intrinsic(vec_f4e2m1, length, *, loc=None, ip=None)
Convert a vector of float4e2m1 to a vector of float16.

Parameters
:
vec_f4e2m1 (1D vector of float4e2m1) – The input vector of float4e2m1.
length (int) – The length of the input vector.
Returns
:
The output 1D vector of float16 with the same length as the input vector.
Return type
:
1D vector of float16
cutlass.cute.arch.cvt_i8x4_to_f32x4(src_vec4, *, loc=None, ip=None)
cutlass.cute.arch.cvt_i8x2_to_f32x2(src_vec2, *, loc=None, ip=None)
cutlass.cute.arch.cvt_i8_bf16(src_i8, *, loc=None, ip=None)
cutlass.cute.arch.cvt_f32x2_bf16x2(src_vec2, *, loc=None, ip=None)

cutlass.utils
The cutlass.utils module contains utilities for developing kernels with CuTe DSL.

cutlass.utils.get_smem_capacity_in_bytes(compute_capability: str) → int
Get the shared memory capacity in bytes for a given compute capability.

Returns the maximum shared memory capacity in bytes available for the specified GPU compute capability.

Parameters
:
compute_capability (str) – The compute capability string (e.g. “70”, “75”, “80”)
Returns
:
The shared memory capacity in bytes
Return type
:
int
Raises
:
ValueError – If the compute capability is not supported
class cutlass.utils.SmemAllocator
Bases: object

A helper class for managing shared memory allocation on GPU.

This class manages shared memory and provides APIs for allocation of raw bytes, numeric types, arrays, and tensors with specified layouts and alignments.

Note
The base pointer is aligned to 1024 bytes upon initialization.
There is no need to explicitly specify shared memory size in kernel launch.
Currently only supports static layouts. Dynamic layouts are not supported.
Examples:

smem = SmemAllocator()

# Allocate raw bytes
buf_ptr = smem.allocate(100)  # 100 bytes

# Allocate numeric type
int8_ptr = smem.allocate(Int8)  # 1 byte

# Define a struct
@cute.struct
class SharedStorage:
    alpha: cutlass.Float32
    x: cutlass.Int32

# Allocate struct
struct_ptr = smem.allocate(SharedStorage)  # 8 bytes

# use of struct members
struct_ptr.alpha = 1.0
struct_ptr.x = 2

# Allocate array
int8_array = smem.allocate_array(Int8, 10)  # 10 bytes

# Allocate tensor
layout = cute.make_layout((16, 16))
tensor = smem.allocate_tensor(Int8, layout)  # 256 bytes

static capacity_in_bytes(compute_capability: str) → int
Get the shared memory capacity in bytes for a given compute capability.

Returns the maximum shared memory capacity in bytes available for the specified GPU compute capability.

Parameters
:
compute_capability (str) – The compute capability string (e.g. “70”, “75”, “80”)
Returns
:
The shared memory capacity in bytes
Return type
:
int
Raises
:
ValueError – If the compute capability is not supported
__init__(*, loc=None, ip=None)
Initialize a new SmemAllocator instance.

Creates a new shared memory allocator with a base pointer aligned to 1024 bytes. Tracks the allocator instance for memory management.

Parameters
:
loc (Optional[ir.Location]) – Source location information for debugging, defaults to None
ip (Optional[ir.InsertionPoint]) – Insertion point for MLIR operations, defaults to None
allocate(
size_or_type: int,
byte_alignment: int,
*,
loc=None,
ip=None,
) → cutlass.cute.typing.Pointer
allocate(
size_or_type: Type[cutlass.cutlass_dsl.Numeric],
byte_alignment: int,
*,
loc=None,
ip=None,
) → cutlass.cute.typing.Pointer
allocate(
size_or_type: struct,
byte_alignment: int,
*,
loc=None,
ip=None,
) → cutlass.cute.typing.Pointer
Allocate a block of memory with specified size and alignment.

This method allocates a block of shared memory with the specified size and alignment requirements. It supports allocating raw bytes, numeric types(as scalar value), and struct types.

Parameters
:
size_or_type (Union[int, Type[Numeric], cute.struct]) – The allocation specification, which can be: - An integer specifying the number of bytes to allocate - A Numeric type (e.g., Int8, Float32) to allocate space for one element - A struct type to allocate space for the entire struct
byte_alignment (int, optional) – The minimum byte alignment requirement for the allocation, defaults to 1
loc (Optional[ir.Location]) – Source location information for debugging, defaults to None
ip (Optional[ir.InsertionPoint]) – Insertion point for MLIR operations, defaults to None
Returns
:
For raw bytes and numeric types, returns a pointer to the allocated memory. For struct types, returns an initialized struct instance at the allocated location.
Return type
:
cute.Pointer
Raises
:
ValueError – If size is negative or alignment is less than 1
TypeError – If size_or_type is not an integer, Numeric type, or struct
RuntimeError – If allocation would exceed available shared memory
allocate_array(
element_type: Type[cutlass.cutlass_dsl.Numeric],
num_elems: int = 1,
*,
loc=None,
ip=None,
)
Allocate an array of elements in shared memory.

Parameters
:
element_type (Type[Numeric]) – The type of elements to allocate
num_elems (int, optional) – Number of elements to allocate, defaults to 1
Returns
:
Pointer to the start of the allocated array
Return type
:
cute.Pointer
Raises
:
ValueError – If num_elems is less than 1
TypeError – If element_type is not a Numeric type
allocate_tensor(
element_type: Type[cutlass.cutlass_dsl.Numeric],
layout: int | cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout,
byte_alignment: int = 1,
swizzle: cutlass._mlir.ir.register_value_caster | None = None,
*,
loc=None,
ip=None,
)
Allocate a tensor in shared memory.

Note: Currently only supports static layouts. Dynamic layouts are not supported.

Parameters
:
element_type (Type[Numeric]) – The type of elements in the tensor
layout (Union[int, cute.Layout, cute.ComposedLayout]) – The layout specification for the tensor. Must be a static layout.
byte_alignment (int, optional) – The byte alignment requirement, defaults to 1
swizzle (cute.Swizzle, optional) – Swizzle for position-dependent swizzling, defaults to None
Returns
:
The allocated tensor with specified properties
Return type
:
cute.Tensor
Raises
:
TypeError – If element_type is not a Numeric type or if swizzle conflicts with layout
ValueError – If allocation is not byte-aligned
NotImplementedError – If dynamic layout is specified
class cutlass.utils.TmemAllocator(
alloc_result_dst_smem_ptr: cutlass.cute.typing.Pointer,
barrier_for_retrieve: NamedBarrier,
allocator_warp_id: int = 0,
is_two_cta: bool = False,
num_allocated_columns: int = 0,
two_cta_tmem_dealloc_mbar_ptr: cutlass.cute.typing.Pointer | None = None,
)
Bases: object

A class for managing tensor memory allocation on Blackwell GPU.

This class manages allocation/deallocation of tensor memory, including the mbarrier synchronization for two cta use case.

Variables
:
_alloc_result_dst_smem_ptr – The smem pointer that holds the base address of allocated tensor memory.
_barrier_for_retrieve – The barrier for retrieving tensor memory ptr.
_allocator_warp_id – The warp id of the allocator warp.
_is_two_cta – Whether the allocator is for two cta.
_num_allocated_columns – The number of columns allocated in the tensor memory.
_two_cta_tmem_dealloc_mbar_ptr – The mbarrier pointer required when deallocating tensor memory for two cta.
_init_dealloc_mbarrier()
__init__(
alloc_result_dst_smem_ptr: cutlass.cute.typing.Pointer,
barrier_for_retrieve: NamedBarrier,
allocator_warp_id: int = 0,
is_two_cta: bool = False,
num_allocated_columns: int = 0,
two_cta_tmem_dealloc_mbar_ptr: cutlass.cute.typing.Pointer | None = None,
)
Initialize the TmemAllocator instance.

Sets up the allocator state by initializing smem pointer that holds the base address of allocated tensor memory, allocator warp id, whether it is for two cta, number of allocated columns, and barrier for retrieving tensor memory ptr. Meanwhile, it also initializes the mbarrier pointer for two cta deallocation case.
check_valid_num_columns(num_columns: int)
Check if the number of columns is valid.

This method checks if the number of columns is valid. It checks if the number of columns is larger than 0, smaller than 512, a multiple of 32, and a power of two.
allocate(num_columns: int)
Allocate a block of tensor memory.

This method allocates a block of tensor memory from allocator warp and returns a handle to retrieve the allocated tensor memory address.
wait_for_alloc()
Wait for the allocator warp to finish allocation.

This method is used to synchronize the allocator warp with the other warps before retrieving tmem ptr.
retrieve_ptr(
dtype: Type[cutlass.cutlass_dsl.Numeric] = cutlass.cutlass_dsl.Float32,
) → cutlass.cute.typing.Pointer
Retrieve the pointer to the allocated tensor memory.

This method can be called by all warps after allocation has been performed by the allocator warp.
relinquish_alloc_permit()
Relinquish the tensor memory allocation permit.

This method relinquishes the tensor memory allocation permit for the allocator warp, promising the allocator warp will not allocate any more tensor memory.
free(
tmem_ptr: cutlass.cute.typing.Pointer,
num_columns: int = 0,
)
Deallocate the tensor memory.

This method sync on mbarrier (for two cta use case) and deallocates the tensor memory from the allocator warp. User can optionally specify the number of columns to deallocate. If not specified, all allocated columns will be deallocated.
class cutlass.utils.LayoutEnum(value)
Bases: Enum

An enumeration.

ROW_MAJOR = 'row_major'
COL_MAJOR = 'col_major'
mma_major_mode()
sm90_mma_major_mode()
is_k_major_a()
is_m_major_a()
is_n_major_b()
is_k_major_b()
is_n_major_c()
is_m_major_c()
static from_tensor(
tensor: cutlass.cute.typing.Tensor,
) → LayoutEnum
class cutlass.utils.WorkTileInfo(
tile_idx: cutlass.cute.typing.Coord,
is_valid_tile: cutlass.cutlass_dsl.Boolean,
)
Bases: object

A class to represent information about a work tile.

Variables
:
tile_idx – The index of the tile.
is_valid_tile – Whether the tile is valid.
__init__(
tile_idx: cutlass.cute.typing.Coord,
is_valid_tile: cutlass.cutlass_dsl.Boolean,
)
property is_valid_tile: cutlass.cutlass_dsl.Boolean
Check latest tile returned by the scheduler is valid or not. Any scheduling requests after all tasks completed will return an invalid tile.

Returns
:
The validity of the tile.
Return type
:
Boolean
property tile_idx: cutlass.cute.typing.Coord
Get the index of the tile.

Returns
:
The index of the tile.
Return type
:
cute.Coord
class cutlass.utils.PersistentTileSchedulerParams(
problem_shape_ntile_mnl: cutlass.cute.typing.Shape,
cluster_shape_mnk: cutlass.cute.typing.Shape,
swizzle_size: int = 1,
raster_along_m: bool = True,
*,
loc=None,
ip=None,
)
Bases: object

A class to represent parameters for a persistent tile scheduler.

This class is designed to manage and compute the layout of clusters and tiles in a batched gemm problem.

Variables
:
cluster_shape_mn – Shape of the cluster in (m, n) dimensions (K dimension cta count must be 1).
problem_layout_ncluster_mnl – Layout of the problem in terms of number of clusters in (m, n, l) dimensions.
__init__(
problem_shape_ntile_mnl: cutlass.cute.typing.Shape,
cluster_shape_mnk: cutlass.cute.typing.Shape,
swizzle_size: int = 1,
raster_along_m: bool = True,
*,
loc=None,
ip=None,
)
Initializes the PersistentTileSchedulerParams with the given parameters.

Parameters
:
problem_shape_ntile_mnl (cute.Shape) – The shape of the problem in terms of number of CTA (Cooperative Thread Array) in (m, n, l) dimensions.
cluster_shape_mnk (cute.Shape) – The shape of the cluster in (m, n) dimensions.
swizzle_size (int) – Swizzling size in the unit of cluster. 1 means no swizzle
raster_along_m (bool) – Rasterization order of clusters. Only used when swizzle_size > 1. True means along M, false means along N.
Raises
:
ValueError – If cluster_shape_k is not 1.
get_grid_shape(
max_active_clusters: cutlass.cutlass_dsl.Int32,
*,
loc=None,
ip=None,
) → Tuple[cutlass.cutlass_dsl.Integer, cutlass.cutlass_dsl.Integer, cutlass.cutlass_dsl.Integer]
Computes the grid shape based on the maximum active clusters allowed.

Parameters
:
max_active_clusters (Int32) – The maximum number of active clusters that can run in one wave.
Returns
:
A tuple containing the grid shape in (m, n, persistent_clusters). - m: self.cluster_shape_m. - n: self.cluster_shape_n. - persistent_clusters: Number of persistent clusters that can run.
class cutlass.utils.StaticPersistentTileScheduler(
params: PersistentTileSchedulerParams,
num_persistent_clusters: cutlass.cutlass_dsl.Int32,
current_work_linear_idx: cutlass.cutlass_dsl.Int32,
cta_id_in_cluster: cutlass.cute.typing.Coord,
num_tiles_executed: cutlass.cutlass_dsl.Int32,
)
Bases: object

A scheduler for static persistent tile execution in CUTLASS/CuTe kernels.

Variables
:
params – Tile schedule related params, including cluster shape and problem_layout_ncluster_mnl
num_persistent_clusters – Number of persistent clusters that can be launched
cta_id_in_cluster – ID of the CTA within its cluster
_num_tiles_executed – Counter for executed tiles
_current_work_linear_idx – Current cluster index
__init__(
params: PersistentTileSchedulerParams,
num_persistent_clusters: cutlass.cutlass_dsl.Int32,
current_work_linear_idx: cutlass.cutlass_dsl.Int32,
cta_id_in_cluster: cutlass.cute.typing.Coord,
num_tiles_executed: cutlass.cutlass_dsl.Int32,
)
Initializes the StaticPersistentTileScheduler with the given parameters.

Parameters
:
params (PersistentTileSchedulerParams) – Tile schedule related params, including cluster shape and problem_layout_ncluster_mnl.
num_persistent_clusters (Int32) – Number of persistent clusters that can be launched.
current_work_linear_idx (Int32) – Current cluster index.
cta_id_in_cluster (cute.Coord) – ID of the CTA within its cluster.
num_tiles_executed (Int32) – Counter for executed tiles.
static create(
params: PersistentTileSchedulerParams,
block_idx: Tuple[cutlass.cutlass_dsl.Integer, cutlass.cutlass_dsl.Integer, cutlass.cutlass_dsl.Integer],
grid_dim: Tuple[cutlass.cutlass_dsl.Integer, cutlass.cutlass_dsl.Integer, cutlass.cutlass_dsl.Integer],
*,
loc=None,
ip=None,
)
Initialize the static persistent tile scheduler.

Parameters
:
params (PersistentTileSchedulerParams) – Parameters for the persistent tile scheduler.
block_idx (Tuple[Integer, Integer, Integer]) – The 3d block index in the format (bidx, bidy, bidz).
grid_dim (Tuple[Integer, Integer, Integer]) – The 3d grid dimensions for kernel launch.
Returns
:
A StaticPersistentTileScheduler object.
Return type
:
StaticPersistentTileScheduler
static get_grid_shape(
params: PersistentTileSchedulerParams,
max_active_clusters: cutlass.cutlass_dsl.Int32,
*,
loc=None,
ip=None,
) → Tuple[cutlass.cutlass_dsl.Integer, cutlass.cutlass_dsl.Integer, cutlass.cutlass_dsl.Integer]
Calculates the grid shape to be launched on GPU using problem shape, threadblock shape, and active cluster size.

Parameters
:
params (PersistentTileSchedulerParams) – Parameters for grid shape calculation.
max_active_clusters (Int32) – Maximum active clusters allowed.
Returns
:
The calculated 3d grid shape.
Return type
:
Tuple[Integer, Integer, Integer]
_get_current_work_for_linear_idx(
current_work_linear_idx: cutlass.cutlass_dsl.Int32,
*,
loc=None,
ip=None,
) → WorkTileInfo
Compute current tile coord given current_work_linear_idx and cta_id_in_cluster.

Parameters
:
current_work_linear_idx (Int32) – The linear index of the current work.
Returns
:
An object containing information about the current tile coordinates and validity status.
Return type
:
WorkTileInfo
get_current_work(
*,
loc=None,
ip=None,
) → WorkTileInfo
initial_work_tile_info(
*,
loc=None,
ip=None,
) → WorkTileInfo
advance_to_next_work(
*,
advance_count: int = 1,
loc=None,
ip=None,
)
property num_tiles_executed: cutlass.cutlass_dsl.Int32
class cutlass.utils.TensorMapUpdateMode(value)
Bases: Enum

Enum class defining tensor map update modes.

Modes: GMEM: Update tensormap in global memory SMEM: Load tensormap from global memory to shared memory, update it in shared memory, then store back to global memory

GMEM = 1
SMEM = 2
class cutlass.utils.TensorMapManager(
tensormap_update_mode: TensorMapUpdateMode,
bytes_per_tensormap: int,
)
Bases: object

Manages TensorMap operations including initialization and updates. Provides utilities to convert tensormap pointer to across different memory spaces.

tensormap_update_mode: TensorMapUpdateMode
bytes_per_tensormap: int
get_tensormap_ptr(
ptr: cutlass.cute.typing.Pointer,
address_space=cutlass._mlir.dialects.cute.AddressSpace.gmem,
*,
loc=None,
ip=None,
) → cutlass.cute.typing.Pointer
init_tensormap_from_atom(
copy_atom: CopyAtom,
dst_ptr: cutlass.cute.typing.Pointer,
warp_id: int,
) → None
fence_tensormap_initialization() → None
fence_tensormap_update(
tensormap_ptr: cutlass.cute.typing.Pointer,
) → None
update_tensormap(
tensor_gmem: Tuple[cutlass.cute.typing.Tensor, ...],
tma_copy_atom: Tuple[CopyAtom, ...],
tensormap_gmem_ptr: Tuple[cutlass.cute.typing.Pointer, ...],
warp_id: int,
tensormap_smem_ptr: Tuple[cutlass.cute.typing.Pointer, ...],
) → None
__init__(
tensormap_update_mode: TensorMapUpdateMode,
bytes_per_tensormap: int,
) → None
class cutlass.utils.GroupSearchResult(
group_idx: cutlass.cutlass_dsl.Int32,
cta_tile_idx_m: cutlass.cutlass_dsl.Int32,
cta_tile_idx_n: cutlass.cutlass_dsl.Int32,
problem_shape_m: cutlass.cutlass_dsl.Int32,
problem_shape_n: cutlass.cutlass_dsl.Int32,
problem_shape_k: cutlass.cutlass_dsl.Int32,
cta_tile_count_k: cutlass.cutlass_dsl.Int32,
)
Bases: object

The result of the group search for grouped gemm.

Parameters
:
group_idx (Int32) – The result group index
cta_tile_idx_m (Int32) – CTA tile index along M dimension after rasterization
cta_tile_idx_n (Int32) – CTA tile index along N dimension after rasterization
problem_shape_m (Int32) – The M dimension of the gemm problem
problem_shape_n (Int32) – The N dimension of the gemm problem
problem_shape_k (Int32) – The K dimension of the gemm problem
cta_tile_count_k (Int32) – Number of tiles along K dimension
__init__(
group_idx: cutlass.cutlass_dsl.Int32,
cta_tile_idx_m: cutlass.cutlass_dsl.Int32,
cta_tile_idx_n: cutlass.cutlass_dsl.Int32,
problem_shape_m: cutlass.cutlass_dsl.Int32,
problem_shape_n: cutlass.cutlass_dsl.Int32,
problem_shape_k: cutlass.cutlass_dsl.Int32,
cta_tile_count_k: cutlass.cutlass_dsl.Int32,
) → None
class cutlass.utils.GroupedGemmGroupSearchState(
start_group_idx: cutlass.cutlass_dsl.Int32,
tile_count_prev_group: cutlass.cutlass_dsl.Int32,
tile_count_searched: cutlass.cutlass_dsl.Int32,
)
Bases: object

The state of group index search for grouped gemm.

The state will be initialized once and updated in every round of group index search.

Parameters
:
start_group_idx (Int32) – The group idx to start the search with
tile_count_prev_group (Int32) – Number of tiles before the matched group
tile_count_searched (Int32) – Number of tiles we have searched. When the matched group is found, it records the number of tiles including the matched group
__init__(
start_group_idx: cutlass.cutlass_dsl.Int32,
tile_count_prev_group: cutlass.cutlass_dsl.Int32,
tile_count_searched: cutlass.cutlass_dsl.Int32,
) → None
cutlass.utils.create_initial_search_state() → GroupedGemmGroupSearchState
Create an initial search state for grouped gemm.

Returns
:
A new search state with initial values
Return type
:
GroupedGemmGroupSearchState
class cutlass.utils.GroupedGemmTileSchedulerHelper(
group_count: int,
tile_sched_params: PersistentTileSchedulerParams,
cluster_tile_shape_mnk: tuple[int, int, int],
search_state: GroupedGemmGroupSearchState,
)
Bases: object

A helper to translate the raw block index (x, y, z) from tile scheduler to real CTA tile index for grouped gemm.

Parameters
:
group_count (int) – Number of groups in current grouped gemm problem
tile_sched_params (PersistentTileSchedulerParams) – Parameter used to create the tile scheduler this helper works with
cluster_tile_shape_mnk (tuple[int, int, int]) – The shape of cluster tile as (m, n, k)
search_state (GroupedGemmGroupSearchState) – The initial search state
__init__(
group_count: int,
tile_sched_params: PersistentTileSchedulerParams,
cluster_tile_shape_mnk: tuple[int, int, int],
search_state: GroupedGemmGroupSearchState,
) → None
delinearize_z(
cta_tile_coord: tuple,
problem_shape_mnkl: cutlass.cute.typing.Tensor,
) → GroupSearchResult
Delinearize the linear z index and return GroupSearchResult.

This function should be used by warps that need to know the CTA tile index on M and N dimensions.

Parameters
:
cta_tile_coord (tuple of Int32) – The raw CTA coordinate from tile scheduler
problem_shape_mnkl (cute.Tensor) – Tensor containing gemm problem size (M, N, K, L) for each group
Returns
:
The search result containing group index and tile coordinates
Return type
:
GroupSearchResult
search_cluster_tile_count_k(
cta_tile_coord: tuple,
problem_shape_mnkl: cutlass.cute.typing.Tensor,
) → Tuple[cutlass.cutlass_dsl.Int32, cutlass.cutlass_dsl.Int32]
Search the matched group for given linear index and compute the number of tiles along K dimension for the matched group.

This function should be used by warps that are only interested in the number of tiles along K dimension.

Parameters
:
cta_tile_coord (tuple of Int32) – The raw CTA coordinate from tile scheduler
problem_shape_mnkl (cute.Tensor) – Tensor containing gemm problem size (M, N, K, L) for all groups
Returns
:
A tuple containing cluster count along K dimension and the group index
Return type
:
Tuple[Int32, Int32]
_prefix_sum(
value_per_thread: cutlass.cutlass_dsl.Int32,
) → cutlass.cutlass_dsl.Int32
Perform prefix sum within a full warp.

Parameters
:
value_per_thread (Int32) – The value for this thread to contribute to the prefix sum
Returns
:
The prefix sum result for this thread
Return type
:
Int32
_get_problem_for_group(
problem_shape_mnkl: cutlass.cute.typing.Tensor,
group_idx: cutlass.cutlass_dsl.Int32,
) → cutlass.cute.typing.Tensor
Load gemm problem (m,n,k,l) for the specified group from global memory to register.

Parameters
:
problem_shape_mnkl (cute.Tensor) – Tensor in global memory with layout (group_count, 4):(4, 1)
group_idx (Int32) – The index of the group to load
Returns
:
The problem shape tensor for the specified group
Return type
:
cute.Tensor
_get_cluster_tile_count_mn(
problem_shape: cutlass.cute.typing.Tensor,
) → cutlass.cutlass_dsl.Int32
Compute total cluster count.

Parameters
:
problem_shape (cute.Tensor) – Tensor containing problem shape (m, n, k, l)
Returns
:
The total cluster tile count for M and N dimensions
Return type
:
Int32
_compute_cta_tile_coord(
cluster_tile_idx: cutlass.cutlass_dsl.Int32,
cta_tile_coord_in_cluster: tuple,
cluster_tile_count_m: cutlass.cutlass_dsl.Int32,
cluster_tile_count_n: cutlass.cutlass_dsl.Int32,
) → tuple
Compute CTA tile indices along M and N dimensions based on the linear index within a group.

It uses the AlongM mode to decompose the linear index onto M and N dimensions.

Parameters
:
cluster_tile_idx (Int32) – The linear index within a group
cta_tile_coord_in_cluster (tuple of Int32) – CTA indices along M and N dimensions within a cluster
cluster_tile_count_m (Int32) – The number of clusters along M dimension of the matched group
cluster_tile_count_n (Int32) – The number of clusters along N dimension of the matched group
Returns
:
A tuple containing CTA tile indices along M and N dimensions
Return type
:
tuple of (Int32, Int32)
_group_search(
linear_idx: cutlass.cutlass_dsl.Int32,
problem_shape_mnkl: cutlass.cute.typing.Tensor,
init_group_idx: cutlass.cutlass_dsl.Int32,
init_tile_count_searched: cutlass.cutlass_dsl.Int32,
) → GroupedGemmGroupSearchState
Search which group the linear index belongs to.

Parameters
:
linear_idx (Int32) – The linear index to be decomposed
problem_shape_mnkl (cute.Tensor) – Tensor containing gemm problem size (M, N, K, L) for all groups
init_group_idx (Int32) – The group idx to start the search with
init_tile_count_searched (Int32) – The number of tiles we have searched
Returns
:
The updated search state
Return type
:
GroupedGemmGroupSearchState


_group_search_and_load_problem_shape(
linear_idx: cutlass.cutlass_dsl.Int32,
problem_shape_mnkl: cutlass.cute.typing.Tensor,
start_group_idx: cutlass.cutlass_dsl.Int32,
tile_count_searched: cutlass.cutlass_dsl.Int32,
) → Tuple[cutlass.cutlass_dsl.Int32, cutlass.cute.typing.Tensor]
Perform group search and load problem shape for the matched group.

Parameters
:
linear_idx (Int32) – The linear index to be decomposed
problem_shape_mnkl (cute.Tensor) – Tensor containing gemm problem size (M, N, K, L) for all groups
start_group_idx (Int32) – The group idx to start the search with
tile_count_searched (Int32) – The number of tiles we have searched
Returns
:
A tuple containing the final group index and the problem shape tensor
Return type
:
Tuple[Int32, cute.Tensor]

class cutlass.utils.HardwareInfo(device_id: int = 0)
Bases: object

device_id: CUDA device ID to get the hardware info.

__init__(device_id: int = 0)
get_max_active_clusters(cluster_size: int) → int
get_l2_cache_size_in_bytes() → int
get_device_multiprocessor_count() → int
_checkCudaErrors(result) → None
_cudaGetErrorEnum(error) → str
_cuda_driver_version_ge(major: int, minor: int) → bool
_cuda_driver_version_lt(major: int, minor: int) → bool
_empty_kernel()
_host_function()
_get_device_function(device) → None
cutlass.utils.compute_epilogue_tile_shape(
cta_tile_shape: cutlass.cute.typing.Shape,
use_2cta_instrs: bool,
layout_d: LayoutEnum,
elem_ty_d: Type[cutlass.cutlass_dsl.Numeric],
*,
layout_c: LayoutEnum | None = None,
elem_ty_c: Type[cutlass.cutlass_dsl.Numeric] | None = None,
loc=None,
ip=None,
) → cutlass.cute.typing.Tile
Attempts to compute a reasonable epilogue tile based on block tile shape or allows the user to provide one.

Parameters
:
cta_tile_shape (cute.Shape) – A tuple or list representing the dimensions of the CTA tile, where cta_tile_shape[0] corresponds to the height (M) and cta_tile_shape[1] corresponds to the width (N) of the tile.
use_2cta_instrs (bool) – A flag indicating whether the configuration is for a 2SM setup.
layout_d (LayoutEnum) – The layout enum of the output tensor D.
elem_ty_d (Type[Numeric]) – The element type of output tensor D.
layout_c (LayoutEnum, optional) – The layout enum of the input tensor C. Defaults to None.
elem_ty_c (Union[Type[Numeric], None], optional) – The element type for input tensor C. Defaults to None.
Returns
:
Returns epilog tiler, which is used in subsequent epilog partitions.
Return type
:
cute.Tile
Raises
:
ValueError – If the computed tile cute.size does not meet minimum requirements based on CTA dimensions.
cutlass.utils.get_smem_store_op(
layout_d: LayoutEnum,
elem_ty_d: Type[cutlass.cutlass_dsl.Numeric],
elem_ty_acc: Type[cutlass.cutlass_dsl.Numeric],
tiled_tmem_load: TiledCopy,
*,
loc=None,
ip=None,
) → CopyAtom
Selects the largest vectorized smem store atom available subject to constraint of gmem layout and chosen TMEM_LOAD’s thread-value ownership.

Parameters
:
layout_d (LayoutEnum) – The layout enum of the output tensor D.
elem_ty_d (Type[Numeric]) – The element type for output tensor D.
elem_ty_acc (Type[Numeric]) – The element type for accumulator.
tiled_tmem_load (cute.TiledCopy) – An instance of TiledCopy that represents the tmem load operation.
Returns
:
Either SmemStoreMatrix or SimtSyncCopy, based on the input parameters.
Return type
:
cute.CopyAtom
cutlass.utils.get_tmem_load_op(
cta_tile_shape: cutlass.cute.typing.Shape,
layout_d: LayoutEnum,
elem_ty_d: Type[cutlass.cutlass_dsl.Numeric],
elem_ty_acc: Type[cutlass.cutlass_dsl.Numeric],
epi_tile: cutlass.cute.typing.Tile,
use_2cta_instrs: bool,
*,
loc=None,
ip=None,
) → CopyAtom
Finds a performant TMEM_LOAD copy op for the selected epilogue tile (epi_tile), element types, and tcgen05.mma instruction used.

Parameters
:
cta_tile_shape (cute.Shape) – A tuple or list representing the dimensions of the CTA tile.
layout_d (LayoutEnum) – The layout enum of the output tensor D.
elem_ty_d (Type[Numeric]) – The element type for output tensor D.
elem_ty_acc (Type[Numeric]) – The element type for accumulation.
epi_tile (cute.Tile) – The epilogue tile configuration.
use_2cta_instrs (bool) – A flag indicating whether the configuration is for 2 SMs.
Returns
:
An instance of Sm100TmemLoad with the computed configuration.
Return type
:
cute.CopyAtom
Raises
:
ValueError – If the function cannot handle the given combination of accumulation and dimension types, or if it cannot determine the appropriate configuration based on the input parameters.
cutlass.utils.get_num_tmem_alloc_cols(
tmem_tensors: cutlass.cute.typing.Tensor | List[cutlass.cute.typing.Tensor],
rounding=True,
) → int
Get the total number of TMEM allocation columns for the given TMEM tensors.

Parameters
:
tmem_tensors (Union[cute.Tensor, List[cute.Tensor]]) – The TMEM tensors to get the number of allocation columns for.
rounding (bool) – Whether to round up the number of allocation columns to the nearest power of 2.
Returns
:
The total number of TMEM allocation columns.
Return type
:
int
Raises
:
ValueError – If the number of TMEM allocation columns exceeds the maximum capacity of 512 or is less than 32.
cutlass.utils.make_smem_layout_a(
tiled_mma: TiledMma,
mma_tiler_mnk: cutlass.cute.typing.Tile,
a_dtype: Type[cutlass.cutlass_dsl.Numeric],
num_stages: int,
*,
loc=None,
ip=None,
) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout
This function helps with:

Get the partitioned shape of the A tensor based on the tiled_mma & MMA tiler.
Select the heuristic SMEM layout atom based on the A tensor’s majorness, the data type, and the major mode size.
cute.Tile the SMEM layout atom to the MMA tile shape.
Stage the SMEM layout based on the number of stages.
Parameters
:
tiled_mma (cute.TiledMma) – The tiled MMA used to partition tensor A
mma_tiler_mnk (cute.cute.Tile) – The MMA tile shape
a_dtype (Type[Numeric]) – The element type for tensor A
num_stages (int) – The number of pipeline stages for tensor A
Returns
:
SMEM layout for tensor A
Return type
:
Union[cute.Layout, cute.ComposedLayout]
cutlass.utils.make_smem_layout_b(
tiled_mma: TiledMma,
mma_tiler_mnk: cutlass.cute.typing.Tile,
b_dtype: Type[cutlass.cutlass_dsl.Numeric],
num_stages: int,
*,
loc=None,
ip=None,
) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout
This function helps:

Get the partitioned shape of the B tensor based on the tiled_mma & MMA tiler.
Select the heuristic SMEM layout atom based on the B tensor’s majorness, the data type, and the major mode size.
cute.Tile the SMEM layout atom to the MMA tile shape.
Stage the SMEM layout based on the number of stages.
Parameters
:
tiled_mma (cute.TiledMma) – The tiled MMA which is used to partition the B tensor.
mma_tiler_mnk (cute.cute.Tile) – The MMA tile shape.
b_dtype (Type[Numeric]) – The element type for the B tensor.
num_stages (int) – The stage of the B tensor.
Returns
:
SMEM layout for the B tensor.
Return type
:
Union[cute.Layout, cute.ComposedLayout]
cutlass.utils.make_smem_layout_epi(
epi_dtype: Type[cutlass.cutlass_dsl.Numeric],
epi_layout: LayoutEnum,
epi_tile: cutlass.cute.typing.Tile,
epi_stage: int,
*,
loc=None,
ip=None,
) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout
This function helps:

Select the heuristic SMEM layout atom based on the epilog tile shape, the epilog tensor’s majorness, and the element type.
cute.Tile the SMEM layout atom to the epilog tile shape.
Stage the SMEM layout based on the number of stages.
Parameters
:
epi_dtype (Type[Numeric]) – The element type for the epilog tensor.
epi_layout (LayoutEnum) – The layout enum for the epilog tensor.
epi_tile (cute.cute.Tile) – The epilogue tile shape.
epi_stage (int) – The stage of the epilog tensor.
Returns
:
SMEM layout for epilog tensors (usually C & D which are processed in the epilog)
Return type
:
Union[cute.Layout, cute.ComposedLayout]
cutlass.utils.make_trivial_tiled_mma(
ab_dtype: Type[cutlass.cutlass_dsl.Numeric],
a_leading_mode: OperandMajorMode,
b_leading_mode: OperandMajorMode,
acc_dtype: Type[cutlass.cutlass_dsl.Numeric],
cta_group: CtaGroup,
mma_tiler_mn: Tuple[int, int],
a_source: OperandSource = cutlass._mlir.dialects.cute.MmaFragKind.smem_desc,
*,
loc=None,
ip=None,
) → TiledMma
Make a tiled MMA atom with given data type, leading dimension, cta group and mma tile shape. By default, the MMA atom is created with SMEM operand source for A.

Parameters
:
ab_dtype (type[Numeric]) – Data type of operands A and B.
a_leading_mode (tcgen05.OperandMajorMode) – Leading dimension of operand A (1 for K, 0 for M/N).
b_leading_mode (tcgen05.OperandMajorMode) – Leading dimension of operand B (1 for K, 0 for M/N).
acc_dtype (type[Numeric]) – Data type of the accumulator.
cta_group (tcgen05.CtaGroup) – The CTA group to use.
mma_tiler_mn (Tuple[int, int]) – The shape (M, N, K) of the MMA tiler.
a_source (cutlass.cute.nvgpu.tcgen05.OperandSource) – The source of operand A (SMEM by default or TMEM).
Returns
:
A tiled MMA atom.
Return type
:
cute.TiledMma
Raises
:
TypeError – If the data type is not supported.
cutlass.utils.make_blockscaled_trivial_tiled_mma(
ab_dtype: Type[cutlass.cutlass_dsl.Numeric],
a_leading_mode: OperandMajorMode,
b_leading_mode: OperandMajorMode,
sf_dtype: Type[cutlass.cutlass_dsl.Numeric],
sf_vec_size: int,
cta_group: CtaGroup,
mma_tiler_mn: Tuple[int, int],
a_source: OperandSource = cutlass._mlir.dialects.cute.MmaFragKind.smem_desc,
*,
loc=None,
ip=None,
) → TiledMma
Make a BlockScaled tiled MMA atom with given data type, leading dimension, cta group and mma tile shape. By default, the MMA atom is created with SMEM operand source for A.

Parameters
:
ab_dtype (type[Numeric]) – Data type of operands A and B.
a_leading_mode (tcgen05.OperandMajorMode) – Leading dimension of operand A (1 for K, 0 for M/N).
b_leading_mode (tcgen05.OperandMajorMode) – Leading dimension of operand B (1 for K, 0 for M/N).
sf_dtype (type[Numeric]) – Data type of the Scale Factor.
sf_vec_size (int) – The vector size of the Scale Factor.
cta_group (tcgen05.CtaGroup) – The CTA group to use.
mma_tiler_mn (Tuple[int, int]) – The shape (M, N, K) of the MMA tiler.
a_source (cutlass.cute.nvgpu.tcgen05.OperandSource) – The source of operand A (SMEM by default or TMEM).
Returns
:
A tiled MMA atom.
Return type
:
cute.TiledMma
Raises
:
TypeError – If the data type is not supported.