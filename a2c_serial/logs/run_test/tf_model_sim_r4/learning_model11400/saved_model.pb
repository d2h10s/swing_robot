÷
Ü
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.6.02v2.6.0-rc2-32-g919f693420e8°þ
w
Dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameDense1/kernel
p
!Dense1/kernel/Read/ReadVariableOpReadVariableOpDense1/kernel*
_output_shapes
:	*
dtype0
o
Dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameDense1/bias
h
Dense1/bias/Read/ReadVariableOpReadVariableOpDense1/bias*
_output_shapes	
:*
dtype0
u
Actor/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameActor/kernel
n
 Actor/kernel/Read/ReadVariableOpReadVariableOpActor/kernel*
_output_shapes
:	*
dtype0
l

Actor/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
Actor/bias
e
Actor/bias/Read/ReadVariableOpReadVariableOp
Actor/bias*
_output_shapes
:*
dtype0
w
Critic/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameCritic/kernel
p
!Critic/kernel/Read/ReadVariableOpReadVariableOpCritic/kernel*
_output_shapes
:	*
dtype0
n
Critic/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameCritic/bias
g
Critic/bias/Read/ReadVariableOpReadVariableOpCritic/bias*
_output_shapes
:*
dtype0

NoOpNoOp
³
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*î
valueäBá BÚ
j
nn
trainable_variables
	variables
regularization_losses
	keras_api

signatures
Ô
layer-0
layer_with_weights-0
layer-1
	layer_with_weights-1
	layer-2

layer_with_weights-2

layer-3
trainable_variables
	variables
regularization_losses
	keras_api
*
0
1
2
3
4
5
*
0
1
2
3
4
5
 
­
trainable_variables
non_trainable_variables
	variables
layer_metrics
metrics

layers
regularization_losses
layer_regularization_losses
 
 
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
 regularization_losses
!	keras_api
h

kernel
bias
"trainable_variables
#	variables
$regularization_losses
%	keras_api
*
0
1
2
3
4
5
*
0
1
2
3
4
5
 
­
trainable_variables
&non_trainable_variables
	variables
'layer_metrics
(metrics

)layers
regularization_losses
*layer_regularization_losses
SQ
VARIABLE_VALUEDense1/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEDense1/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEActor/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUE
Actor/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUECritic/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUECritic/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

0
 

0
1

0
1
 
­
trainable_variables
+non_trainable_variables
	variables
,layer_metrics
-metrics

.layers
regularization_losses
/layer_regularization_losses

0
1

0
1
 
­
trainable_variables
0non_trainable_variables
	variables
1layer_metrics
2metrics

3layers
 regularization_losses
4layer_regularization_losses

0
1

0
1
 
­
"trainable_variables
5non_trainable_variables
#	variables
6layer_metrics
7metrics

8layers
$regularization_losses
9layer_regularization_losses
 
 
 

0
1
	2

3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
r
serving_default_input_1Placeholder*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Dense1/kernelDense1/biasCritic/kernelCritic/biasActor/kernel
Actor/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
::*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_signature_wrapper_1018469509
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ò
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!Dense1/kernel/Read/ReadVariableOpDense1/bias/Read/ReadVariableOp Actor/kernel/Read/ReadVariableOpActor/bias/Read/ReadVariableOp!Critic/kernel/Read/ReadVariableOpCritic/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__traced_save_1018469961
õ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameDense1/kernelDense1/biasActor/kernel
Actor/biasCritic/kernelCritic/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 */
f*R(
&__inference__traced_restore_1018469989ÍÔ

ù
F__inference_Dense1_layer_call_and_return_conditional_losses_1018469036

inputs1
matmul_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù

*__inference_Actor_layer_call_fn_1018469890

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_Actor_layer_call_and_return_conditional_losses_10184690682
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·


.__inference_a2c_model_layer_call_fn_1018469566
x
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:
	unknown_3:	
	unknown_4:
identity

identity_1¢StatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
::*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_a2c_model_layer_call_and_return_conditional_losses_10184694102
StatefulPartitionedCallr
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identityv

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes

:2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:F B
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex

ü
#__inference__traced_save_1018469961
file_prefix,
(savev2_dense1_kernel_read_readvariableop*
&savev2_dense1_bias_read_readvariableop+
'savev2_actor_kernel_read_readvariableop)
%savev2_actor_bias_read_readvariableop,
(savev2_critic_kernel_read_readvariableop*
&savev2_critic_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameÍ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ß
valueÕBÒB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
SaveV2/shape_and_slices´
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_dense1_kernel_read_readvariableop&savev2_dense1_bias_read_readvariableop'savev2_actor_kernel_read_readvariableop%savev2_actor_bias_read_readvariableop(savev2_critic_kernel_read_readvariableop&savev2_critic_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	22
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*K
_input_shapes:
8: :	::	::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	: 

_output_shapes
::

_output_shapes
: 
·
«
E__inference_model_layer_call_and_return_conditional_losses_1018469240
input_1$
dense1_1018469223:	 
dense1_1018469225:	$
critic_1018469228:	
critic_1018469230:#
actor_1018469233:	
actor_1018469235:
identity

identity_1¢Actor/StatefulPartitionedCall¢Critic/StatefulPartitionedCall¢Dense1/StatefulPartitionedCall
Dense1/StatefulPartitionedCallStatefulPartitionedCallinput_1dense1_1018469223dense1_1018469225*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_Dense1_layer_call_and_return_conditional_losses_10184690362 
Dense1/StatefulPartitionedCallº
Critic/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0critic_1018469228critic_1018469230*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_Critic_layer_call_and_return_conditional_losses_10184690522 
Critic/StatefulPartitionedCallµ
Actor/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0actor_1018469233actor_1018469235*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_Actor_layer_call_and_return_conditional_losses_10184690682
Actor/StatefulPartitionedCall
IdentityIdentity&Actor/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity'Critic/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1°
NoOpNoOp^Actor/StatefulPartitionedCall^Critic/StatefulPartitionedCall^Dense1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2>
Actor/StatefulPartitionedCallActor/StatefulPartitionedCall2@
Critic/StatefulPartitionedCallCritic/StatefulPartitionedCall2@
Dense1/StatefulPartitionedCallDense1/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ð
þ
E__inference_model_layer_call_and_return_conditional_losses_1018469837

inputs8
%dense1_matmul_readvariableop_resource:	5
&dense1_biasadd_readvariableop_resource:	8
%critic_matmul_readvariableop_resource:	4
&critic_biasadd_readvariableop_resource:7
$actor_matmul_readvariableop_resource:	3
%actor_biasadd_readvariableop_resource:
identity

identity_1¢Actor/BiasAdd/ReadVariableOp¢Actor/MatMul/ReadVariableOp¢Critic/BiasAdd/ReadVariableOp¢Critic/MatMul/ReadVariableOp¢Dense1/BiasAdd/ReadVariableOp¢Dense1/MatMul/ReadVariableOp£
Dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
Dense1/MatMul/ReadVariableOp
Dense1/MatMulMatMulinputs$Dense1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
Dense1/MatMul¢
Dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
Dense1/BiasAdd/ReadVariableOp
Dense1/BiasAddBiasAddDense1/MatMul:product:0%Dense1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
Dense1/BiasAdde
Dense1/ReluReluDense1/BiasAdd:output:0*
T0*
_output_shapes
:	2
Dense1/Relu£
Critic/MatMul/ReadVariableOpReadVariableOp%critic_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
Critic/MatMul/ReadVariableOp
Critic/MatMulMatMulDense1/Relu:activations:0$Critic/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Critic/MatMul¡
Critic/BiasAdd/ReadVariableOpReadVariableOp&critic_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
Critic/BiasAdd/ReadVariableOp
Critic/BiasAddBiasAddCritic/MatMul:product:0%Critic/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Critic/BiasAdd 
Actor/MatMul/ReadVariableOpReadVariableOp$actor_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
Actor/MatMul/ReadVariableOp
Actor/MatMulMatMulDense1/Relu:activations:0#Actor/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Actor/MatMul
Actor/BiasAdd/ReadVariableOpReadVariableOp%actor_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
Actor/BiasAdd/ReadVariableOp
Actor/BiasAddBiasAddActor/MatMul:product:0$Actor/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Actor/BiasAddh
IdentityIdentityActor/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:2

Identitym

Identity_1IdentityCritic/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:2

Identity_1
NoOpNoOp^Actor/BiasAdd/ReadVariableOp^Actor/MatMul/ReadVariableOp^Critic/BiasAdd/ReadVariableOp^Critic/MatMul/ReadVariableOp^Dense1/BiasAdd/ReadVariableOp^Dense1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:: : : : : : 2<
Actor/BiasAdd/ReadVariableOpActor/BiasAdd/ReadVariableOp2:
Actor/MatMul/ReadVariableOpActor/MatMul/ReadVariableOp2>
Critic/BiasAdd/ReadVariableOpCritic/BiasAdd/ReadVariableOp2<
Critic/MatMul/ReadVariableOpCritic/MatMul/ReadVariableOp2>
Dense1/BiasAdd/ReadVariableOpDense1/BiasAdd/ReadVariableOp2<
Dense1/MatMul/ReadVariableOpDense1/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´
ª
E__inference_model_layer_call_and_return_conditional_losses_1018469076

inputs$
dense1_1018469037:	 
dense1_1018469039:	$
critic_1018469053:	
critic_1018469055:#
actor_1018469069:	
actor_1018469071:
identity

identity_1¢Actor/StatefulPartitionedCall¢Critic/StatefulPartitionedCall¢Dense1/StatefulPartitionedCall
Dense1/StatefulPartitionedCallStatefulPartitionedCallinputsdense1_1018469037dense1_1018469039*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_Dense1_layer_call_and_return_conditional_losses_10184690362 
Dense1/StatefulPartitionedCallº
Critic/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0critic_1018469053critic_1018469055*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_Critic_layer_call_and_return_conditional_losses_10184690522 
Critic/StatefulPartitionedCallµ
Actor/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0actor_1018469069actor_1018469071*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_Actor_layer_call_and_return_conditional_losses_10184690682
Actor/StatefulPartitionedCall
IdentityIdentity&Actor/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity'Critic/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1°
NoOpNoOp^Actor/StatefulPartitionedCall^Critic/StatefulPartitionedCall^Dense1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2>
Actor/StatefulPartitionedCallActor/StatefulPartitionedCall2@
Critic/StatefulPartitionedCallCritic/StatefulPartitionedCall2@
Dense1/StatefulPartitionedCallDense1/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
"
Å
I__inference_a2c_model_layer_call_and_return_conditional_losses_1018469611
x>
+model_dense1_matmul_readvariableop_resource:	;
,model_dense1_biasadd_readvariableop_resource:	>
+model_critic_matmul_readvariableop_resource:	:
,model_critic_biasadd_readvariableop_resource:=
*model_actor_matmul_readvariableop_resource:	9
+model_actor_biasadd_readvariableop_resource:
identity

identity_1¢"model/Actor/BiasAdd/ReadVariableOp¢!model/Actor/MatMul/ReadVariableOp¢#model/Critic/BiasAdd/ReadVariableOp¢"model/Critic/MatMul/ReadVariableOp¢#model/Dense1/BiasAdd/ReadVariableOp¢"model/Dense1/MatMul/ReadVariableOpb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dimk

ExpandDims
ExpandDimsxExpandDims/dim:output:0*
T0*
_output_shapes

:2

ExpandDimsµ
"model/Dense1/MatMul/ReadVariableOpReadVariableOp+model_dense1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02$
"model/Dense1/MatMul/ReadVariableOp
model/Dense1/MatMulMatMulExpandDims:output:0*model/Dense1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
model/Dense1/MatMul´
#model/Dense1/BiasAdd/ReadVariableOpReadVariableOp,model_dense1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#model/Dense1/BiasAdd/ReadVariableOp­
model/Dense1/BiasAddBiasAddmodel/Dense1/MatMul:product:0+model/Dense1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
model/Dense1/BiasAddw
model/Dense1/ReluRelumodel/Dense1/BiasAdd:output:0*
T0*
_output_shapes
:	2
model/Dense1/Reluµ
"model/Critic/MatMul/ReadVariableOpReadVariableOp+model_critic_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02$
"model/Critic/MatMul/ReadVariableOpª
model/Critic/MatMulMatMulmodel/Dense1/Relu:activations:0*model/Critic/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
model/Critic/MatMul³
#model/Critic/BiasAdd/ReadVariableOpReadVariableOp,model_critic_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#model/Critic/BiasAdd/ReadVariableOp¬
model/Critic/BiasAddBiasAddmodel/Critic/MatMul:product:0+model/Critic/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
model/Critic/BiasAdd²
!model/Actor/MatMul/ReadVariableOpReadVariableOp*model_actor_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02#
!model/Actor/MatMul/ReadVariableOp§
model/Actor/MatMulMatMulmodel/Dense1/Relu:activations:0)model/Actor/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
model/Actor/MatMul°
"model/Actor/BiasAdd/ReadVariableOpReadVariableOp+model_actor_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"model/Actor/BiasAdd/ReadVariableOp¨
model/Actor/BiasAddBiasAddmodel/Actor/MatMul:product:0*model/Actor/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
model/Actor/BiasAddn
IdentityIdentitymodel/Actor/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:2

Identitys

Identity_1Identitymodel/Critic/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:2

Identity_1­
NoOpNoOp#^model/Actor/BiasAdd/ReadVariableOp"^model/Actor/MatMul/ReadVariableOp$^model/Critic/BiasAdd/ReadVariableOp#^model/Critic/MatMul/ReadVariableOp$^model/Dense1/BiasAdd/ReadVariableOp#^model/Dense1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2H
"model/Actor/BiasAdd/ReadVariableOp"model/Actor/BiasAdd/ReadVariableOp2F
!model/Actor/MatMul/ReadVariableOp!model/Actor/MatMul/ReadVariableOp2J
#model/Critic/BiasAdd/ReadVariableOp#model/Critic/BiasAdd/ReadVariableOp2H
"model/Critic/MatMul/ReadVariableOp"model/Critic/MatMul/ReadVariableOp2J
#model/Dense1/BiasAdd/ReadVariableOp#model/Dense1/BiasAdd/ReadVariableOp2H
"model/Dense1/MatMul/ReadVariableOp"model/Dense1/MatMul/ReadVariableOp:F B
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
£"
Ë
I__inference_a2c_model_layer_call_and_return_conditional_losses_1018469663
input_1>
+model_dense1_matmul_readvariableop_resource:	;
,model_dense1_biasadd_readvariableop_resource:	>
+model_critic_matmul_readvariableop_resource:	:
,model_critic_biasadd_readvariableop_resource:=
*model_actor_matmul_readvariableop_resource:	9
+model_actor_biasadd_readvariableop_resource:
identity

identity_1¢"model/Actor/BiasAdd/ReadVariableOp¢!model/Actor/MatMul/ReadVariableOp¢#model/Critic/BiasAdd/ReadVariableOp¢"model/Critic/MatMul/ReadVariableOp¢#model/Dense1/BiasAdd/ReadVariableOp¢"model/Dense1/MatMul/ReadVariableOpb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dimq

ExpandDims
ExpandDimsinput_1ExpandDims/dim:output:0*
T0*
_output_shapes

:2

ExpandDimsµ
"model/Dense1/MatMul/ReadVariableOpReadVariableOp+model_dense1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02$
"model/Dense1/MatMul/ReadVariableOp
model/Dense1/MatMulMatMulExpandDims:output:0*model/Dense1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
model/Dense1/MatMul´
#model/Dense1/BiasAdd/ReadVariableOpReadVariableOp,model_dense1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#model/Dense1/BiasAdd/ReadVariableOp­
model/Dense1/BiasAddBiasAddmodel/Dense1/MatMul:product:0+model/Dense1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
model/Dense1/BiasAddw
model/Dense1/ReluRelumodel/Dense1/BiasAdd:output:0*
T0*
_output_shapes
:	2
model/Dense1/Reluµ
"model/Critic/MatMul/ReadVariableOpReadVariableOp+model_critic_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02$
"model/Critic/MatMul/ReadVariableOpª
model/Critic/MatMulMatMulmodel/Dense1/Relu:activations:0*model/Critic/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
model/Critic/MatMul³
#model/Critic/BiasAdd/ReadVariableOpReadVariableOp,model_critic_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#model/Critic/BiasAdd/ReadVariableOp¬
model/Critic/BiasAddBiasAddmodel/Critic/MatMul:product:0+model/Critic/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
model/Critic/BiasAdd²
!model/Actor/MatMul/ReadVariableOpReadVariableOp*model_actor_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02#
!model/Actor/MatMul/ReadVariableOp§
model/Actor/MatMulMatMulmodel/Dense1/Relu:activations:0)model/Actor/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
model/Actor/MatMul°
"model/Actor/BiasAdd/ReadVariableOpReadVariableOp+model_actor_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"model/Actor/BiasAdd/ReadVariableOp¨
model/Actor/BiasAddBiasAddmodel/Actor/MatMul:product:0*model/Actor/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
model/Actor/BiasAddn
IdentityIdentitymodel/Actor/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:2

Identitys

Identity_1Identitymodel/Critic/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:2

Identity_1­
NoOpNoOp#^model/Actor/BiasAdd/ReadVariableOp"^model/Actor/MatMul/ReadVariableOp$^model/Critic/BiasAdd/ReadVariableOp#^model/Critic/MatMul/ReadVariableOp$^model/Dense1/BiasAdd/ReadVariableOp#^model/Dense1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2H
"model/Actor/BiasAdd/ReadVariableOp"model/Actor/BiasAdd/ReadVariableOp2F
!model/Actor/MatMul/ReadVariableOp!model/Actor/MatMul/ReadVariableOp2J
#model/Critic/BiasAdd/ReadVariableOp#model/Critic/BiasAdd/ReadVariableOp2H
"model/Critic/MatMul/ReadVariableOp"model/Critic/MatMul/ReadVariableOp2J
#model/Dense1/BiasAdd/ReadVariableOp#model/Dense1/BiasAdd/ReadVariableOp2H
"model/Dense1/MatMul/ReadVariableOp"model/Dense1/MatMul/ReadVariableOp:L H
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
«

÷
E__inference_Actor_layer_call_and_return_conditional_losses_1018469068

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â
ã
I__inference_a2c_model_layer_call_and_return_conditional_losses_1018469289
x#
model_1018469273:	
model_1018469275:	#
model_1018469277:	
model_1018469279:#
model_1018469281:	
model_1018469283:
identity

identity_1¢model/StatefulPartitionedCallb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dimt

ExpandDims
ExpandDimsxExpandDims/dim:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

ExpandDimsó
model/StatefulPartitionedCallStatefulPartitionedCallExpandDims:output:0model_1018469273model_1018469275model_1018469277model_1018469279model_1018469281model_1018469283*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
::*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_layer_call_and_return_conditional_losses_10184692722
model/StatefulPartitionedCallx
IdentityIdentity&model/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identity|

Identity_1Identity&model/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes

:2

Identity_1n
NoOpNoOp^model/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall:F B
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
µ(

%__inference__wrapped_model_1018469018
input_1H
5a2c_model_model_dense1_matmul_readvariableop_resource:	E
6a2c_model_model_dense1_biasadd_readvariableop_resource:	H
5a2c_model_model_critic_matmul_readvariableop_resource:	D
6a2c_model_model_critic_biasadd_readvariableop_resource:G
4a2c_model_model_actor_matmul_readvariableop_resource:	C
5a2c_model_model_actor_biasadd_readvariableop_resource:
identity

identity_1¢,a2c_model/model/Actor/BiasAdd/ReadVariableOp¢+a2c_model/model/Actor/MatMul/ReadVariableOp¢-a2c_model/model/Critic/BiasAdd/ReadVariableOp¢,a2c_model/model/Critic/MatMul/ReadVariableOp¢-a2c_model/model/Dense1/BiasAdd/ReadVariableOp¢,a2c_model/model/Dense1/MatMul/ReadVariableOpv
a2c_model/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
a2c_model/ExpandDims/dim
a2c_model/ExpandDims
ExpandDimsinput_1!a2c_model/ExpandDims/dim:output:0*
T0*
_output_shapes

:2
a2c_model/ExpandDimsÓ
,a2c_model/model/Dense1/MatMul/ReadVariableOpReadVariableOp5a2c_model_model_dense1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02.
,a2c_model/model/Dense1/MatMul/ReadVariableOpÇ
a2c_model/model/Dense1/MatMulMatMula2c_model/ExpandDims:output:04a2c_model/model/Dense1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
a2c_model/model/Dense1/MatMulÒ
-a2c_model/model/Dense1/BiasAdd/ReadVariableOpReadVariableOp6a2c_model_model_dense1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-a2c_model/model/Dense1/BiasAdd/ReadVariableOpÕ
a2c_model/model/Dense1/BiasAddBiasAdd'a2c_model/model/Dense1/MatMul:product:05a2c_model/model/Dense1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2 
a2c_model/model/Dense1/BiasAdd
a2c_model/model/Dense1/ReluRelu'a2c_model/model/Dense1/BiasAdd:output:0*
T0*
_output_shapes
:	2
a2c_model/model/Dense1/ReluÓ
,a2c_model/model/Critic/MatMul/ReadVariableOpReadVariableOp5a2c_model_model_critic_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02.
,a2c_model/model/Critic/MatMul/ReadVariableOpÒ
a2c_model/model/Critic/MatMulMatMul)a2c_model/model/Dense1/Relu:activations:04a2c_model/model/Critic/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
a2c_model/model/Critic/MatMulÑ
-a2c_model/model/Critic/BiasAdd/ReadVariableOpReadVariableOp6a2c_model_model_critic_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-a2c_model/model/Critic/BiasAdd/ReadVariableOpÔ
a2c_model/model/Critic/BiasAddBiasAdd'a2c_model/model/Critic/MatMul:product:05a2c_model/model/Critic/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2 
a2c_model/model/Critic/BiasAddÐ
+a2c_model/model/Actor/MatMul/ReadVariableOpReadVariableOp4a2c_model_model_actor_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02-
+a2c_model/model/Actor/MatMul/ReadVariableOpÏ
a2c_model/model/Actor/MatMulMatMul)a2c_model/model/Dense1/Relu:activations:03a2c_model/model/Actor/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
a2c_model/model/Actor/MatMulÎ
,a2c_model/model/Actor/BiasAdd/ReadVariableOpReadVariableOp5a2c_model_model_actor_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,a2c_model/model/Actor/BiasAdd/ReadVariableOpÐ
a2c_model/model/Actor/BiasAddBiasAdd&a2c_model/model/Actor/MatMul:product:04a2c_model/model/Actor/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
a2c_model/model/Actor/BiasAddx
IdentityIdentity&a2c_model/model/Actor/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:2

Identity}

Identity_1Identity'a2c_model/model/Critic/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:2

Identity_1é
NoOpNoOp-^a2c_model/model/Actor/BiasAdd/ReadVariableOp,^a2c_model/model/Actor/MatMul/ReadVariableOp.^a2c_model/model/Critic/BiasAdd/ReadVariableOp-^a2c_model/model/Critic/MatMul/ReadVariableOp.^a2c_model/model/Dense1/BiasAdd/ReadVariableOp-^a2c_model/model/Dense1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2\
,a2c_model/model/Actor/BiasAdd/ReadVariableOp,a2c_model/model/Actor/BiasAdd/ReadVariableOp2Z
+a2c_model/model/Actor/MatMul/ReadVariableOp+a2c_model/model/Actor/MatMul/ReadVariableOp2^
-a2c_model/model/Critic/BiasAdd/ReadVariableOp-a2c_model/model/Critic/BiasAdd/ReadVariableOp2\
,a2c_model/model/Critic/MatMul/ReadVariableOp,a2c_model/model/Critic/MatMul/ReadVariableOp2^
-a2c_model/model/Dense1/BiasAdd/ReadVariableOp-a2c_model/model/Dense1/BiasAdd/ReadVariableOp2\
,a2c_model/model/Dense1/MatMul/ReadVariableOp,a2c_model/model/Dense1/MatMul/ReadVariableOp:L H
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ê


*__inference_model_layer_call_fn_1018469727

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:
	unknown_3:	
	unknown_4:
identity

identity_1¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_layer_call_and_return_conditional_losses_10184691642
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê
þ
E__inference_model_layer_call_and_return_conditional_losses_1018469813

inputs8
%dense1_matmul_readvariableop_resource:	5
&dense1_biasadd_readvariableop_resource:	8
%critic_matmul_readvariableop_resource:	4
&critic_biasadd_readvariableop_resource:7
$actor_matmul_readvariableop_resource:	3
%actor_biasadd_readvariableop_resource:
identity

identity_1¢Actor/BiasAdd/ReadVariableOp¢Actor/MatMul/ReadVariableOp¢Critic/BiasAdd/ReadVariableOp¢Critic/MatMul/ReadVariableOp¢Dense1/BiasAdd/ReadVariableOp¢Dense1/MatMul/ReadVariableOp£
Dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
Dense1/MatMul/ReadVariableOp
Dense1/MatMulMatMulinputs$Dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dense1/MatMul¢
Dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
Dense1/BiasAdd/ReadVariableOp
Dense1/BiasAddBiasAddDense1/MatMul:product:0%Dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dense1/BiasAddn
Dense1/ReluReluDense1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dense1/Relu£
Critic/MatMul/ReadVariableOpReadVariableOp%critic_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
Critic/MatMul/ReadVariableOp
Critic/MatMulMatMulDense1/Relu:activations:0$Critic/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Critic/MatMul¡
Critic/BiasAdd/ReadVariableOpReadVariableOp&critic_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
Critic/BiasAdd/ReadVariableOp
Critic/BiasAddBiasAddCritic/MatMul:product:0%Critic/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Critic/BiasAdd 
Actor/MatMul/ReadVariableOpReadVariableOp$actor_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
Actor/MatMul/ReadVariableOp
Actor/MatMulMatMulDense1/Relu:activations:0#Actor/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Actor/MatMul
Actor/BiasAdd/ReadVariableOpReadVariableOp%actor_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
Actor/BiasAdd/ReadVariableOp
Actor/BiasAddBiasAddActor/MatMul:product:0$Actor/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Actor/BiasAddq
IdentityIdentityActor/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityv

Identity_1IdentityCritic/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1
NoOpNoOp^Actor/BiasAdd/ReadVariableOp^Actor/MatMul/ReadVariableOp^Critic/BiasAdd/ReadVariableOp^Critic/MatMul/ReadVariableOp^Dense1/BiasAdd/ReadVariableOp^Dense1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2<
Actor/BiasAdd/ReadVariableOpActor/BiasAdd/ReadVariableOp2:
Actor/MatMul/ReadVariableOpActor/MatMul/ReadVariableOp2>
Critic/BiasAdd/ReadVariableOpCritic/BiasAdd/ReadVariableOp2<
Critic/MatMul/ReadVariableOpCritic/MatMul/ReadVariableOp2>
Dense1/BiasAdd/ReadVariableOpDense1/BiasAdd/ReadVariableOp2<
Dense1/MatMul/ReadVariableOpDense1/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬

ø
F__inference_Critic_layer_call_and_return_conditional_losses_1018469919

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü

+__inference_Dense1_layer_call_fn_1018469870

inputs
unknown:	
	unknown_0:	
identity¢StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_Dense1_layer_call_and_return_conditional_losses_10184690362
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
£"
Ë
I__inference_a2c_model_layer_call_and_return_conditional_losses_1018469689
input_1>
+model_dense1_matmul_readvariableop_resource:	;
,model_dense1_biasadd_readvariableop_resource:	>
+model_critic_matmul_readvariableop_resource:	:
,model_critic_biasadd_readvariableop_resource:=
*model_actor_matmul_readvariableop_resource:	9
+model_actor_biasadd_readvariableop_resource:
identity

identity_1¢"model/Actor/BiasAdd/ReadVariableOp¢!model/Actor/MatMul/ReadVariableOp¢#model/Critic/BiasAdd/ReadVariableOp¢"model/Critic/MatMul/ReadVariableOp¢#model/Dense1/BiasAdd/ReadVariableOp¢"model/Dense1/MatMul/ReadVariableOpb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dimq

ExpandDims
ExpandDimsinput_1ExpandDims/dim:output:0*
T0*
_output_shapes

:2

ExpandDimsµ
"model/Dense1/MatMul/ReadVariableOpReadVariableOp+model_dense1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02$
"model/Dense1/MatMul/ReadVariableOp
model/Dense1/MatMulMatMulExpandDims:output:0*model/Dense1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
model/Dense1/MatMul´
#model/Dense1/BiasAdd/ReadVariableOpReadVariableOp,model_dense1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#model/Dense1/BiasAdd/ReadVariableOp­
model/Dense1/BiasAddBiasAddmodel/Dense1/MatMul:product:0+model/Dense1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
model/Dense1/BiasAddw
model/Dense1/ReluRelumodel/Dense1/BiasAdd:output:0*
T0*
_output_shapes
:	2
model/Dense1/Reluµ
"model/Critic/MatMul/ReadVariableOpReadVariableOp+model_critic_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02$
"model/Critic/MatMul/ReadVariableOpª
model/Critic/MatMulMatMulmodel/Dense1/Relu:activations:0*model/Critic/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
model/Critic/MatMul³
#model/Critic/BiasAdd/ReadVariableOpReadVariableOp,model_critic_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#model/Critic/BiasAdd/ReadVariableOp¬
model/Critic/BiasAddBiasAddmodel/Critic/MatMul:product:0+model/Critic/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
model/Critic/BiasAdd²
!model/Actor/MatMul/ReadVariableOpReadVariableOp*model_actor_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02#
!model/Actor/MatMul/ReadVariableOp§
model/Actor/MatMulMatMulmodel/Dense1/Relu:activations:0)model/Actor/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
model/Actor/MatMul°
"model/Actor/BiasAdd/ReadVariableOpReadVariableOp+model_actor_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"model/Actor/BiasAdd/ReadVariableOp¨
model/Actor/BiasAddBiasAddmodel/Actor/MatMul:product:0*model/Actor/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
model/Actor/BiasAddn
IdentityIdentitymodel/Actor/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:2

Identitys

Identity_1Identitymodel/Critic/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:2

Identity_1­
NoOpNoOp#^model/Actor/BiasAdd/ReadVariableOp"^model/Actor/MatMul/ReadVariableOp$^model/Critic/BiasAdd/ReadVariableOp#^model/Critic/MatMul/ReadVariableOp$^model/Dense1/BiasAdd/ReadVariableOp#^model/Dense1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2H
"model/Actor/BiasAdd/ReadVariableOp"model/Actor/BiasAdd/ReadVariableOp2F
!model/Actor/MatMul/ReadVariableOp!model/Actor/MatMul/ReadVariableOp2J
#model/Critic/BiasAdd/ReadVariableOp#model/Critic/BiasAdd/ReadVariableOp2H
"model/Critic/MatMul/ReadVariableOp"model/Critic/MatMul/ReadVariableOp2J
#model/Dense1/BiasAdd/ReadVariableOp#model/Dense1/BiasAdd/ReadVariableOp2H
"model/Dense1/MatMul/ReadVariableOp"model/Dense1/MatMul/ReadVariableOp:L H
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ð
þ
E__inference_model_layer_call_and_return_conditional_losses_1018469351

inputs8
%dense1_matmul_readvariableop_resource:	5
&dense1_biasadd_readvariableop_resource:	8
%critic_matmul_readvariableop_resource:	4
&critic_biasadd_readvariableop_resource:7
$actor_matmul_readvariableop_resource:	3
%actor_biasadd_readvariableop_resource:
identity

identity_1¢Actor/BiasAdd/ReadVariableOp¢Actor/MatMul/ReadVariableOp¢Critic/BiasAdd/ReadVariableOp¢Critic/MatMul/ReadVariableOp¢Dense1/BiasAdd/ReadVariableOp¢Dense1/MatMul/ReadVariableOp£
Dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
Dense1/MatMul/ReadVariableOp
Dense1/MatMulMatMulinputs$Dense1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
Dense1/MatMul¢
Dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
Dense1/BiasAdd/ReadVariableOp
Dense1/BiasAddBiasAddDense1/MatMul:product:0%Dense1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
Dense1/BiasAdde
Dense1/ReluReluDense1/BiasAdd:output:0*
T0*
_output_shapes
:	2
Dense1/Relu£
Critic/MatMul/ReadVariableOpReadVariableOp%critic_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
Critic/MatMul/ReadVariableOp
Critic/MatMulMatMulDense1/Relu:activations:0$Critic/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Critic/MatMul¡
Critic/BiasAdd/ReadVariableOpReadVariableOp&critic_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
Critic/BiasAdd/ReadVariableOp
Critic/BiasAddBiasAddCritic/MatMul:product:0%Critic/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Critic/BiasAdd 
Actor/MatMul/ReadVariableOpReadVariableOp$actor_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
Actor/MatMul/ReadVariableOp
Actor/MatMulMatMulDense1/Relu:activations:0#Actor/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Actor/MatMul
Actor/BiasAdd/ReadVariableOpReadVariableOp%actor_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
Actor/BiasAdd/ReadVariableOp
Actor/BiasAddBiasAddActor/MatMul:product:0$Actor/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Actor/BiasAddh
IdentityIdentityActor/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:2

Identitym

Identity_1IdentityCritic/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:2

Identity_1
NoOpNoOp^Actor/BiasAdd/ReadVariableOp^Actor/MatMul/ReadVariableOp^Critic/BiasAdd/ReadVariableOp^Critic/MatMul/ReadVariableOp^Dense1/BiasAdd/ReadVariableOp^Dense1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:: : : : : : 2<
Actor/BiasAdd/ReadVariableOpActor/BiasAdd/ReadVariableOp2:
Actor/MatMul/ReadVariableOpActor/MatMul/ReadVariableOp2>
Critic/BiasAdd/ReadVariableOpCritic/BiasAdd/ReadVariableOp2<
Critic/MatMul/ReadVariableOpCritic/MatMul/ReadVariableOp2>
Dense1/BiasAdd/ReadVariableOpDense1/BiasAdd/ReadVariableOp2<
Dense1/MatMul/ReadVariableOpDense1/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·
«
E__inference_model_layer_call_and_return_conditional_losses_1018469220
input_1$
dense1_1018469203:	 
dense1_1018469205:	$
critic_1018469208:	
critic_1018469210:#
actor_1018469213:	
actor_1018469215:
identity

identity_1¢Actor/StatefulPartitionedCall¢Critic/StatefulPartitionedCall¢Dense1/StatefulPartitionedCall
Dense1/StatefulPartitionedCallStatefulPartitionedCallinput_1dense1_1018469203dense1_1018469205*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_Dense1_layer_call_and_return_conditional_losses_10184690362 
Dense1/StatefulPartitionedCallº
Critic/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0critic_1018469208critic_1018469210*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_Critic_layer_call_and_return_conditional_losses_10184690522 
Critic/StatefulPartitionedCallµ
Actor/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0actor_1018469213actor_1018469215*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_Actor_layer_call_and_return_conditional_losses_10184690682
Actor/StatefulPartitionedCall
IdentityIdentity&Actor/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity'Critic/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1°
NoOpNoOp^Actor/StatefulPartitionedCall^Critic/StatefulPartitionedCall^Dense1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2>
Actor/StatefulPartitionedCallActor/StatefulPartitionedCall2@
Critic/StatefulPartitionedCallCritic/StatefulPartitionedCall2@
Dense1/StatefulPartitionedCallDense1/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
É


.__inference_a2c_model_layer_call_fn_1018469528
input_1
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:
	unknown_3:	
	unknown_4:
identity

identity_1¢StatefulPartitionedCall³
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
::*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_a2c_model_layer_call_and_return_conditional_losses_10184692892
StatefulPartitionedCallr
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identityv

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes

:2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
í


*__inference_model_layer_call_fn_1018469200
input_1
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:
	unknown_3:	
	unknown_4:
identity

identity_1¢StatefulPartitionedCallÁ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_layer_call_and_return_conditional_losses_10184691642
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ê
þ
E__inference_model_layer_call_and_return_conditional_losses_1018469789

inputs8
%dense1_matmul_readvariableop_resource:	5
&dense1_biasadd_readvariableop_resource:	8
%critic_matmul_readvariableop_resource:	4
&critic_biasadd_readvariableop_resource:7
$actor_matmul_readvariableop_resource:	3
%actor_biasadd_readvariableop_resource:
identity

identity_1¢Actor/BiasAdd/ReadVariableOp¢Actor/MatMul/ReadVariableOp¢Critic/BiasAdd/ReadVariableOp¢Critic/MatMul/ReadVariableOp¢Dense1/BiasAdd/ReadVariableOp¢Dense1/MatMul/ReadVariableOp£
Dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
Dense1/MatMul/ReadVariableOp
Dense1/MatMulMatMulinputs$Dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dense1/MatMul¢
Dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
Dense1/BiasAdd/ReadVariableOp
Dense1/BiasAddBiasAddDense1/MatMul:product:0%Dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dense1/BiasAddn
Dense1/ReluReluDense1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dense1/Relu£
Critic/MatMul/ReadVariableOpReadVariableOp%critic_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
Critic/MatMul/ReadVariableOp
Critic/MatMulMatMulDense1/Relu:activations:0$Critic/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Critic/MatMul¡
Critic/BiasAdd/ReadVariableOpReadVariableOp&critic_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
Critic/BiasAdd/ReadVariableOp
Critic/BiasAddBiasAddCritic/MatMul:product:0%Critic/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Critic/BiasAdd 
Actor/MatMul/ReadVariableOpReadVariableOp$actor_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
Actor/MatMul/ReadVariableOp
Actor/MatMulMatMulDense1/Relu:activations:0#Actor/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Actor/MatMul
Actor/BiasAdd/ReadVariableOpReadVariableOp%actor_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
Actor/BiasAdd/ReadVariableOp
Actor/BiasAddBiasAddActor/MatMul:product:0$Actor/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Actor/BiasAddq
IdentityIdentityActor/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityv

Identity_1IdentityCritic/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1
NoOpNoOp^Actor/BiasAdd/ReadVariableOp^Actor/MatMul/ReadVariableOp^Critic/BiasAdd/ReadVariableOp^Critic/MatMul/ReadVariableOp^Dense1/BiasAdd/ReadVariableOp^Dense1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2<
Actor/BiasAdd/ReadVariableOpActor/BiasAdd/ReadVariableOp2:
Actor/MatMul/ReadVariableOpActor/MatMul/ReadVariableOp2>
Critic/BiasAdd/ReadVariableOpCritic/BiasAdd/ReadVariableOp2<
Critic/MatMul/ReadVariableOpCritic/MatMul/ReadVariableOp2>
Dense1/BiasAdd/ReadVariableOpDense1/BiasAdd/ReadVariableOp2<
Dense1/MatMul/ReadVariableOpDense1/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«

÷
E__inference_Actor_layer_call_and_return_conditional_losses_1018469900

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â
ã
I__inference_a2c_model_layer_call_and_return_conditional_losses_1018469410
x#
model_1018469394:	
model_1018469396:	#
model_1018469398:	
model_1018469400:#
model_1018469402:	
model_1018469404:
identity

identity_1¢model/StatefulPartitionedCallb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dimt

ExpandDims
ExpandDimsxExpandDims/dim:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

ExpandDimsó
model/StatefulPartitionedCallStatefulPartitionedCallExpandDims:output:0model_1018469394model_1018469396model_1018469398model_1018469400model_1018469402model_1018469404*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
::*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_layer_call_and_return_conditional_losses_10184693512
model/StatefulPartitionedCallx
IdentityIdentity&model/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identity|

Identity_1Identity&model/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes

:2

Identity_1n
NoOpNoOp^model/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall:F B
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
ð
þ
E__inference_model_layer_call_and_return_conditional_losses_1018469272

inputs8
%dense1_matmul_readvariableop_resource:	5
&dense1_biasadd_readvariableop_resource:	8
%critic_matmul_readvariableop_resource:	4
&critic_biasadd_readvariableop_resource:7
$actor_matmul_readvariableop_resource:	3
%actor_biasadd_readvariableop_resource:
identity

identity_1¢Actor/BiasAdd/ReadVariableOp¢Actor/MatMul/ReadVariableOp¢Critic/BiasAdd/ReadVariableOp¢Critic/MatMul/ReadVariableOp¢Dense1/BiasAdd/ReadVariableOp¢Dense1/MatMul/ReadVariableOp£
Dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
Dense1/MatMul/ReadVariableOp
Dense1/MatMulMatMulinputs$Dense1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
Dense1/MatMul¢
Dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
Dense1/BiasAdd/ReadVariableOp
Dense1/BiasAddBiasAddDense1/MatMul:product:0%Dense1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
Dense1/BiasAdde
Dense1/ReluReluDense1/BiasAdd:output:0*
T0*
_output_shapes
:	2
Dense1/Relu£
Critic/MatMul/ReadVariableOpReadVariableOp%critic_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
Critic/MatMul/ReadVariableOp
Critic/MatMulMatMulDense1/Relu:activations:0$Critic/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Critic/MatMul¡
Critic/BiasAdd/ReadVariableOpReadVariableOp&critic_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
Critic/BiasAdd/ReadVariableOp
Critic/BiasAddBiasAddCritic/MatMul:product:0%Critic/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Critic/BiasAdd 
Actor/MatMul/ReadVariableOpReadVariableOp$actor_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
Actor/MatMul/ReadVariableOp
Actor/MatMulMatMulDense1/Relu:activations:0#Actor/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Actor/MatMul
Actor/BiasAdd/ReadVariableOpReadVariableOp%actor_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
Actor/BiasAdd/ReadVariableOp
Actor/BiasAddBiasAddActor/MatMul:product:0$Actor/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Actor/BiasAddh
IdentityIdentityActor/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:2

Identitym

Identity_1IdentityCritic/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:2

Identity_1
NoOpNoOp^Actor/BiasAdd/ReadVariableOp^Actor/MatMul/ReadVariableOp^Critic/BiasAdd/ReadVariableOp^Critic/MatMul/ReadVariableOp^Dense1/BiasAdd/ReadVariableOp^Dense1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:: : : : : : 2<
Actor/BiasAdd/ReadVariableOpActor/BiasAdd/ReadVariableOp2:
Actor/MatMul/ReadVariableOpActor/MatMul/ReadVariableOp2>
Critic/BiasAdd/ReadVariableOpCritic/BiasAdd/ReadVariableOp2<
Critic/MatMul/ReadVariableOpCritic/MatMul/ReadVariableOp2>
Dense1/BiasAdd/ReadVariableOpDense1/BiasAdd/ReadVariableOp2<
Dense1/MatMul/ReadVariableOpDense1/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê


*__inference_model_layer_call_fn_1018469708

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:
	unknown_3:	
	unknown_4:
identity

identity_1¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_layer_call_and_return_conditional_losses_10184690762
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
û

+__inference_Critic_layer_call_fn_1018469909

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_Critic_layer_call_and_return_conditional_losses_10184690522
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs



(__inference_signature_wrapper_1018469509
input_1
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:
	unknown_3:	
	unknown_4:
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
::*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference__wrapped_model_10184690182
StatefulPartitionedCallr
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identityv

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes

:2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
·


.__inference_a2c_model_layer_call_fn_1018469547
x
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:
	unknown_3:	
	unknown_4:
identity

identity_1¢StatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
::*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_a2c_model_layer_call_and_return_conditional_losses_10184692892
StatefulPartitionedCallr
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identityv

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes

:2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:F B
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
ü
î
&__inference__traced_restore_1018469989
file_prefix1
assignvariableop_dense1_kernel:	-
assignvariableop_1_dense1_bias:	2
assignvariableop_2_actor_kernel:	+
assignvariableop_3_actor_bias:3
 assignvariableop_4_critic_kernel:	,
assignvariableop_5_critic_bias:

identity_7¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5Ó
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ß
valueÕBÒB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
RestoreV2/shape_and_slicesÎ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1£
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¤
AssignVariableOp_2AssignVariableOpassignvariableop_2_actor_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¢
AssignVariableOp_3AssignVariableOpassignvariableop_3_actor_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¥
AssignVariableOp_4AssignVariableOp assignvariableop_4_critic_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5£
AssignVariableOp_5AssignVariableOpassignvariableop_5_critic_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpä

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_6c

Identity_7IdentityIdentity_6:output:0^NoOp_1*
T0*
_output_shapes
: 2

Identity_7Î
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"!

identity_7Identity_7:output:0*!
_input_shapes
: : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_5:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
í


*__inference_model_layer_call_fn_1018469093
input_1
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:
	unknown_3:	
	unknown_4:
identity

identity_1¢StatefulPartitionedCallÁ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_layer_call_and_return_conditional_losses_10184690762
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
¬

ø
F__inference_Critic_layer_call_and_return_conditional_losses_1018469052

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
"
Å
I__inference_a2c_model_layer_call_and_return_conditional_losses_1018469637
x>
+model_dense1_matmul_readvariableop_resource:	;
,model_dense1_biasadd_readvariableop_resource:	>
+model_critic_matmul_readvariableop_resource:	:
,model_critic_biasadd_readvariableop_resource:=
*model_actor_matmul_readvariableop_resource:	9
+model_actor_biasadd_readvariableop_resource:
identity

identity_1¢"model/Actor/BiasAdd/ReadVariableOp¢!model/Actor/MatMul/ReadVariableOp¢#model/Critic/BiasAdd/ReadVariableOp¢"model/Critic/MatMul/ReadVariableOp¢#model/Dense1/BiasAdd/ReadVariableOp¢"model/Dense1/MatMul/ReadVariableOpb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dimk

ExpandDims
ExpandDimsxExpandDims/dim:output:0*
T0*
_output_shapes

:2

ExpandDimsµ
"model/Dense1/MatMul/ReadVariableOpReadVariableOp+model_dense1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02$
"model/Dense1/MatMul/ReadVariableOp
model/Dense1/MatMulMatMulExpandDims:output:0*model/Dense1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
model/Dense1/MatMul´
#model/Dense1/BiasAdd/ReadVariableOpReadVariableOp,model_dense1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#model/Dense1/BiasAdd/ReadVariableOp­
model/Dense1/BiasAddBiasAddmodel/Dense1/MatMul:product:0+model/Dense1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
model/Dense1/BiasAddw
model/Dense1/ReluRelumodel/Dense1/BiasAdd:output:0*
T0*
_output_shapes
:	2
model/Dense1/Reluµ
"model/Critic/MatMul/ReadVariableOpReadVariableOp+model_critic_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02$
"model/Critic/MatMul/ReadVariableOpª
model/Critic/MatMulMatMulmodel/Dense1/Relu:activations:0*model/Critic/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
model/Critic/MatMul³
#model/Critic/BiasAdd/ReadVariableOpReadVariableOp,model_critic_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#model/Critic/BiasAdd/ReadVariableOp¬
model/Critic/BiasAddBiasAddmodel/Critic/MatMul:product:0+model/Critic/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
model/Critic/BiasAdd²
!model/Actor/MatMul/ReadVariableOpReadVariableOp*model_actor_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02#
!model/Actor/MatMul/ReadVariableOp§
model/Actor/MatMulMatMulmodel/Dense1/Relu:activations:0)model/Actor/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
model/Actor/MatMul°
"model/Actor/BiasAdd/ReadVariableOpReadVariableOp+model_actor_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"model/Actor/BiasAdd/ReadVariableOp¨
model/Actor/BiasAddBiasAddmodel/Actor/MatMul:product:0*model/Actor/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
model/Actor/BiasAddn
IdentityIdentitymodel/Actor/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:2

Identitys

Identity_1Identitymodel/Critic/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:2

Identity_1­
NoOpNoOp#^model/Actor/BiasAdd/ReadVariableOp"^model/Actor/MatMul/ReadVariableOp$^model/Critic/BiasAdd/ReadVariableOp#^model/Critic/MatMul/ReadVariableOp$^model/Dense1/BiasAdd/ReadVariableOp#^model/Dense1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2H
"model/Actor/BiasAdd/ReadVariableOp"model/Actor/BiasAdd/ReadVariableOp2F
!model/Actor/MatMul/ReadVariableOp!model/Actor/MatMul/ReadVariableOp2J
#model/Critic/BiasAdd/ReadVariableOp#model/Critic/BiasAdd/ReadVariableOp2H
"model/Critic/MatMul/ReadVariableOp"model/Critic/MatMul/ReadVariableOp2J
#model/Dense1/BiasAdd/ReadVariableOp#model/Dense1/BiasAdd/ReadVariableOp2H
"model/Dense1/MatMul/ReadVariableOp"model/Dense1/MatMul/ReadVariableOp:F B
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
Æ


*__inference_model_layer_call_fn_1018469765

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:
	unknown_3:	
	unknown_4:
identity

identity_1¢StatefulPartitionedCall®
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
::*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_layer_call_and_return_conditional_losses_10184693512
StatefulPartitionedCallr
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identityv

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes

:2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð
þ
E__inference_model_layer_call_and_return_conditional_losses_1018469861

inputs8
%dense1_matmul_readvariableop_resource:	5
&dense1_biasadd_readvariableop_resource:	8
%critic_matmul_readvariableop_resource:	4
&critic_biasadd_readvariableop_resource:7
$actor_matmul_readvariableop_resource:	3
%actor_biasadd_readvariableop_resource:
identity

identity_1¢Actor/BiasAdd/ReadVariableOp¢Actor/MatMul/ReadVariableOp¢Critic/BiasAdd/ReadVariableOp¢Critic/MatMul/ReadVariableOp¢Dense1/BiasAdd/ReadVariableOp¢Dense1/MatMul/ReadVariableOp£
Dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
Dense1/MatMul/ReadVariableOp
Dense1/MatMulMatMulinputs$Dense1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
Dense1/MatMul¢
Dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
Dense1/BiasAdd/ReadVariableOp
Dense1/BiasAddBiasAddDense1/MatMul:product:0%Dense1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
Dense1/BiasAdde
Dense1/ReluReluDense1/BiasAdd:output:0*
T0*
_output_shapes
:	2
Dense1/Relu£
Critic/MatMul/ReadVariableOpReadVariableOp%critic_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
Critic/MatMul/ReadVariableOp
Critic/MatMulMatMulDense1/Relu:activations:0$Critic/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Critic/MatMul¡
Critic/BiasAdd/ReadVariableOpReadVariableOp&critic_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
Critic/BiasAdd/ReadVariableOp
Critic/BiasAddBiasAddCritic/MatMul:product:0%Critic/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Critic/BiasAdd 
Actor/MatMul/ReadVariableOpReadVariableOp$actor_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
Actor/MatMul/ReadVariableOp
Actor/MatMulMatMulDense1/Relu:activations:0#Actor/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Actor/MatMul
Actor/BiasAdd/ReadVariableOpReadVariableOp%actor_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
Actor/BiasAdd/ReadVariableOp
Actor/BiasAddBiasAddActor/MatMul:product:0$Actor/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Actor/BiasAddh
IdentityIdentityActor/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:2

Identitym

Identity_1IdentityCritic/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:2

Identity_1
NoOpNoOp^Actor/BiasAdd/ReadVariableOp^Actor/MatMul/ReadVariableOp^Critic/BiasAdd/ReadVariableOp^Critic/MatMul/ReadVariableOp^Dense1/BiasAdd/ReadVariableOp^Dense1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:: : : : : : 2<
Actor/BiasAdd/ReadVariableOpActor/BiasAdd/ReadVariableOp2:
Actor/MatMul/ReadVariableOpActor/MatMul/ReadVariableOp2>
Critic/BiasAdd/ReadVariableOpCritic/BiasAdd/ReadVariableOp2<
Critic/MatMul/ReadVariableOpCritic/MatMul/ReadVariableOp2>
Dense1/BiasAdd/ReadVariableOpDense1/BiasAdd/ReadVariableOp2<
Dense1/MatMul/ReadVariableOpDense1/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´
ª
E__inference_model_layer_call_and_return_conditional_losses_1018469164

inputs$
dense1_1018469147:	 
dense1_1018469149:	$
critic_1018469152:	
critic_1018469154:#
actor_1018469157:	
actor_1018469159:
identity

identity_1¢Actor/StatefulPartitionedCall¢Critic/StatefulPartitionedCall¢Dense1/StatefulPartitionedCall
Dense1/StatefulPartitionedCallStatefulPartitionedCallinputsdense1_1018469147dense1_1018469149*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_Dense1_layer_call_and_return_conditional_losses_10184690362 
Dense1/StatefulPartitionedCallº
Critic/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0critic_1018469152critic_1018469154*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_Critic_layer_call_and_return_conditional_losses_10184690522 
Critic/StatefulPartitionedCallµ
Actor/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0actor_1018469157actor_1018469159*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_Actor_layer_call_and_return_conditional_losses_10184690682
Actor/StatefulPartitionedCall
IdentityIdentity&Actor/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity'Critic/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1°
NoOpNoOp^Actor/StatefulPartitionedCall^Critic/StatefulPartitionedCall^Dense1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2>
Actor/StatefulPartitionedCallActor/StatefulPartitionedCall2@
Critic/StatefulPartitionedCallCritic/StatefulPartitionedCall2@
Dense1/StatefulPartitionedCallDense1/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ


*__inference_model_layer_call_fn_1018469746

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:
	unknown_3:	
	unknown_4:
identity

identity_1¢StatefulPartitionedCall®
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
::*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_layer_call_and_return_conditional_losses_10184692722
StatefulPartitionedCallr
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identityv

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes

:2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É


.__inference_a2c_model_layer_call_fn_1018469585
input_1
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:
	unknown_3:	
	unknown_4:
identity

identity_1¢StatefulPartitionedCall³
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
::*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_a2c_model_layer_call_and_return_conditional_losses_10184694102
StatefulPartitionedCallr
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identityv

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes

:2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1

ù
F__inference_Dense1_layer_call_and_return_conditional_losses_1018469881

inputs1
matmul_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"¨L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ó
serving_default¿
7
input_1,
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ3
output_1'
StatefulPartitionedCall:03
output_2'
StatefulPartitionedCall:1tensorflow/serving/predict:Ñg
Ú
nn
trainable_variables
	variables
regularization_losses
	keras_api

signatures
:__call__
;_default_save_signature
*<&call_and_return_all_conditional_losses"
_tf_keras_model
©
layer-0
layer_with_weights-0
layer-1
	layer_with_weights-1
	layer-2

layer_with_weights-2

layer-3
trainable_variables
	variables
regularization_losses
	keras_api
=__call__
*>&call_and_return_all_conditional_losses"
_tf_keras_network
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
trainable_variables
non_trainable_variables
	variables
layer_metrics
metrics

layers
regularization_losses
layer_regularization_losses
:__call__
;_default_save_signature
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
,
?serving_default"
signature_map
"
_tf_keras_input_layer
»

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
@__call__
*A&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
trainable_variables
	variables
 regularization_losses
!	keras_api
B__call__
*C&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
"trainable_variables
#	variables
$regularization_losses
%	keras_api
D__call__
*E&call_and_return_all_conditional_losses"
_tf_keras_layer
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
­
trainable_variables
&non_trainable_variables
	variables
'layer_metrics
(metrics

)layers
regularization_losses
*layer_regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
 :	2Dense1/kernel
:2Dense1/bias
:	2Actor/kernel
:2
Actor/bias
 :	2Critic/kernel
:2Critic/bias
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
trainable_variables
+non_trainable_variables
	variables
,layer_metrics
-metrics

.layers
regularization_losses
/layer_regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
trainable_variables
0non_trainable_variables
	variables
1layer_metrics
2metrics

3layers
 regularization_losses
4layer_regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
"trainable_variables
5non_trainable_variables
#	variables
6layer_metrics
7metrics

8layers
$regularization_losses
9layer_regularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
<
0
1
	2

3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ô2ñ
.__inference_a2c_model_layer_call_fn_1018469528
.__inference_a2c_model_layer_call_fn_1018469547
.__inference_a2c_model_layer_call_fn_1018469566
.__inference_a2c_model_layer_call_fn_1018469585®
¥²¡
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÐBÍ
%__inference__wrapped_model_1018469018input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
à2Ý
I__inference_a2c_model_layer_call_and_return_conditional_losses_1018469611
I__inference_a2c_model_layer_call_and_return_conditional_losses_1018469637
I__inference_a2c_model_layer_call_and_return_conditional_losses_1018469663
I__inference_a2c_model_layer_call_and_return_conditional_losses_1018469689®
¥²¡
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Î2Ë
*__inference_model_layer_call_fn_1018469093
*__inference_model_layer_call_fn_1018469708
*__inference_model_layer_call_fn_1018469727
*__inference_model_layer_call_fn_1018469200
*__inference_model_layer_call_fn_1018469746
*__inference_model_layer_call_fn_1018469765À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ð2í
E__inference_model_layer_call_and_return_conditional_losses_1018469789
E__inference_model_layer_call_and_return_conditional_losses_1018469813
E__inference_model_layer_call_and_return_conditional_losses_1018469220
E__inference_model_layer_call_and_return_conditional_losses_1018469240
E__inference_model_layer_call_and_return_conditional_losses_1018469837
E__inference_model_layer_call_and_return_conditional_losses_1018469861À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÏBÌ
(__inference_signature_wrapper_1018469509input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_Dense1_layer_call_fn_1018469870¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_Dense1_layer_call_and_return_conditional_losses_1018469881¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_Actor_layer_call_fn_1018469890¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_Actor_layer_call_and_return_conditional_losses_1018469900¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_Critic_layer_call_fn_1018469909¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_Critic_layer_call_and_return_conditional_losses_1018469919¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ¦
E__inference_Actor_layer_call_and_return_conditional_losses_1018469900]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
*__inference_Actor_layer_call_fn_1018469890P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
F__inference_Critic_layer_call_and_return_conditional_losses_1018469919]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_Critic_layer_call_fn_1018469909P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
F__inference_Dense1_layer_call_and_return_conditional_losses_1018469881]/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_Dense1_layer_call_fn_1018469870P/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ³
%__inference__wrapped_model_1018469018,¢)
"¢

input_1ÿÿÿÿÿÿÿÿÿ
ª "QªN
%
output_1
output_1
%
output_2
output_2¼
I__inference_a2c_model_layer_call_and_return_conditional_losses_1018469611o*¢'
 ¢

xÿÿÿÿÿÿÿÿÿ
p 
ª "9¢6
/,

0/0

0/1
 ¼
I__inference_a2c_model_layer_call_and_return_conditional_losses_1018469637o*¢'
 ¢

xÿÿÿÿÿÿÿÿÿ
p
ª "9¢6
/,

0/0

0/1
 Â
I__inference_a2c_model_layer_call_and_return_conditional_losses_1018469663u0¢-
&¢#

input_1ÿÿÿÿÿÿÿÿÿ
p 
ª "9¢6
/,

0/0

0/1
 Â
I__inference_a2c_model_layer_call_and_return_conditional_losses_1018469689u0¢-
&¢#

input_1ÿÿÿÿÿÿÿÿÿ
p
ª "9¢6
/,

0/0

0/1
 
.__inference_a2c_model_layer_call_fn_1018469528g0¢-
&¢#

input_1ÿÿÿÿÿÿÿÿÿ
p 
ª "+(

0

1
.__inference_a2c_model_layer_call_fn_1018469547a*¢'
 ¢

xÿÿÿÿÿÿÿÿÿ
p 
ª "+(

0

1
.__inference_a2c_model_layer_call_fn_1018469566a*¢'
 ¢

xÿÿÿÿÿÿÿÿÿ
p
ª "+(

0

1
.__inference_a2c_model_layer_call_fn_1018469585g0¢-
&¢#

input_1ÿÿÿÿÿÿÿÿÿ
p
ª "+(

0

1Ù
E__inference_model_layer_call_and_return_conditional_losses_10184692208¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "K¢H
A>

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ
 Ù
E__inference_model_layer_call_and_return_conditional_losses_10184692408¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "K¢H
A>

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ
 Ø
E__inference_model_layer_call_and_return_conditional_losses_10184697897¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "K¢H
A>

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ
 Ø
E__inference_model_layer_call_and_return_conditional_losses_10184698137¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "K¢H
A>

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ
 Å
E__inference_model_layer_call_and_return_conditional_losses_1018469837|7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "9¢6
/,

0/0

0/1
 Å
E__inference_model_layer_call_and_return_conditional_losses_1018469861|7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "9¢6
/,

0/0

0/1
 °
*__inference_model_layer_call_fn_10184690938¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "=:

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ°
*__inference_model_layer_call_fn_10184692008¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "=:

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ¯
*__inference_model_layer_call_fn_10184697087¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "=:

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ¯
*__inference_model_layer_call_fn_10184697277¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "=:

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ
*__inference_model_layer_call_fn_1018469746n7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "+(

0

1
*__inference_model_layer_call_fn_1018469765n7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "+(

0

1Á
(__inference_signature_wrapper_10184695097¢4
¢ 
-ª*
(
input_1
input_1ÿÿÿÿÿÿÿÿÿ"QªN
%
output_1
output_1
%
output_2
output_2