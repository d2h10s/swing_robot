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
(__inference_signature_wrapper_1170346438
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
#__inference__traced_save_1170346890
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
&__inference__traced_restore_1170346918ÍÔ
Ê
þ
E__inference_model_layer_call_and_return_conditional_losses_1170346742

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
í


*__inference_model_layer_call_fn_1170346022
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
E__inference_model_layer_call_and_return_conditional_losses_11703460052
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
µ(

%__inference__wrapped_model_1170345947
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
ð
þ
E__inference_model_layer_call_and_return_conditional_losses_1170346766

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
«

÷
E__inference_Actor_layer_call_and_return_conditional_losses_1170345997

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
´
ª
E__inference_model_layer_call_and_return_conditional_losses_1170346093

inputs$
dense1_1170346076:	 
dense1_1170346078:	$
critic_1170346081:	
critic_1170346083:#
actor_1170346086:	
actor_1170346088:
identity

identity_1¢Actor/StatefulPartitionedCall¢Critic/StatefulPartitionedCall¢Dense1/StatefulPartitionedCall
Dense1/StatefulPartitionedCallStatefulPartitionedCallinputsdense1_1170346076dense1_1170346078*
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
F__inference_Dense1_layer_call_and_return_conditional_losses_11703459652 
Dense1/StatefulPartitionedCallº
Critic/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0critic_1170346081critic_1170346083*
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
F__inference_Critic_layer_call_and_return_conditional_losses_11703459812 
Critic/StatefulPartitionedCallµ
Actor/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0actor_1170346086actor_1170346088*
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
E__inference_Actor_layer_call_and_return_conditional_losses_11703459972
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
¬

ø
F__inference_Critic_layer_call_and_return_conditional_losses_1170346848

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
´
ª
E__inference_model_layer_call_and_return_conditional_losses_1170346005

inputs$
dense1_1170345966:	 
dense1_1170345968:	$
critic_1170345982:	
critic_1170345984:#
actor_1170345998:	
actor_1170346000:
identity

identity_1¢Actor/StatefulPartitionedCall¢Critic/StatefulPartitionedCall¢Dense1/StatefulPartitionedCall
Dense1/StatefulPartitionedCallStatefulPartitionedCallinputsdense1_1170345966dense1_1170345968*
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
F__inference_Dense1_layer_call_and_return_conditional_losses_11703459652 
Dense1/StatefulPartitionedCallº
Critic/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0critic_1170345982critic_1170345984*
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
F__inference_Critic_layer_call_and_return_conditional_losses_11703459812 
Critic/StatefulPartitionedCallµ
Actor/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0actor_1170345998actor_1170346000*
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
E__inference_Actor_layer_call_and_return_conditional_losses_11703459972
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
ü
î
&__inference__traced_restore_1170346918
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
ê


*__inference_model_layer_call_fn_1170346637

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
E__inference_model_layer_call_and_return_conditional_losses_11703460052
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
«

÷
E__inference_Actor_layer_call_and_return_conditional_losses_1170346829

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
I__inference_a2c_model_layer_call_and_return_conditional_losses_1170346339
x#
model_1170346323:	
model_1170346325:	#
model_1170346327:	
model_1170346329:#
model_1170346331:	
model_1170346333:
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
model/StatefulPartitionedCallStatefulPartitionedCallExpandDims:output:0model_1170346323model_1170346325model_1170346327model_1170346329model_1170346331model_1170346333*
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
E__inference_model_layer_call_and_return_conditional_losses_11703462802
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
·
«
E__inference_model_layer_call_and_return_conditional_losses_1170346169
input_1$
dense1_1170346152:	 
dense1_1170346154:	$
critic_1170346157:	
critic_1170346159:#
actor_1170346162:	
actor_1170346164:
identity

identity_1¢Actor/StatefulPartitionedCall¢Critic/StatefulPartitionedCall¢Dense1/StatefulPartitionedCall
Dense1/StatefulPartitionedCallStatefulPartitionedCallinput_1dense1_1170346152dense1_1170346154*
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
F__inference_Dense1_layer_call_and_return_conditional_losses_11703459652 
Dense1/StatefulPartitionedCallº
Critic/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0critic_1170346157critic_1170346159*
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
F__inference_Critic_layer_call_and_return_conditional_losses_11703459812 
Critic/StatefulPartitionedCallµ
Actor/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0actor_1170346162actor_1170346164*
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
E__inference_Actor_layer_call_and_return_conditional_losses_11703459972
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
.__inference_a2c_model_layer_call_fn_1170346457
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
I__inference_a2c_model_layer_call_and_return_conditional_losses_11703462182
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
"
Å
I__inference_a2c_model_layer_call_and_return_conditional_losses_1170346566
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

ù
F__inference_Dense1_layer_call_and_return_conditional_losses_1170346810

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
£"
Ë
I__inference_a2c_model_layer_call_and_return_conditional_losses_1170346618
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
Â
ã
I__inference_a2c_model_layer_call_and_return_conditional_losses_1170346218
x#
model_1170346202:	
model_1170346204:	#
model_1170346206:	
model_1170346208:#
model_1170346210:	
model_1170346212:
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
model/StatefulPartitionedCallStatefulPartitionedCallExpandDims:output:0model_1170346202model_1170346204model_1170346206model_1170346208model_1170346210model_1170346212*
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
E__inference_model_layer_call_and_return_conditional_losses_11703462012
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

ü
#__inference__traced_save_1170346890
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

ù
F__inference_Dense1_layer_call_and_return_conditional_losses_1170345965

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
Ê
þ
E__inference_model_layer_call_and_return_conditional_losses_1170346718

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
ð
þ
E__inference_model_layer_call_and_return_conditional_losses_1170346280

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
ð
þ
E__inference_model_layer_call_and_return_conditional_losses_1170346790

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
û

+__inference_Critic_layer_call_fn_1170346838

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
F__inference_Critic_layer_call_and_return_conditional_losses_11703459812
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
(__inference_signature_wrapper_1170346438
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
%__inference__wrapped_model_11703459472
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
ù

*__inference_Actor_layer_call_fn_1170346819

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
E__inference_Actor_layer_call_and_return_conditional_losses_11703459972
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
.__inference_a2c_model_layer_call_fn_1170346495
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
I__inference_a2c_model_layer_call_and_return_conditional_losses_11703463392
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
í


*__inference_model_layer_call_fn_1170346129
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
E__inference_model_layer_call_and_return_conditional_losses_11703460932
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
"
Å
I__inference_a2c_model_layer_call_and_return_conditional_losses_1170346540
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
*__inference_model_layer_call_fn_1170346675

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
E__inference_model_layer_call_and_return_conditional_losses_11703462012
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
¬

ø
F__inference_Critic_layer_call_and_return_conditional_losses_1170345981

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
Æ


*__inference_model_layer_call_fn_1170346694

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
E__inference_model_layer_call_and_return_conditional_losses_11703462802
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
£"
Ë
I__inference_a2c_model_layer_call_and_return_conditional_losses_1170346592
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
ê


*__inference_model_layer_call_fn_1170346656

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
E__inference_model_layer_call_and_return_conditional_losses_11703460932
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
ð
þ
E__inference_model_layer_call_and_return_conditional_losses_1170346201

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
·


.__inference_a2c_model_layer_call_fn_1170346476
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
I__inference_a2c_model_layer_call_and_return_conditional_losses_11703462182
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
É


.__inference_a2c_model_layer_call_fn_1170346514
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
I__inference_a2c_model_layer_call_and_return_conditional_losses_11703463392
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
·
«
E__inference_model_layer_call_and_return_conditional_losses_1170346149
input_1$
dense1_1170346132:	 
dense1_1170346134:	$
critic_1170346137:	
critic_1170346139:#
actor_1170346142:	
actor_1170346144:
identity

identity_1¢Actor/StatefulPartitionedCall¢Critic/StatefulPartitionedCall¢Dense1/StatefulPartitionedCall
Dense1/StatefulPartitionedCallStatefulPartitionedCallinput_1dense1_1170346132dense1_1170346134*
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
F__inference_Dense1_layer_call_and_return_conditional_losses_11703459652 
Dense1/StatefulPartitionedCallº
Critic/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0critic_1170346137critic_1170346139*
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
F__inference_Critic_layer_call_and_return_conditional_losses_11703459812 
Critic/StatefulPartitionedCallµ
Actor/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0actor_1170346142actor_1170346144*
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
E__inference_Actor_layer_call_and_return_conditional_losses_11703459972
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
ü

+__inference_Dense1_layer_call_fn_1170346799

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
F__inference_Dense1_layer_call_and_return_conditional_losses_11703459652
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
.__inference_a2c_model_layer_call_fn_1170346457
.__inference_a2c_model_layer_call_fn_1170346476
.__inference_a2c_model_layer_call_fn_1170346495
.__inference_a2c_model_layer_call_fn_1170346514®
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
%__inference__wrapped_model_1170345947input_1"
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
I__inference_a2c_model_layer_call_and_return_conditional_losses_1170346540
I__inference_a2c_model_layer_call_and_return_conditional_losses_1170346566
I__inference_a2c_model_layer_call_and_return_conditional_losses_1170346592
I__inference_a2c_model_layer_call_and_return_conditional_losses_1170346618®
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
*__inference_model_layer_call_fn_1170346022
*__inference_model_layer_call_fn_1170346637
*__inference_model_layer_call_fn_1170346656
*__inference_model_layer_call_fn_1170346129
*__inference_model_layer_call_fn_1170346675
*__inference_model_layer_call_fn_1170346694À
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
E__inference_model_layer_call_and_return_conditional_losses_1170346718
E__inference_model_layer_call_and_return_conditional_losses_1170346742
E__inference_model_layer_call_and_return_conditional_losses_1170346149
E__inference_model_layer_call_and_return_conditional_losses_1170346169
E__inference_model_layer_call_and_return_conditional_losses_1170346766
E__inference_model_layer_call_and_return_conditional_losses_1170346790À
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
(__inference_signature_wrapper_1170346438input_1"
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
+__inference_Dense1_layer_call_fn_1170346799¢
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
F__inference_Dense1_layer_call_and_return_conditional_losses_1170346810¢
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
*__inference_Actor_layer_call_fn_1170346819¢
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
E__inference_Actor_layer_call_and_return_conditional_losses_1170346829¢
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
+__inference_Critic_layer_call_fn_1170346838¢
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
F__inference_Critic_layer_call_and_return_conditional_losses_1170346848¢
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
E__inference_Actor_layer_call_and_return_conditional_losses_1170346829]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
*__inference_Actor_layer_call_fn_1170346819P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
F__inference_Critic_layer_call_and_return_conditional_losses_1170346848]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_Critic_layer_call_fn_1170346838P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
F__inference_Dense1_layer_call_and_return_conditional_losses_1170346810]/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_Dense1_layer_call_fn_1170346799P/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ³
%__inference__wrapped_model_1170345947,¢)
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
I__inference_a2c_model_layer_call_and_return_conditional_losses_1170346540o*¢'
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
I__inference_a2c_model_layer_call_and_return_conditional_losses_1170346566o*¢'
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
I__inference_a2c_model_layer_call_and_return_conditional_losses_1170346592u0¢-
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
I__inference_a2c_model_layer_call_and_return_conditional_losses_1170346618u0¢-
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
.__inference_a2c_model_layer_call_fn_1170346457g0¢-
&¢#

input_1ÿÿÿÿÿÿÿÿÿ
p 
ª "+(

0

1
.__inference_a2c_model_layer_call_fn_1170346476a*¢'
 ¢

xÿÿÿÿÿÿÿÿÿ
p 
ª "+(

0

1
.__inference_a2c_model_layer_call_fn_1170346495a*¢'
 ¢

xÿÿÿÿÿÿÿÿÿ
p
ª "+(

0

1
.__inference_a2c_model_layer_call_fn_1170346514g0¢-
&¢#

input_1ÿÿÿÿÿÿÿÿÿ
p
ª "+(

0

1Ù
E__inference_model_layer_call_and_return_conditional_losses_11703461498¢5
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
E__inference_model_layer_call_and_return_conditional_losses_11703461698¢5
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
E__inference_model_layer_call_and_return_conditional_losses_11703467187¢4
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
E__inference_model_layer_call_and_return_conditional_losses_11703467427¢4
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
E__inference_model_layer_call_and_return_conditional_losses_1170346766|7¢4
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
E__inference_model_layer_call_and_return_conditional_losses_1170346790|7¢4
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
*__inference_model_layer_call_fn_11703460228¢5
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
*__inference_model_layer_call_fn_11703461298¢5
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
*__inference_model_layer_call_fn_11703466377¢4
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
*__inference_model_layer_call_fn_11703466567¢4
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
*__inference_model_layer_call_fn_1170346675n7¢4
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
*__inference_model_layer_call_fn_1170346694n7¢4
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
(__inference_signature_wrapper_11703464387¢4
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