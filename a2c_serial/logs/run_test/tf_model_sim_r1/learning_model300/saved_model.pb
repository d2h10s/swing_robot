¾£
¿£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02unknown8ý
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
	variables
non_trainable_variables
layer_regularization_losses

layers
layer_metrics
metrics
regularization_losses
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
	variables
&non_trainable_variables
'layer_regularization_losses

(layers
)layer_metrics
*metrics
regularization_losses
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

0
 
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
	variables
+non_trainable_variables
,layer_regularization_losses

-layers
.layer_metrics
/metrics
regularization_losses

0
1

0
1
 
­
trainable_variables
	variables
0non_trainable_variables
1layer_regularization_losses

2layers
3layer_metrics
4metrics
 regularization_losses

0
1

0
1
 
­
"trainable_variables
#	variables
5non_trainable_variables
6layer_regularization_losses

7layers
8layer_metrics
9metrics
$regularization_losses
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
 
r
serving_default_input_1Placeholder*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ

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

*-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_signature_wrapper_26795441
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
í
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
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_save_26795893
ð
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
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference__traced_restore_26795921±Ó
Þ
~
)__inference_Critic_layer_call_fn_26795851

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallô
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
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_Critic_layer_call_and_return_conditional_losses_267950882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·
Â
G__inference_a2c_model_layer_call_and_return_conditional_losses_26795557
x6
2functional_1_dense1_matmul_readvariableop_resource7
3functional_1_dense1_biasadd_readvariableop_resource6
2functional_1_critic_matmul_readvariableop_resource7
3functional_1_critic_biasadd_readvariableop_resource5
1functional_1_actor_matmul_readvariableop_resource6
2functional_1_actor_biasadd_readvariableop_resource
identity

identity_1b
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

ExpandDimsÊ
)functional_1/Dense1/MatMul/ReadVariableOpReadVariableOp2functional_1_dense1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02+
)functional_1/Dense1/MatMul/ReadVariableOp´
functional_1/Dense1/MatMulMatMulExpandDims:output:01functional_1/Dense1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
functional_1/Dense1/MatMulÉ
*functional_1/Dense1/BiasAdd/ReadVariableOpReadVariableOp3functional_1_dense1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*functional_1/Dense1/BiasAdd/ReadVariableOpÉ
functional_1/Dense1/BiasAddBiasAdd$functional_1/Dense1/MatMul:product:02functional_1/Dense1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
functional_1/Dense1/BiasAdd
functional_1/Dense1/ReluRelu$functional_1/Dense1/BiasAdd:output:0*
T0*
_output_shapes
:	2
functional_1/Dense1/ReluÊ
)functional_1/Critic/MatMul/ReadVariableOpReadVariableOp2functional_1_critic_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02+
)functional_1/Critic/MatMul/ReadVariableOpÆ
functional_1/Critic/MatMulMatMul&functional_1/Dense1/Relu:activations:01functional_1/Critic/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Critic/MatMulÈ
*functional_1/Critic/BiasAdd/ReadVariableOpReadVariableOp3functional_1_critic_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*functional_1/Critic/BiasAdd/ReadVariableOpÈ
functional_1/Critic/BiasAddBiasAdd$functional_1/Critic/MatMul:product:02functional_1/Critic/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Critic/BiasAddÇ
(functional_1/Actor/MatMul/ReadVariableOpReadVariableOp1functional_1_actor_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02*
(functional_1/Actor/MatMul/ReadVariableOpÃ
functional_1/Actor/MatMulMatMul&functional_1/Dense1/Relu:activations:00functional_1/Actor/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Actor/MatMulÅ
)functional_1/Actor/BiasAdd/ReadVariableOpReadVariableOp2functional_1_actor_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)functional_1/Actor/BiasAdd/ReadVariableOpÄ
functional_1/Actor/BiasAddBiasAdd#functional_1/Actor/MatMul:product:01functional_1/Actor/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Actor/BiasAddn
IdentityIdentity#functional_1/Actor/BiasAdd:output:0*
T0*
_output_shapes

:2

Identitys

Identity_1Identity$functional_1/Critic/BiasAdd:output:0*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ:::::::F B
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex

ü
J__inference_functional_1_layer_call_and_return_conditional_losses_26795755

inputs)
%dense1_matmul_readvariableop_resource*
&dense1_biasadd_readvariableop_resource)
%critic_matmul_readvariableop_resource*
&critic_biasadd_readvariableop_resource(
$actor_matmul_readvariableop_resource)
%actor_biasadd_readvariableop_resource
identity

identity_1£
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
Actor/BiasAddj
IdentityIdentityActor/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityo

Identity_1IdentityCritic/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ:::::::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À
ü
J__inference_functional_1_layer_call_and_return_conditional_losses_26795669

inputs)
%dense1_matmul_readvariableop_resource*
&dense1_biasadd_readvariableop_resource)
%critic_matmul_readvariableop_resource*
&critic_biasadd_readvariableop_resource(
$actor_matmul_readvariableop_resource)
%actor_biasadd_readvariableop_resource
identity

identity_1£
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
Actor/BiasAdda
IdentityIdentityActor/BiasAdd:output:0*
T0*
_output_shapes

:2

Identityf

Identity_1IdentityCritic/BiasAdd:output:0*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*5
_input_shapes$
"::::::::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï
«
C__inference_Actor_layer_call_and_return_conditional_losses_26795823

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
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
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È	
Ñ
/__inference_functional_1_layer_call_fn_26795231
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1¢StatefulPartitionedCallÃ
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

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_functional_1_layer_call_and_return_conditional_losses_267952142
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
·
Â
G__inference_a2c_model_layer_call_and_return_conditional_losses_26795583
x6
2functional_1_dense1_matmul_readvariableop_resource7
3functional_1_dense1_biasadd_readvariableop_resource6
2functional_1_critic_matmul_readvariableop_resource7
3functional_1_critic_biasadd_readvariableop_resource5
1functional_1_actor_matmul_readvariableop_resource6
2functional_1_actor_biasadd_readvariableop_resource
identity

identity_1b
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

ExpandDimsÊ
)functional_1/Dense1/MatMul/ReadVariableOpReadVariableOp2functional_1_dense1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02+
)functional_1/Dense1/MatMul/ReadVariableOp´
functional_1/Dense1/MatMulMatMulExpandDims:output:01functional_1/Dense1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
functional_1/Dense1/MatMulÉ
*functional_1/Dense1/BiasAdd/ReadVariableOpReadVariableOp3functional_1_dense1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*functional_1/Dense1/BiasAdd/ReadVariableOpÉ
functional_1/Dense1/BiasAddBiasAdd$functional_1/Dense1/MatMul:product:02functional_1/Dense1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
functional_1/Dense1/BiasAdd
functional_1/Dense1/ReluRelu$functional_1/Dense1/BiasAdd:output:0*
T0*
_output_shapes
:	2
functional_1/Dense1/ReluÊ
)functional_1/Critic/MatMul/ReadVariableOpReadVariableOp2functional_1_critic_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02+
)functional_1/Critic/MatMul/ReadVariableOpÆ
functional_1/Critic/MatMulMatMul&functional_1/Dense1/Relu:activations:01functional_1/Critic/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Critic/MatMulÈ
*functional_1/Critic/BiasAdd/ReadVariableOpReadVariableOp3functional_1_critic_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*functional_1/Critic/BiasAdd/ReadVariableOpÈ
functional_1/Critic/BiasAddBiasAdd$functional_1/Critic/MatMul:product:02functional_1/Critic/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Critic/BiasAddÇ
(functional_1/Actor/MatMul/ReadVariableOpReadVariableOp1functional_1_actor_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02*
(functional_1/Actor/MatMul/ReadVariableOpÃ
functional_1/Actor/MatMulMatMul&functional_1/Dense1/Relu:activations:00functional_1/Actor/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Actor/MatMulÅ
)functional_1/Actor/BiasAdd/ReadVariableOpReadVariableOp2functional_1_actor_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)functional_1/Actor/BiasAdd/ReadVariableOpÄ
functional_1/Actor/BiasAddBiasAdd#functional_1/Actor/MatMul:product:01functional_1/Actor/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Actor/BiasAddn
IdentityIdentity#functional_1/Actor/BiasAdd:output:0*
T0*
_output_shapes

:2

Identitys

Identity_1Identity$functional_1/Critic/BiasAdd:output:0*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ:::::::F B
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
®
Ø
J__inference_functional_1_layer_call_and_return_conditional_losses_26795132
input_1
dense1_26795073
dense1_26795075
critic_26795099
critic_26795101
actor_26795125
actor_26795127
identity

identity_1¢Actor/StatefulPartitionedCall¢Critic/StatefulPartitionedCall¢Dense1/StatefulPartitionedCall
Dense1/StatefulPartitionedCallStatefulPartitionedCallinput_1dense1_26795073dense1_26795075*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_Dense1_layer_call_and_return_conditional_losses_267950622 
Dense1/StatefulPartitionedCall±
Critic/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0critic_26795099critic_26795101*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_Critic_layer_call_and_return_conditional_losses_267950882 
Critic/StatefulPartitionedCall¬
Actor/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0actor_26795125actor_26795127*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Actor_layer_call_and_return_conditional_losses_267951142
Actor/StatefulPartitionedCallÜ
IdentityIdentity&Actor/StatefulPartitionedCall:output:0^Actor/StatefulPartitionedCall^Critic/StatefulPartitionedCall^Dense1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityá

Identity_1Identity'Critic/StatefulPartitionedCall:output:0^Actor/StatefulPartitionedCall^Critic/StatefulPartitionedCall^Dense1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::::2>
Actor/StatefulPartitionedCallActor/StatefulPartitionedCall2@
Critic/StatefulPartitionedCallCritic/StatefulPartitionedCall2@
Dense1/StatefulPartitionedCallDense1/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
È	
Ñ
/__inference_functional_1_layer_call_fn_26795192
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1¢StatefulPartitionedCallÃ
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

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_functional_1_layer_call_and_return_conditional_losses_267951752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1

ü
J__inference_functional_1_layer_call_and_return_conditional_losses_26795731

inputs)
%dense1_matmul_readvariableop_resource*
&dense1_biasadd_readvariableop_resource)
%critic_matmul_readvariableop_resource*
&critic_biasadd_readvariableop_resource(
$actor_matmul_readvariableop_resource)
%actor_biasadd_readvariableop_resource
identity

identity_1£
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
Actor/BiasAddj
IdentityIdentityActor/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityo

Identity_1IdentityCritic/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ:::::::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À
ü
J__inference_functional_1_layer_call_and_return_conditional_losses_26795645

inputs)
%dense1_matmul_readvariableop_resource*
&dense1_biasadd_readvariableop_resource)
%critic_matmul_readvariableop_resource*
&critic_biasadd_readvariableop_resource(
$actor_matmul_readvariableop_resource)
%actor_biasadd_readvariableop_resource
identity

identity_1£
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
Actor/BiasAdda
IdentityIdentityActor/BiasAdd:output:0*
T0*
_output_shapes

:2

Identityf

Identity_1IdentityCritic/BiasAdd:output:0*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*5
_input_shapes$
"::::::::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë
ú
!__inference__traced_save_26795893
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
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_a2e60c7a43f94930b2874db6a60dce34/part2	
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

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

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
À
ü
J__inference_functional_1_layer_call_and_return_conditional_losses_26795284

inputs)
%dense1_matmul_readvariableop_resource*
&dense1_biasadd_readvariableop_resource)
%critic_matmul_readvariableop_resource*
&critic_biasadd_readvariableop_resource(
$actor_matmul_readvariableop_resource)
%actor_biasadd_readvariableop_resource
identity

identity_1£
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
Actor/BiasAdda
IdentityIdentityActor/BiasAdd:output:0*
T0*
_output_shapes

:2

Identityf

Identity_1IdentityCritic/BiasAdd:output:0*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*5
_input_shapes$
"::::::::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Î
,__inference_a2c_model_layer_call_fn_26795531
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1¢StatefulPartitionedCall®
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

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_a2c_model_layer_call_and_return_conditional_losses_267953842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Å	
Ð
/__inference_functional_1_layer_call_fn_26795793

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1¢StatefulPartitionedCallÂ
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

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_functional_1_layer_call_and_return_conditional_losses_267952142
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À
ü
J__inference_functional_1_layer_call_and_return_conditional_losses_26795260

inputs)
%dense1_matmul_readvariableop_resource*
&dense1_biasadd_readvariableop_resource)
%critic_matmul_readvariableop_resource*
&critic_biasadd_readvariableop_resource(
$actor_matmul_readvariableop_resource)
%actor_biasadd_readvariableop_resource
identity

identity_1£
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
Actor/BiasAdda
IdentityIdentityActor/BiasAdd:output:0*
T0*
_output_shapes

:2

Identityf

Identity_1IdentityCritic/BiasAdd:output:0*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*5
_input_shapes$
"::::::::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
Ø
J__inference_functional_1_layer_call_and_return_conditional_losses_26795152
input_1
dense1_26795135
dense1_26795137
critic_26795140
critic_26795142
actor_26795145
actor_26795147
identity

identity_1¢Actor/StatefulPartitionedCall¢Critic/StatefulPartitionedCall¢Dense1/StatefulPartitionedCall
Dense1/StatefulPartitionedCallStatefulPartitionedCallinput_1dense1_26795135dense1_26795137*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_Dense1_layer_call_and_return_conditional_losses_267950622 
Dense1/StatefulPartitionedCall±
Critic/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0critic_26795140critic_26795142*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_Critic_layer_call_and_return_conditional_losses_267950882 
Critic/StatefulPartitionedCall¬
Actor/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0actor_26795145actor_26795147*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Actor_layer_call_and_return_conditional_losses_267951142
Actor/StatefulPartitionedCallÜ
IdentityIdentity&Actor/StatefulPartitionedCall:output:0^Actor/StatefulPartitionedCall^Critic/StatefulPartitionedCall^Dense1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityá

Identity_1Identity'Critic/StatefulPartitionedCall:output:0^Actor/StatefulPartitionedCall^Critic/StatefulPartitionedCall^Dense1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::::2>
Actor/StatefulPartitionedCallActor/StatefulPartitionedCall2@
Critic/StatefulPartitionedCallCritic/StatefulPartitionedCall2@
Dense1/StatefulPartitionedCallDense1/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
É
È
G__inference_a2c_model_layer_call_and_return_conditional_losses_26795493
input_16
2functional_1_dense1_matmul_readvariableop_resource7
3functional_1_dense1_biasadd_readvariableop_resource6
2functional_1_critic_matmul_readvariableop_resource7
3functional_1_critic_biasadd_readvariableop_resource5
1functional_1_actor_matmul_readvariableop_resource6
2functional_1_actor_biasadd_readvariableop_resource
identity

identity_1b
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

ExpandDimsÊ
)functional_1/Dense1/MatMul/ReadVariableOpReadVariableOp2functional_1_dense1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02+
)functional_1/Dense1/MatMul/ReadVariableOp´
functional_1/Dense1/MatMulMatMulExpandDims:output:01functional_1/Dense1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
functional_1/Dense1/MatMulÉ
*functional_1/Dense1/BiasAdd/ReadVariableOpReadVariableOp3functional_1_dense1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*functional_1/Dense1/BiasAdd/ReadVariableOpÉ
functional_1/Dense1/BiasAddBiasAdd$functional_1/Dense1/MatMul:product:02functional_1/Dense1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
functional_1/Dense1/BiasAdd
functional_1/Dense1/ReluRelu$functional_1/Dense1/BiasAdd:output:0*
T0*
_output_shapes
:	2
functional_1/Dense1/ReluÊ
)functional_1/Critic/MatMul/ReadVariableOpReadVariableOp2functional_1_critic_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02+
)functional_1/Critic/MatMul/ReadVariableOpÆ
functional_1/Critic/MatMulMatMul&functional_1/Dense1/Relu:activations:01functional_1/Critic/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Critic/MatMulÈ
*functional_1/Critic/BiasAdd/ReadVariableOpReadVariableOp3functional_1_critic_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*functional_1/Critic/BiasAdd/ReadVariableOpÈ
functional_1/Critic/BiasAddBiasAdd$functional_1/Critic/MatMul:product:02functional_1/Critic/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Critic/BiasAddÇ
(functional_1/Actor/MatMul/ReadVariableOpReadVariableOp1functional_1_actor_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02*
(functional_1/Actor/MatMul/ReadVariableOpÃ
functional_1/Actor/MatMulMatMul&functional_1/Dense1/Relu:activations:00functional_1/Actor/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Actor/MatMulÅ
)functional_1/Actor/BiasAdd/ReadVariableOpReadVariableOp2functional_1_actor_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)functional_1/Actor/BiasAdd/ReadVariableOpÄ
functional_1/Actor/BiasAddBiasAdd#functional_1/Actor/MatMul:product:01functional_1/Actor/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Actor/BiasAddn
IdentityIdentity#functional_1/Actor/BiasAdd:output:0*
T0*
_output_shapes

:2

Identitys

Identity_1Identity$functional_1/Critic/BiasAdd:output:0*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ:::::::L H
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1

º
G__inference_a2c_model_layer_call_and_return_conditional_losses_26795384
x
functional_1_26795368
functional_1_26795370
functional_1_26795372
functional_1_26795374
functional_1_26795376
functional_1_26795378
identity

identity_1¢$functional_1/StatefulPartitionedCallb
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

ExpandDims¡
$functional_1/StatefulPartitionedCallStatefulPartitionedCallExpandDims:output:0functional_1_26795368functional_1_26795370functional_1_26795372functional_1_26795374functional_1_26795376functional_1_26795378*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
::*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_functional_1_layer_call_and_return_conditional_losses_267952842&
$functional_1/StatefulPartitionedCall
IdentityIdentity-functional_1/StatefulPartitionedCall:output:0%^functional_1/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity£

Identity_1Identity-functional_1/StatefulPartitionedCall:output:1%^functional_1/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::::2L
$functional_1/StatefulPartitionedCall$functional_1/StatefulPartitionedCall:F B
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
	
È
,__inference_a2c_model_layer_call_fn_26795621
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1¢StatefulPartitionedCall¨
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

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_a2c_model_layer_call_and_return_conditional_losses_267953842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:F B
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
Ð
¬
D__inference_Critic_layer_call_and_return_conditional_losses_26795842

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
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
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï
«
C__inference_Actor_layer_call_and_return_conditional_losses_26795114

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
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
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
×
J__inference_functional_1_layer_call_and_return_conditional_losses_26795175

inputs
dense1_26795158
dense1_26795160
critic_26795163
critic_26795165
actor_26795168
actor_26795170
identity

identity_1¢Actor/StatefulPartitionedCall¢Critic/StatefulPartitionedCall¢Dense1/StatefulPartitionedCall
Dense1/StatefulPartitionedCallStatefulPartitionedCallinputsdense1_26795158dense1_26795160*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_Dense1_layer_call_and_return_conditional_losses_267950622 
Dense1/StatefulPartitionedCall±
Critic/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0critic_26795163critic_26795165*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_Critic_layer_call_and_return_conditional_losses_267950882 
Critic/StatefulPartitionedCall¬
Actor/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0actor_26795168actor_26795170*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Actor_layer_call_and_return_conditional_losses_267951142
Actor/StatefulPartitionedCallÜ
IdentityIdentity&Actor/StatefulPartitionedCall:output:0^Actor/StatefulPartitionedCall^Critic/StatefulPartitionedCall^Dense1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityá

Identity_1Identity'Critic/StatefulPartitionedCall:output:0^Actor/StatefulPartitionedCall^Critic/StatefulPartitionedCall^Dense1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::::2>
Actor/StatefulPartitionedCallActor/StatefulPartitionedCall2@
Critic/StatefulPartitionedCallCritic/StatefulPartitionedCall2@
Dense1/StatefulPartitionedCallDense1/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê 
à
#__inference__wrapped_model_26795047
input_1@
<a2c_model_functional_1_dense1_matmul_readvariableop_resourceA
=a2c_model_functional_1_dense1_biasadd_readvariableop_resource@
<a2c_model_functional_1_critic_matmul_readvariableop_resourceA
=a2c_model_functional_1_critic_biasadd_readvariableop_resource?
;a2c_model_functional_1_actor_matmul_readvariableop_resource@
<a2c_model_functional_1_actor_biasadd_readvariableop_resource
identity

identity_1v
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
a2c_model/ExpandDimsè
3a2c_model/functional_1/Dense1/MatMul/ReadVariableOpReadVariableOp<a2c_model_functional_1_dense1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype025
3a2c_model/functional_1/Dense1/MatMul/ReadVariableOpÜ
$a2c_model/functional_1/Dense1/MatMulMatMula2c_model/ExpandDims:output:0;a2c_model/functional_1/Dense1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2&
$a2c_model/functional_1/Dense1/MatMulç
4a2c_model/functional_1/Dense1/BiasAdd/ReadVariableOpReadVariableOp=a2c_model_functional_1_dense1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype026
4a2c_model/functional_1/Dense1/BiasAdd/ReadVariableOpñ
%a2c_model/functional_1/Dense1/BiasAddBiasAdd.a2c_model/functional_1/Dense1/MatMul:product:0<a2c_model/functional_1/Dense1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2'
%a2c_model/functional_1/Dense1/BiasAddª
"a2c_model/functional_1/Dense1/ReluRelu.a2c_model/functional_1/Dense1/BiasAdd:output:0*
T0*
_output_shapes
:	2$
"a2c_model/functional_1/Dense1/Reluè
3a2c_model/functional_1/Critic/MatMul/ReadVariableOpReadVariableOp<a2c_model_functional_1_critic_matmul_readvariableop_resource*
_output_shapes
:	*
dtype025
3a2c_model/functional_1/Critic/MatMul/ReadVariableOpî
$a2c_model/functional_1/Critic/MatMulMatMul0a2c_model/functional_1/Dense1/Relu:activations:0;a2c_model/functional_1/Critic/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2&
$a2c_model/functional_1/Critic/MatMulæ
4a2c_model/functional_1/Critic/BiasAdd/ReadVariableOpReadVariableOp=a2c_model_functional_1_critic_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4a2c_model/functional_1/Critic/BiasAdd/ReadVariableOpð
%a2c_model/functional_1/Critic/BiasAddBiasAdd.a2c_model/functional_1/Critic/MatMul:product:0<a2c_model/functional_1/Critic/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2'
%a2c_model/functional_1/Critic/BiasAddå
2a2c_model/functional_1/Actor/MatMul/ReadVariableOpReadVariableOp;a2c_model_functional_1_actor_matmul_readvariableop_resource*
_output_shapes
:	*
dtype024
2a2c_model/functional_1/Actor/MatMul/ReadVariableOpë
#a2c_model/functional_1/Actor/MatMulMatMul0a2c_model/functional_1/Dense1/Relu:activations:0:a2c_model/functional_1/Actor/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2%
#a2c_model/functional_1/Actor/MatMulã
3a2c_model/functional_1/Actor/BiasAdd/ReadVariableOpReadVariableOp<a2c_model_functional_1_actor_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3a2c_model/functional_1/Actor/BiasAdd/ReadVariableOpì
$a2c_model/functional_1/Actor/BiasAddBiasAdd-a2c_model/functional_1/Actor/MatMul:product:0;a2c_model/functional_1/Actor/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2&
$a2c_model/functional_1/Actor/BiasAddx
IdentityIdentity-a2c_model/functional_1/Actor/BiasAdd:output:0*
T0*
_output_shapes

:2

Identity}

Identity_1Identity.a2c_model/functional_1/Critic/BiasAdd:output:0*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ:::::::L H
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ý
 
$__inference__traced_restore_26795921
file_prefix"
assignvariableop_dense1_kernel"
assignvariableop_1_dense1_bias#
assignvariableop_2_actor_kernel!
assignvariableop_3_actor_bias$
 assignvariableop_4_critic_kernel"
assignvariableop_5_critic_bias

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

Identity_6Ö

Identity_7IdentityIdentity_6:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*
T0*
_output_shapes
: 2

Identity_7"!

identity_7Identity_7:output:0*-
_input_shapes
: ::::::2$
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
	
È
,__inference_a2c_model_layer_call_fn_26795602
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1¢StatefulPartitionedCall¨
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

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_a2c_model_layer_call_and_return_conditional_losses_267953842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:F B
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
Å	
Ð
/__inference_functional_1_layer_call_fn_26795774

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1¢StatefulPartitionedCallÂ
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

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_functional_1_layer_call_and_return_conditional_losses_267951752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¡	
Ð
/__inference_functional_1_layer_call_fn_26795707

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1¢StatefulPartitionedCall°
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

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_functional_1_layer_call_and_return_conditional_losses_267952842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ
~
)__inference_Dense1_layer_call_fn_26795813

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallõ
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
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_Dense1_layer_call_and_return_conditional_losses_267950622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¡	
Ð
/__inference_functional_1_layer_call_fn_26795688

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1¢StatefulPartitionedCall°
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

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_functional_1_layer_call_and_return_conditional_losses_267952602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
¬
D__inference_Critic_layer_call_and_return_conditional_losses_26795088

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
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
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
×
J__inference_functional_1_layer_call_and_return_conditional_losses_26795214

inputs
dense1_26795197
dense1_26795199
critic_26795202
critic_26795204
actor_26795207
actor_26795209
identity

identity_1¢Actor/StatefulPartitionedCall¢Critic/StatefulPartitionedCall¢Dense1/StatefulPartitionedCall
Dense1/StatefulPartitionedCallStatefulPartitionedCallinputsdense1_26795197dense1_26795199*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_Dense1_layer_call_and_return_conditional_losses_267950622 
Dense1/StatefulPartitionedCall±
Critic/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0critic_26795202critic_26795204*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_Critic_layer_call_and_return_conditional_losses_267950882 
Critic/StatefulPartitionedCall¬
Actor/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0actor_26795207actor_26795209*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Actor_layer_call_and_return_conditional_losses_267951142
Actor/StatefulPartitionedCallÜ
IdentityIdentity&Actor/StatefulPartitionedCall:output:0^Actor/StatefulPartitionedCall^Critic/StatefulPartitionedCall^Dense1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityá

Identity_1Identity'Critic/StatefulPartitionedCall:output:0^Actor/StatefulPartitionedCall^Critic/StatefulPartitionedCall^Dense1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::::2>
Actor/StatefulPartitionedCallActor/StatefulPartitionedCall2@
Critic/StatefulPartitionedCallCritic/StatefulPartitionedCall2@
Dense1/StatefulPartitionedCallDense1/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
}
(__inference_Actor_layer_call_fn_26795832

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCalló
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
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Actor_layer_call_and_return_conditional_losses_267951142
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Î
,__inference_a2c_model_layer_call_fn_26795512
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1¢StatefulPartitionedCall®
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

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_a2c_model_layer_call_and_return_conditional_losses_267953842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
¯
¬
D__inference_Dense1_layer_call_and_return_conditional_losses_26795804

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
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
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯
¬
D__inference_Dense1_layer_call_and_return_conditional_losses_26795062

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
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
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É
È
G__inference_a2c_model_layer_call_and_return_conditional_losses_26795467
input_16
2functional_1_dense1_matmul_readvariableop_resource7
3functional_1_dense1_biasadd_readvariableop_resource6
2functional_1_critic_matmul_readvariableop_resource7
3functional_1_critic_biasadd_readvariableop_resource5
1functional_1_actor_matmul_readvariableop_resource6
2functional_1_actor_biasadd_readvariableop_resource
identity

identity_1b
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

ExpandDimsÊ
)functional_1/Dense1/MatMul/ReadVariableOpReadVariableOp2functional_1_dense1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02+
)functional_1/Dense1/MatMul/ReadVariableOp´
functional_1/Dense1/MatMulMatMulExpandDims:output:01functional_1/Dense1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
functional_1/Dense1/MatMulÉ
*functional_1/Dense1/BiasAdd/ReadVariableOpReadVariableOp3functional_1_dense1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*functional_1/Dense1/BiasAdd/ReadVariableOpÉ
functional_1/Dense1/BiasAddBiasAdd$functional_1/Dense1/MatMul:product:02functional_1/Dense1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
functional_1/Dense1/BiasAdd
functional_1/Dense1/ReluRelu$functional_1/Dense1/BiasAdd:output:0*
T0*
_output_shapes
:	2
functional_1/Dense1/ReluÊ
)functional_1/Critic/MatMul/ReadVariableOpReadVariableOp2functional_1_critic_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02+
)functional_1/Critic/MatMul/ReadVariableOpÆ
functional_1/Critic/MatMulMatMul&functional_1/Dense1/Relu:activations:01functional_1/Critic/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Critic/MatMulÈ
*functional_1/Critic/BiasAdd/ReadVariableOpReadVariableOp3functional_1_critic_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*functional_1/Critic/BiasAdd/ReadVariableOpÈ
functional_1/Critic/BiasAddBiasAdd$functional_1/Critic/MatMul:product:02functional_1/Critic/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Critic/BiasAddÇ
(functional_1/Actor/MatMul/ReadVariableOpReadVariableOp1functional_1_actor_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02*
(functional_1/Actor/MatMul/ReadVariableOpÃ
functional_1/Actor/MatMulMatMul&functional_1/Dense1/Relu:activations:00functional_1/Actor/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Actor/MatMulÅ
)functional_1/Actor/BiasAdd/ReadVariableOpReadVariableOp2functional_1_actor_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)functional_1/Actor/BiasAdd/ReadVariableOpÄ
functional_1/Actor/BiasAddBiasAdd#functional_1/Actor/MatMul:product:01functional_1/Actor/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Actor/BiasAddn
IdentityIdentity#functional_1/Actor/BiasAdd:output:0*
T0*
_output_shapes

:2

Identitys

Identity_1Identity$functional_1/Critic/BiasAdd:output:0*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ:::::::L H
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ì
È
&__inference_signature_wrapper_26795441
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1¢StatefulPartitionedCall
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

*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__wrapped_model_267950472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1"¸L
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
StatefulPartitionedCall:1tensorflow/serving/predict:ê
Á
nn
trainable_variables
	variables
regularization_losses
	keras_api

signatures
:__call__
*;&call_and_return_all_conditional_losses
<_default_save_signature"ý
_tf_keras_modelã{"class_name": "a2c_model", "name": "a2c_model", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "a2c_model"}}
Í!
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
*>&call_and_return_all_conditional_losses"¼
_tf_keras_network {"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "Dense1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": 41}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Actor", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Actor", "inbound_nodes": [[["Dense1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Critic", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Critic", "inbound_nodes": [[["Dense1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["Actor", 0, 0], ["Critic", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "Dense1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": 41}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Actor", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Actor", "inbound_nodes": [[["Dense1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Critic", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Critic", "inbound_nodes": [[["Dense1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["Actor", 0, 0], ["Critic", 0, 0]]}}}
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
	variables
non_trainable_variables
layer_regularization_losses

layers
layer_metrics
metrics
regularization_losses
:__call__
<_default_save_signature
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
,
?serving_default"
signature_map
é"æ
_tf_keras_input_layerÆ{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
æ

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
@__call__
*A&call_and_return_all_conditional_losses"Á
_tf_keras_layer§{"class_name": "Dense", "name": "Dense1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Dense1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": 41}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}}
ï

kernel
bias
trainable_variables
	variables
 regularization_losses
!	keras_api
B__call__
*C&call_and_return_all_conditional_losses"Ê
_tf_keras_layer°{"class_name": "Dense", "name": "Actor", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Actor", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
ñ

kernel
bias
"trainable_variables
#	variables
$regularization_losses
%	keras_api
D__call__
*E&call_and_return_all_conditional_losses"Ì
_tf_keras_layer²{"class_name": "Dense", "name": "Critic", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Critic", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
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
	variables
&non_trainable_variables
'layer_regularization_losses

(layers
)layer_metrics
*metrics
regularization_losses
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
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
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
	variables
+non_trainable_variables
,layer_regularization_losses

-layers
.layer_metrics
/metrics
regularization_losses
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
	variables
0non_trainable_variables
1layer_regularization_losses

2layers
3layer_metrics
4metrics
 regularization_losses
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
#	variables
5non_trainable_variables
6layer_regularization_losses

7layers
8layer_metrics
9metrics
$regularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
ì2é
,__inference_a2c_model_layer_call_fn_26795602
,__inference_a2c_model_layer_call_fn_26795621
,__inference_a2c_model_layer_call_fn_26795531
,__inference_a2c_model_layer_call_fn_26795512®
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
Ø2Õ
G__inference_a2c_model_layer_call_and_return_conditional_losses_26795467
G__inference_a2c_model_layer_call_and_return_conditional_losses_26795583
G__inference_a2c_model_layer_call_and_return_conditional_losses_26795493
G__inference_a2c_model_layer_call_and_return_conditional_losses_26795557®
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
Ý2Ú
#__inference__wrapped_model_26795047²
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *"¢

input_1ÿÿÿÿÿÿÿÿÿ
ì2é
/__inference_functional_1_layer_call_fn_26795231
/__inference_functional_1_layer_call_fn_26795774
/__inference_functional_1_layer_call_fn_26795793
/__inference_functional_1_layer_call_fn_26795707
/__inference_functional_1_layer_call_fn_26795688
/__inference_functional_1_layer_call_fn_26795192À
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
2
J__inference_functional_1_layer_call_and_return_conditional_losses_26795132
J__inference_functional_1_layer_call_and_return_conditional_losses_26795645
J__inference_functional_1_layer_call_and_return_conditional_losses_26795731
J__inference_functional_1_layer_call_and_return_conditional_losses_26795669
J__inference_functional_1_layer_call_and_return_conditional_losses_26795152
J__inference_functional_1_layer_call_and_return_conditional_losses_26795755À
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
5B3
&__inference_signature_wrapper_26795441input_1
Ó2Ð
)__inference_Dense1_layer_call_fn_26795813¢
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
î2ë
D__inference_Dense1_layer_call_and_return_conditional_losses_26795804¢
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
Ò2Ï
(__inference_Actor_layer_call_fn_26795832¢
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
í2ê
C__inference_Actor_layer_call_and_return_conditional_losses_26795823¢
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
Ó2Ð
)__inference_Critic_layer_call_fn_26795851¢
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
î2ë
D__inference_Critic_layer_call_and_return_conditional_losses_26795842¢
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
 ¤
C__inference_Actor_layer_call_and_return_conditional_losses_26795823]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
(__inference_Actor_layer_call_fn_26795832P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¥
D__inference_Critic_layer_call_and_return_conditional_losses_26795842]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
)__inference_Critic_layer_call_fn_26795851P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¥
D__inference_Dense1_layer_call_and_return_conditional_losses_26795804]/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 }
)__inference_Dense1_layer_call_fn_26795813P/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ±
#__inference__wrapped_model_26795047,¢)
"¢

input_1ÿÿÿÿÿÿÿÿÿ
ª "QªN
%
output_1
output_1
%
output_2
output_2À
G__inference_a2c_model_layer_call_and_return_conditional_losses_26795467u0¢-
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
 À
G__inference_a2c_model_layer_call_and_return_conditional_losses_26795493u0¢-
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
 º
G__inference_a2c_model_layer_call_and_return_conditional_losses_26795557o*¢'
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
 º
G__inference_a2c_model_layer_call_and_return_conditional_losses_26795583o*¢'
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
 
,__inference_a2c_model_layer_call_fn_26795512g0¢-
&¢#

input_1ÿÿÿÿÿÿÿÿÿ
p
ª "+(

0

1
,__inference_a2c_model_layer_call_fn_26795531g0¢-
&¢#

input_1ÿÿÿÿÿÿÿÿÿ
p 
ª "+(

0

1
,__inference_a2c_model_layer_call_fn_26795602a*¢'
 ¢

xÿÿÿÿÿÿÿÿÿ
p
ª "+(

0

1
,__inference_a2c_model_layer_call_fn_26795621a*¢'
 ¢

xÿÿÿÿÿÿÿÿÿ
p 
ª "+(

0

1Þ
J__inference_functional_1_layer_call_and_return_conditional_losses_267951328¢5
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
 Þ
J__inference_functional_1_layer_call_and_return_conditional_losses_267951528¢5
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
 Ê
J__inference_functional_1_layer_call_and_return_conditional_losses_26795645|7¢4
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
 Ê
J__inference_functional_1_layer_call_and_return_conditional_losses_26795669|7¢4
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
 Ý
J__inference_functional_1_layer_call_and_return_conditional_losses_267957317¢4
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
 Ý
J__inference_functional_1_layer_call_and_return_conditional_losses_267957557¢4
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
 µ
/__inference_functional_1_layer_call_fn_267951928¢5
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
1ÿÿÿÿÿÿÿÿÿµ
/__inference_functional_1_layer_call_fn_267952318¢5
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
1ÿÿÿÿÿÿÿÿÿ¡
/__inference_functional_1_layer_call_fn_26795688n7¢4
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
1¡
/__inference_functional_1_layer_call_fn_26795707n7¢4
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
1´
/__inference_functional_1_layer_call_fn_267957747¢4
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
1ÿÿÿÿÿÿÿÿÿ´
/__inference_functional_1_layer_call_fn_267957937¢4
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
1ÿÿÿÿÿÿÿÿÿ¿
&__inference_signature_wrapper_267954417¢4
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