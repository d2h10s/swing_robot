??
??
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
dtypetype?
?
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
executor_typestring ?
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.3.02unknown8??
w
Dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_nameDense1/kernel
p
!Dense1/kernel/Read/ReadVariableOpReadVariableOpDense1/kernel*
_output_shapes
:	?*
dtype0
o
Dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameDense1/bias
h
Dense1/bias/Read/ReadVariableOpReadVariableOpDense1/bias*
_output_shapes	
:?*
dtype0
u
Actor/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_nameActor/kernel
n
 Actor/kernel/Read/ReadVariableOpReadVariableOpActor/kernel*
_output_shapes
:	?*
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
shape:	?*
shared_nameCritic/kernel
p
!Critic/kernel/Read/ReadVariableOpReadVariableOpCritic/kernel*
_output_shapes
:	?*
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
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
j
nn
trainable_variables
regularization_losses
	variables
	keras_api

signatures
?
nn

signatures
#	_self_saveable_object_factories

trainable_variables
regularization_losses
	variables
	keras_api
*
0
1
2
3
4
5
 
*
0
1
2
3
4
5
?
layer_regularization_losses
trainable_variables
regularization_losses
	variables
non_trainable_variables

layers
metrics
layer_metrics
 
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
#_self_saveable_object_factories
trainable_variables
regularization_losses
 	variables
!	keras_api
 
 
*
0
1
2
3
4
5
 
*
0
1
2
3
4
5
?
"layer_regularization_losses

trainable_variables
regularization_losses
	variables
#non_trainable_variables

$layers
%metrics
&layer_metrics
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
%
#'_self_saveable_object_factories
?

kernel
bias
#(_self_saveable_object_factories
)trainable_variables
*regularization_losses
+	variables
,	keras_api
?

kernel
bias
#-_self_saveable_object_factories
.trainable_variables
/regularization_losses
0	variables
1	keras_api
?

kernel
bias
#2_self_saveable_object_factories
3trainable_variables
4regularization_losses
5	variables
6	keras_api
 
*
0
1
2
3
4
5
 
*
0
1
2
3
4
5
?
7layer_regularization_losses
trainable_variables
regularization_losses
 	variables
8non_trainable_variables

9layers
:metrics
;layer_metrics
 
 

0
 
 
 
 

0
1
 

0
1
?
<layer_regularization_losses
)trainable_variables
*regularization_losses
+	variables
=non_trainable_variables

>layers
?metrics
@layer_metrics
 

0
1
 

0
1
?
Alayer_regularization_losses
.trainable_variables
/regularization_losses
0	variables
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_metrics
 

0
1
 

0
1
?
Flayer_regularization_losses
3trainable_variables
4regularization_losses
5	variables
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_metrics
 
 

0
1
2
3
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
:?????????*
dtype0*
shape:?????????
?
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
GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_161572
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
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
GPU 2J 8? *(
f#R!
__inference__traced_save_162094
?
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
GPU 2J 8? *+
f&R$
"__inference__traced_restore_162122??
?

?
E__inference_a2c_model_layer_call_and_return_conditional_losses_161794
input_1
a2c_model_161778
a2c_model_161780
a2c_model_161782
a2c_model_161784
a2c_model_161786
a2c_model_161788
identity

identity_1??!a2c_model/StatefulPartitionedCall?
!a2c_model/StatefulPartitionedCallStatefulPartitionedCallinput_1a2c_model_161778a2c_model_161780a2c_model_161782a2c_model_161784a2c_model_161786a2c_model_161788*
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
GPU 2J 8? *2
f-R+
)__inference_restored_function_body_1614182#
!a2c_model/StatefulPartitionedCall?
IdentityIdentity*a2c_model/StatefulPartitionedCall:output:0"^a2c_model/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity?

Identity_1Identity*a2c_model/StatefulPartitionedCall:output:1"^a2c_model/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::::2F
!a2c_model/StatefulPartitionedCall!a2c_model/StatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
B__inference_a2c_model_layer_call_and_return_conditional_losses_598
input_16
2functional_1_dense1_matmul_readvariableop_resource7
3functional_1_dense1_biasadd_readvariableop_resource6
2functional_1_critic_matmul_readvariableop_resource7
3functional_1_critic_biasadd_readvariableop_resource5
1functional_1_actor_matmul_readvariableop_resource6
2functional_1_actor_biasadd_readvariableop_resource
identity

identity_1?b
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

ExpandDims?
)functional_1/Dense1/MatMul/ReadVariableOpReadVariableOp2functional_1_dense1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02+
)functional_1/Dense1/MatMul/ReadVariableOp?
functional_1/Dense1/MatMulMatMulExpandDims:output:01functional_1/Dense1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
functional_1/Dense1/MatMul?
*functional_1/Dense1/BiasAdd/ReadVariableOpReadVariableOp3functional_1_dense1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*functional_1/Dense1/BiasAdd/ReadVariableOp?
functional_1/Dense1/BiasAddBiasAdd$functional_1/Dense1/MatMul:product:02functional_1/Dense1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
functional_1/Dense1/BiasAdd?
functional_1/Dense1/ReluRelu$functional_1/Dense1/BiasAdd:output:0*
T0*
_output_shapes
:	?2
functional_1/Dense1/Relu?
)functional_1/Critic/MatMul/ReadVariableOpReadVariableOp2functional_1_critic_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02+
)functional_1/Critic/MatMul/ReadVariableOp?
functional_1/Critic/MatMulMatMul&functional_1/Dense1/Relu:activations:01functional_1/Critic/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Critic/MatMul?
*functional_1/Critic/BiasAdd/ReadVariableOpReadVariableOp3functional_1_critic_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*functional_1/Critic/BiasAdd/ReadVariableOp?
functional_1/Critic/BiasAddBiasAdd$functional_1/Critic/MatMul:product:02functional_1/Critic/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Critic/BiasAdd?
(functional_1/Actor/MatMul/ReadVariableOpReadVariableOp1functional_1_actor_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(functional_1/Actor/MatMul/ReadVariableOp?
functional_1/Actor/MatMulMatMul&functional_1/Dense1/Relu:activations:00functional_1/Actor/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Actor/MatMul?
)functional_1/Actor/BiasAdd/ReadVariableOpReadVariableOp2functional_1_actor_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)functional_1/Actor/BiasAdd/ReadVariableOp?
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
':?????????:::::::L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
B__inference_a2c_model_layer_call_and_return_conditional_losses_536
x6
2functional_1_dense1_matmul_readvariableop_resource7
3functional_1_dense1_biasadd_readvariableop_resource6
2functional_1_critic_matmul_readvariableop_resource7
3functional_1_critic_biasadd_readvariableop_resource5
1functional_1_actor_matmul_readvariableop_resource6
2functional_1_actor_biasadd_readvariableop_resource
identity

identity_1?b
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

ExpandDims?
)functional_1/Dense1/MatMul/ReadVariableOpReadVariableOp2functional_1_dense1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02+
)functional_1/Dense1/MatMul/ReadVariableOp?
functional_1/Dense1/MatMulMatMulExpandDims:output:01functional_1/Dense1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
functional_1/Dense1/MatMul?
*functional_1/Dense1/BiasAdd/ReadVariableOpReadVariableOp3functional_1_dense1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*functional_1/Dense1/BiasAdd/ReadVariableOp?
functional_1/Dense1/BiasAddBiasAdd$functional_1/Dense1/MatMul:product:02functional_1/Dense1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
functional_1/Dense1/BiasAdd?
functional_1/Dense1/ReluRelu$functional_1/Dense1/BiasAdd:output:0*
T0*
_output_shapes
:	?2
functional_1/Dense1/Relu?
)functional_1/Critic/MatMul/ReadVariableOpReadVariableOp2functional_1_critic_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02+
)functional_1/Critic/MatMul/ReadVariableOp?
functional_1/Critic/MatMulMatMul&functional_1/Dense1/Relu:activations:01functional_1/Critic/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Critic/MatMul?
*functional_1/Critic/BiasAdd/ReadVariableOpReadVariableOp3functional_1_critic_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*functional_1/Critic/BiasAdd/ReadVariableOp?
functional_1/Critic/BiasAddBiasAdd$functional_1/Critic/MatMul:product:02functional_1/Critic/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Critic/BiasAdd?
(functional_1/Actor/MatMul/ReadVariableOpReadVariableOp1functional_1_actor_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(functional_1/Actor/MatMul/ReadVariableOp?
functional_1/Actor/MatMulMatMul&functional_1/Dense1/Relu:activations:00functional_1/Actor/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Actor/MatMul?
)functional_1/Actor/BiasAdd/ReadVariableOpReadVariableOp2functional_1_actor_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)functional_1/Actor/BiasAdd/ReadVariableOp?
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
':?????????:::::::F B
#
_output_shapes
:?????????

_user_specified_namex
? 
?
__inference__wrapped_model_624
input_1@
<a2c_model_functional_1_dense1_matmul_readvariableop_resourceA
=a2c_model_functional_1_dense1_biasadd_readvariableop_resource@
<a2c_model_functional_1_critic_matmul_readvariableop_resourceA
=a2c_model_functional_1_critic_biasadd_readvariableop_resource?
;a2c_model_functional_1_actor_matmul_readvariableop_resource@
<a2c_model_functional_1_actor_biasadd_readvariableop_resource
identity

identity_1?v
a2c_model/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
a2c_model/ExpandDims/dim?
a2c_model/ExpandDims
ExpandDimsinput_1!a2c_model/ExpandDims/dim:output:0*
T0*
_output_shapes

:2
a2c_model/ExpandDims?
3a2c_model/functional_1/Dense1/MatMul/ReadVariableOpReadVariableOp<a2c_model_functional_1_dense1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype025
3a2c_model/functional_1/Dense1/MatMul/ReadVariableOp?
$a2c_model/functional_1/Dense1/MatMulMatMula2c_model/ExpandDims:output:0;a2c_model/functional_1/Dense1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2&
$a2c_model/functional_1/Dense1/MatMul?
4a2c_model/functional_1/Dense1/BiasAdd/ReadVariableOpReadVariableOp=a2c_model_functional_1_dense1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype026
4a2c_model/functional_1/Dense1/BiasAdd/ReadVariableOp?
%a2c_model/functional_1/Dense1/BiasAddBiasAdd.a2c_model/functional_1/Dense1/MatMul:product:0<a2c_model/functional_1/Dense1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2'
%a2c_model/functional_1/Dense1/BiasAdd?
"a2c_model/functional_1/Dense1/ReluRelu.a2c_model/functional_1/Dense1/BiasAdd:output:0*
T0*
_output_shapes
:	?2$
"a2c_model/functional_1/Dense1/Relu?
3a2c_model/functional_1/Critic/MatMul/ReadVariableOpReadVariableOp<a2c_model_functional_1_critic_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype025
3a2c_model/functional_1/Critic/MatMul/ReadVariableOp?
$a2c_model/functional_1/Critic/MatMulMatMul0a2c_model/functional_1/Dense1/Relu:activations:0;a2c_model/functional_1/Critic/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2&
$a2c_model/functional_1/Critic/MatMul?
4a2c_model/functional_1/Critic/BiasAdd/ReadVariableOpReadVariableOp=a2c_model_functional_1_critic_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4a2c_model/functional_1/Critic/BiasAdd/ReadVariableOp?
%a2c_model/functional_1/Critic/BiasAddBiasAdd.a2c_model/functional_1/Critic/MatMul:product:0<a2c_model/functional_1/Critic/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2'
%a2c_model/functional_1/Critic/BiasAdd?
2a2c_model/functional_1/Actor/MatMul/ReadVariableOpReadVariableOp;a2c_model_functional_1_actor_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype024
2a2c_model/functional_1/Actor/MatMul/ReadVariableOp?
#a2c_model/functional_1/Actor/MatMulMatMul0a2c_model/functional_1/Dense1/Relu:activations:0:a2c_model/functional_1/Actor/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2%
#a2c_model/functional_1/Actor/MatMul?
3a2c_model/functional_1/Actor/BiasAdd/ReadVariableOpReadVariableOp<a2c_model_functional_1_actor_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3a2c_model/functional_1/Actor/BiasAdd/ReadVariableOp?
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
':?????????:::::::L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1
?	
?
*__inference_a2c_model_layer_call_fn_161908
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1??StatefulPartitionedCall?
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
GPU 2J 8? *N
fIRG
E__inference_a2c_model_layer_call_and_return_conditional_losses_1615152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:F B
#
_output_shapes
:?????????

_user_specified_namex
?	
?
-__inference_functional_1_layer_call_fn_161756
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_1617392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?	
?
-__inference_functional_1_layer_call_fn_161717
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_1617002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
!__inference_signature_wrapper_637
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1??StatefulPartitionedCall?
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
GPU 2J 8? *'
f"R 
__inference__wrapped_model_6242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
B__inference_a2c_model_layer_call_and_return_conditional_losses_136
x6
2functional_1_dense1_matmul_readvariableop_resource7
3functional_1_dense1_biasadd_readvariableop_resource6
2functional_1_critic_matmul_readvariableop_resource7
3functional_1_critic_biasadd_readvariableop_resource5
1functional_1_actor_matmul_readvariableop_resource6
2functional_1_actor_biasadd_readvariableop_resource
identity

identity_1?b
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

ExpandDims?
)functional_1/Dense1/MatMul/ReadVariableOpReadVariableOp2functional_1_dense1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02+
)functional_1/Dense1/MatMul/ReadVariableOp?
functional_1/Dense1/MatMulMatMulExpandDims:output:01functional_1/Dense1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
functional_1/Dense1/MatMul?
*functional_1/Dense1/BiasAdd/ReadVariableOpReadVariableOp3functional_1_dense1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*functional_1/Dense1/BiasAdd/ReadVariableOp?
functional_1/Dense1/BiasAddBiasAdd$functional_1/Dense1/MatMul:product:02functional_1/Dense1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
functional_1/Dense1/BiasAdd?
functional_1/Dense1/ReluRelu$functional_1/Dense1/BiasAdd:output:0*
T0*
_output_shapes
:	?2
functional_1/Dense1/Relu?
)functional_1/Critic/MatMul/ReadVariableOpReadVariableOp2functional_1_critic_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02+
)functional_1/Critic/MatMul/ReadVariableOp?
functional_1/Critic/MatMulMatMul&functional_1/Dense1/Relu:activations:01functional_1/Critic/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Critic/MatMul?
*functional_1/Critic/BiasAdd/ReadVariableOpReadVariableOp3functional_1_critic_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*functional_1/Critic/BiasAdd/ReadVariableOp?
functional_1/Critic/BiasAddBiasAdd$functional_1/Critic/MatMul:product:02functional_1/Critic/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Critic/BiasAdd?
(functional_1/Actor/MatMul/ReadVariableOpReadVariableOp1functional_1_actor_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(functional_1/Actor/MatMul/ReadVariableOp?
functional_1/Actor/MatMulMatMul&functional_1/Dense1/Relu:activations:00functional_1/Actor/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Actor/MatMul?
)functional_1/Actor/BiasAdd/ReadVariableOpReadVariableOp2functional_1_actor_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)functional_1/Actor/BiasAdd/ReadVariableOp?
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
':?????????:::::::F B
#
_output_shapes
:?????????

_user_specified_namex
?
?
B__inference_Dense1_layer_call_and_return_conditional_losses_162005

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
A__inference_Actor_layer_call_and_return_conditional_losses_162024

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
H__inference_functional_1_layer_call_and_return_conditional_losses_161956

inputs)
%dense1_matmul_readvariableop_resource*
&dense1_biasadd_readvariableop_resource)
%critic_matmul_readvariableop_resource*
&critic_biasadd_readvariableop_resource(
$actor_matmul_readvariableop_resource)
%actor_biasadd_readvariableop_resource
identity

identity_1??
Dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
Dense1/MatMul/ReadVariableOp?
Dense1/MatMulMatMulinputs$Dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Dense1/MatMul?
Dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
Dense1/BiasAdd/ReadVariableOp?
Dense1/BiasAddBiasAddDense1/MatMul:product:0%Dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Dense1/BiasAddn
Dense1/ReluReluDense1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Dense1/Relu?
Critic/MatMul/ReadVariableOpReadVariableOp%critic_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
Critic/MatMul/ReadVariableOp?
Critic/MatMulMatMulDense1/Relu:activations:0$Critic/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Critic/MatMul?
Critic/BiasAdd/ReadVariableOpReadVariableOp&critic_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
Critic/BiasAdd/ReadVariableOp?
Critic/BiasAddBiasAddCritic/MatMul:product:0%Critic/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Critic/BiasAdd?
Actor/MatMul/ReadVariableOpReadVariableOp$actor_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
Actor/MatMul/ReadVariableOp?
Actor/MatMulMatMulDense1/Relu:activations:0#Actor/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Actor/MatMul?
Actor/BiasAdd/ReadVariableOpReadVariableOp%actor_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
Actor/BiasAdd/ReadVariableOp?
Actor/BiasAddBiasAddActor/MatMul:product:0$Actor/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Actor/BiasAddj
IdentityIdentityActor/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identityo

Identity_1IdentityCritic/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:?????????:::::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_restored_function_body_161457
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*(
_output_shapes
::*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_a2c_model_layer_call_and_return_conditional_losses_5362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:F B
#
_output_shapes
:?????????

_user_specified_namex
?
|
'__inference_Critic_layer_call_fn_162052

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Critic_layer_call_and_return_conditional_losses_1616132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
"__inference__traced_restore_162122
file_prefix"
assignvariableop_dense1_kernel"
assignvariableop_1_dense1_bias#
assignvariableop_2_actor_kernel!
assignvariableop_3_actor_bias$
 assignvariableop_4_critic_kernel"
assignvariableop_5_critic_bias

identity_7??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
RestoreV2/shape_and_slices?
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

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_actor_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_actor_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp assignvariableop_4_critic_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_critic_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_6?

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
?
|
'__inference_Dense1_layer_call_fn_162014

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Dense1_layer_call_and_return_conditional_losses_1615872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_a2c_model_layer_call_fn_413
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1??StatefulPartitionedCall?
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
GPU 2J 8? *K
fFRD
B__inference_a2c_model_layer_call_and_return_conditional_losses_3612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:F B
#
_output_shapes
:?????????

_user_specified_namex
?
?
B__inference_a2c_model_layer_call_and_return_conditional_losses_572
input_16
2functional_1_dense1_matmul_readvariableop_resource7
3functional_1_dense1_biasadd_readvariableop_resource6
2functional_1_critic_matmul_readvariableop_resource7
3functional_1_critic_biasadd_readvariableop_resource5
1functional_1_actor_matmul_readvariableop_resource6
2functional_1_actor_biasadd_readvariableop_resource
identity

identity_1?b
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

ExpandDims?
)functional_1/Dense1/MatMul/ReadVariableOpReadVariableOp2functional_1_dense1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02+
)functional_1/Dense1/MatMul/ReadVariableOp?
functional_1/Dense1/MatMulMatMulExpandDims:output:01functional_1/Dense1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
functional_1/Dense1/MatMul?
*functional_1/Dense1/BiasAdd/ReadVariableOpReadVariableOp3functional_1_dense1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*functional_1/Dense1/BiasAdd/ReadVariableOp?
functional_1/Dense1/BiasAddBiasAdd$functional_1/Dense1/MatMul:product:02functional_1/Dense1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
functional_1/Dense1/BiasAdd?
functional_1/Dense1/ReluRelu$functional_1/Dense1/BiasAdd:output:0*
T0*
_output_shapes
:	?2
functional_1/Dense1/Relu?
)functional_1/Critic/MatMul/ReadVariableOpReadVariableOp2functional_1_critic_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02+
)functional_1/Critic/MatMul/ReadVariableOp?
functional_1/Critic/MatMulMatMul&functional_1/Dense1/Relu:activations:01functional_1/Critic/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Critic/MatMul?
*functional_1/Critic/BiasAdd/ReadVariableOpReadVariableOp3functional_1_critic_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*functional_1/Critic/BiasAdd/ReadVariableOp?
functional_1/Critic/BiasAddBiasAdd$functional_1/Critic/MatMul:product:02functional_1/Critic/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Critic/BiasAdd?
(functional_1/Actor/MatMul/ReadVariableOpReadVariableOp1functional_1_actor_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(functional_1/Actor/MatMul/ReadVariableOp?
functional_1/Actor/MatMulMatMul&functional_1/Dense1/Relu:activations:00functional_1/Actor/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Actor/MatMul?
)functional_1/Actor/BiasAdd/ReadVariableOpReadVariableOp2functional_1_actor_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)functional_1/Actor/BiasAdd/ReadVariableOp?
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
':?????????:::::::L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1
?	
?
-__inference_functional_1_layer_call_fn_161975

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_1617002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
B__inference_Critic_layer_call_and_return_conditional_losses_162043

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
E__inference_a2c_model_layer_call_and_return_conditional_losses_161851
x
a2c_model_161835
a2c_model_161837
a2c_model_161839
a2c_model_161841
a2c_model_161843
a2c_model_161845
identity

identity_1??!a2c_model/StatefulPartitionedCall?
!a2c_model/StatefulPartitionedCallStatefulPartitionedCallxa2c_model_161835a2c_model_161837a2c_model_161839a2c_model_161841a2c_model_161843a2c_model_161845*
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
GPU 2J 8? *2
f-R+
)__inference_restored_function_body_1614572#
!a2c_model/StatefulPartitionedCall?
IdentityIdentity*a2c_model/StatefulPartitionedCall:output:0"^a2c_model/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity?

Identity_1Identity*a2c_model/StatefulPartitionedCall:output:1"^a2c_model/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::::2F
!a2c_model/StatefulPartitionedCall!a2c_model/StatefulPartitionedCall:F B
#
_output_shapes
:?????????

_user_specified_namex
?
?
)__inference_restored_function_body_161418
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*(
_output_shapes
::*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_a2c_model_layer_call_and_return_conditional_losses_1362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:F B
#
_output_shapes
:?????????

_user_specified_namex
?
?
$__inference_signature_wrapper_161572
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1??StatefulPartitionedCall?
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
GPU 2J 8? **
f%R#
!__inference__wrapped_model_1614352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
E__inference_functional_1_layer_call_and_return_conditional_losses_333

inputs)
%dense1_matmul_readvariableop_resource*
&dense1_biasadd_readvariableop_resource)
%critic_matmul_readvariableop_resource*
&critic_biasadd_readvariableop_resource(
$actor_matmul_readvariableop_resource)
%actor_biasadd_readvariableop_resource
identity

identity_1??
Dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
Dense1/MatMul/ReadVariableOp?
Dense1/MatMulMatMulinputs$Dense1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
Dense1/MatMul?
Dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
Dense1/BiasAdd/ReadVariableOp?
Dense1/BiasAddBiasAddDense1/MatMul:product:0%Dense1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
Dense1/BiasAdde
Dense1/ReluReluDense1/BiasAdd:output:0*
T0*
_output_shapes
:	?2
Dense1/Relu?
Critic/MatMul/ReadVariableOpReadVariableOp%critic_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
Critic/MatMul/ReadVariableOp?
Critic/MatMulMatMulDense1/Relu:activations:0$Critic/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Critic/MatMul?
Critic/BiasAdd/ReadVariableOpReadVariableOp&critic_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
Critic/BiasAdd/ReadVariableOp?
Critic/BiasAddBiasAddCritic/MatMul:product:0%Critic/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Critic/BiasAdd?
Actor/MatMul/ReadVariableOpReadVariableOp$actor_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
Actor/MatMul/ReadVariableOp?
Actor/MatMulMatMulDense1/Relu:activations:0#Actor/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Actor/MatMul?
Actor/BiasAdd/ReadVariableOpReadVariableOp%actor_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
Actor/BiasAdd/ReadVariableOp?
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

identity_1Identity_1:output:0*>
_input_shapes-
+:?????????:::::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
E__inference_a2c_model_layer_call_and_return_conditional_losses_161515
x
a2c_model_161499
a2c_model_161501
a2c_model_161503
a2c_model_161505
a2c_model_161507
a2c_model_161509
identity

identity_1??!a2c_model/StatefulPartitionedCall?
!a2c_model/StatefulPartitionedCallStatefulPartitionedCallxa2c_model_161499a2c_model_161501a2c_model_161503a2c_model_161505a2c_model_161507a2c_model_161509*
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
GPU 2J 8? *2
f-R+
)__inference_restored_function_body_1614182#
!a2c_model/StatefulPartitionedCall?
IdentityIdentity*a2c_model/StatefulPartitionedCall:output:0"^a2c_model/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity?

Identity_1Identity*a2c_model/StatefulPartitionedCall:output:1"^a2c_model/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::::2F
!a2c_model/StatefulPartitionedCall!a2c_model/StatefulPartitionedCall:F B
#
_output_shapes
:?????????

_user_specified_namex
?	
?
*__inference_a2c_model_layer_call_fn_161889
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1??StatefulPartitionedCall?
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
GPU 2J 8? *N
fIRG
E__inference_a2c_model_layer_call_and_return_conditional_losses_1615152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:F B
#
_output_shapes
:?????????

_user_specified_namex
?
?
B__inference_Critic_layer_call_and_return_conditional_losses_161613

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
!__inference__wrapped_model_161435
input_1
a2c_model_a2c_model_161419
a2c_model_a2c_model_161421
a2c_model_a2c_model_161423
a2c_model_a2c_model_161425
a2c_model_a2c_model_161427
a2c_model_a2c_model_161429
identity

identity_1??+a2c_model/a2c_model/StatefulPartitionedCall?
+a2c_model/a2c_model/StatefulPartitionedCallStatefulPartitionedCallinput_1a2c_model_a2c_model_161419a2c_model_a2c_model_161421a2c_model_a2c_model_161423a2c_model_a2c_model_161425a2c_model_a2c_model_161427a2c_model_a2c_model_161429*
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
GPU 2J 8? *2
f-R+
)__inference_restored_function_body_1614182-
+a2c_model/a2c_model/StatefulPartitionedCall?
IdentityIdentity4a2c_model/a2c_model/StatefulPartitionedCall:output:0,^a2c_model/a2c_model/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity?

Identity_1Identity4a2c_model/a2c_model/StatefulPartitionedCall:output:1,^a2c_model/a2c_model/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::::2Z
+a2c_model/a2c_model/StatefulPartitionedCall+a2c_model/a2c_model/StatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1
?	
?
*__inference_a2c_model_layer_call_fn_161813
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1??StatefulPartitionedCall?
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
GPU 2J 8? *N
fIRG
E__inference_a2c_model_layer_call_and_return_conditional_losses_1615152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1
?	
?
-__inference_functional_1_layer_call_fn_161994

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_1617392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
'__inference_a2c_model_layer_call_fn_400
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1??StatefulPartitionedCall?
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
GPU 2J 8? *K
fFRD
B__inference_a2c_model_layer_call_and_return_conditional_losses_3612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
H__inference_functional_1_layer_call_and_return_conditional_losses_161932

inputs)
%dense1_matmul_readvariableop_resource*
&dense1_biasadd_readvariableop_resource)
%critic_matmul_readvariableop_resource*
&critic_biasadd_readvariableop_resource(
$actor_matmul_readvariableop_resource)
%actor_biasadd_readvariableop_resource
identity

identity_1??
Dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
Dense1/MatMul/ReadVariableOp?
Dense1/MatMulMatMulinputs$Dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Dense1/MatMul?
Dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
Dense1/BiasAdd/ReadVariableOp?
Dense1/BiasAddBiasAddDense1/MatMul:product:0%Dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Dense1/BiasAddn
Dense1/ReluReluDense1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Dense1/Relu?
Critic/MatMul/ReadVariableOpReadVariableOp%critic_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
Critic/MatMul/ReadVariableOp?
Critic/MatMulMatMulDense1/Relu:activations:0$Critic/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Critic/MatMul?
Critic/BiasAdd/ReadVariableOpReadVariableOp&critic_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
Critic/BiasAdd/ReadVariableOp?
Critic/BiasAddBiasAddCritic/MatMul:product:0%Critic/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Critic/BiasAdd?
Actor/MatMul/ReadVariableOpReadVariableOp$actor_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
Actor/MatMul/ReadVariableOp?
Actor/MatMulMatMulDense1/Relu:activations:0#Actor/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Actor/MatMul?
Actor/BiasAdd/ReadVariableOpReadVariableOp%actor_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
Actor/BiasAdd/ReadVariableOp?
Actor/BiasAddBiasAddActor/MatMul:product:0$Actor/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Actor/BiasAddj
IdentityIdentityActor/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identityo

Identity_1IdentityCritic/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:?????????:::::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
E__inference_a2c_model_layer_call_and_return_conditional_losses_161870
x
a2c_model_161854
a2c_model_161856
a2c_model_161858
a2c_model_161860
a2c_model_161862
a2c_model_161864
identity

identity_1??!a2c_model/StatefulPartitionedCall?
!a2c_model/StatefulPartitionedCallStatefulPartitionedCallxa2c_model_161854a2c_model_161856a2c_model_161858a2c_model_161860a2c_model_161862a2c_model_161864*
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
GPU 2J 8? *2
f-R+
)__inference_restored_function_body_1614182#
!a2c_model/StatefulPartitionedCall?
IdentityIdentity*a2c_model/StatefulPartitionedCall:output:0"^a2c_model/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity?

Identity_1Identity*a2c_model/StatefulPartitionedCall:output:1"^a2c_model/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::::2F
!a2c_model/StatefulPartitionedCall!a2c_model/StatefulPartitionedCall:F B
#
_output_shapes
:?????????

_user_specified_namex
?	
?
'__inference_a2c_model_layer_call_fn_387
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1??StatefulPartitionedCall?
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
GPU 2J 8? *K
fFRD
B__inference_a2c_model_layer_call_and_return_conditional_losses_3612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1
?

?
E__inference_a2c_model_layer_call_and_return_conditional_losses_161775
input_1
a2c_model_161759
a2c_model_161761
a2c_model_161763
a2c_model_161765
a2c_model_161767
a2c_model_161769
identity

identity_1??!a2c_model/StatefulPartitionedCall?
!a2c_model/StatefulPartitionedCallStatefulPartitionedCallinput_1a2c_model_161759a2c_model_161761a2c_model_161763a2c_model_161765a2c_model_161767a2c_model_161769*
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
GPU 2J 8? *2
f-R+
)__inference_restored_function_body_1614572#
!a2c_model/StatefulPartitionedCall?
IdentityIdentity*a2c_model/StatefulPartitionedCall:output:0"^a2c_model/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity?

Identity_1Identity*a2c_model/StatefulPartitionedCall:output:1"^a2c_model/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::::2F
!a2c_model/StatefulPartitionedCall!a2c_model/StatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1
?
{
&__inference_Actor_layer_call_fn_162033

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Actor_layer_call_and_return_conditional_losses_1616392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
B__inference_a2c_model_layer_call_and_return_conditional_losses_361
x
functional_1_89352
functional_1_89354
functional_1_89356
functional_1_89358
functional_1_89360
functional_1_89362
identity

identity_1??$functional_1/StatefulPartitionedCallb
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
:?????????2

ExpandDims?
$functional_1/StatefulPartitionedCallStatefulPartitionedCallExpandDims:output:0functional_1_89352functional_1_89354functional_1_89356functional_1_89358functional_1_89360functional_1_89362*
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
GPU 2J 8? *N
fIRG
E__inference_functional_1_layer_call_and_return_conditional_losses_3332&
$functional_1/StatefulPartitionedCall?
IdentityIdentity-functional_1/StatefulPartitionedCall:output:0%^functional_1/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity?

Identity_1Identity-functional_1/StatefulPartitionedCall:output:1%^functional_1/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::::2L
$functional_1/StatefulPartitionedCall$functional_1/StatefulPartitionedCall:F B
#
_output_shapes
:?????????

_user_specified_namex
?
?
A__inference_Actor_layer_call_and_return_conditional_losses_161639

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
H__inference_functional_1_layer_call_and_return_conditional_losses_161700

inputs
dense1_161683
dense1_161685
critic_161688
critic_161690
actor_161693
actor_161695
identity

identity_1??Actor/StatefulPartitionedCall?Critic/StatefulPartitionedCall?Dense1/StatefulPartitionedCall?
Dense1/StatefulPartitionedCallStatefulPartitionedCallinputsdense1_161683dense1_161685*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Dense1_layer_call_and_return_conditional_losses_1615872 
Dense1/StatefulPartitionedCall?
Critic/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0critic_161688critic_161690*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Critic_layer_call_and_return_conditional_losses_1616132 
Critic/StatefulPartitionedCall?
Actor/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0actor_161693actor_161695*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Actor_layer_call_and_return_conditional_losses_1616392
Actor/StatefulPartitionedCall?
IdentityIdentity&Actor/StatefulPartitionedCall:output:0^Actor/StatefulPartitionedCall^Critic/StatefulPartitionedCall^Dense1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity'Critic/StatefulPartitionedCall:output:0^Actor/StatefulPartitionedCall^Critic/StatefulPartitionedCall^Dense1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:?????????::::::2>
Actor/StatefulPartitionedCallActor/StatefulPartitionedCall2@
Critic/StatefulPartitionedCallCritic/StatefulPartitionedCall2@
Dense1/StatefulPartitionedCallDense1/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference__traced_save_162094
file_prefix,
(savev2_dense1_kernel_read_readvariableop*
&savev2_dense1_bias_read_readvariableop+
'savev2_actor_kernel_read_readvariableop)
%savev2_actor_bias_read_readvariableop,
(savev2_critic_kernel_read_readvariableop*
&savev2_critic_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_1fd7be9c772b4d2ab31ed44b4da7c0d7/part2	
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_dense1_kernel_read_readvariableop&savev2_dense1_bias_read_readvariableop'savev2_actor_kernel_read_readvariableop%savev2_actor_bias_read_readvariableop(savev2_critic_kernel_read_readvariableop&savev2_critic_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
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
8: :	?:?:	?::	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: 
?
?
H__inference_functional_1_layer_call_and_return_conditional_losses_161657
input_1
dense1_161598
dense1_161600
critic_161624
critic_161626
actor_161650
actor_161652
identity

identity_1??Actor/StatefulPartitionedCall?Critic/StatefulPartitionedCall?Dense1/StatefulPartitionedCall?
Dense1/StatefulPartitionedCallStatefulPartitionedCallinput_1dense1_161598dense1_161600*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Dense1_layer_call_and_return_conditional_losses_1615872 
Dense1/StatefulPartitionedCall?
Critic/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0critic_161624critic_161626*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Critic_layer_call_and_return_conditional_losses_1616132 
Critic/StatefulPartitionedCall?
Actor/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0actor_161650actor_161652*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Actor_layer_call_and_return_conditional_losses_1616392
Actor/StatefulPartitionedCall?
IdentityIdentity&Actor/StatefulPartitionedCall:output:0^Actor/StatefulPartitionedCall^Critic/StatefulPartitionedCall^Dense1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity'Critic/StatefulPartitionedCall:output:0^Actor/StatefulPartitionedCall^Critic/StatefulPartitionedCall^Dense1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:?????????::::::2>
Actor/StatefulPartitionedCallActor/StatefulPartitionedCall2@
Critic/StatefulPartitionedCallCritic/StatefulPartitionedCall2@
Dense1/StatefulPartitionedCallDense1/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
H__inference_functional_1_layer_call_and_return_conditional_losses_161677
input_1
dense1_161660
dense1_161662
critic_161665
critic_161667
actor_161670
actor_161672
identity

identity_1??Actor/StatefulPartitionedCall?Critic/StatefulPartitionedCall?Dense1/StatefulPartitionedCall?
Dense1/StatefulPartitionedCallStatefulPartitionedCallinput_1dense1_161660dense1_161662*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Dense1_layer_call_and_return_conditional_losses_1615872 
Dense1/StatefulPartitionedCall?
Critic/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0critic_161665critic_161667*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Critic_layer_call_and_return_conditional_losses_1616132 
Critic/StatefulPartitionedCall?
Actor/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0actor_161670actor_161672*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Actor_layer_call_and_return_conditional_losses_1616392
Actor/StatefulPartitionedCall?
IdentityIdentity&Actor/StatefulPartitionedCall:output:0^Actor/StatefulPartitionedCall^Critic/StatefulPartitionedCall^Dense1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity'Critic/StatefulPartitionedCall:output:0^Actor/StatefulPartitionedCall^Critic/StatefulPartitionedCall^Dense1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:?????????::::::2>
Actor/StatefulPartitionedCallActor/StatefulPartitionedCall2@
Critic/StatefulPartitionedCallCritic/StatefulPartitionedCall2@
Dense1/StatefulPartitionedCallDense1/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
'__inference_a2c_model_layer_call_fn_374
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1??StatefulPartitionedCall?
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
GPU 2J 8? *K
fFRD
B__inference_a2c_model_layer_call_and_return_conditional_losses_3612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:F B
#
_output_shapes
:?????????

_user_specified_namex
?	
?
*__inference_a2c_model_layer_call_fn_161832
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1??StatefulPartitionedCall?
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
GPU 2J 8? *N
fIRG
E__inference_a2c_model_layer_call_and_return_conditional_losses_1615152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
H__inference_functional_1_layer_call_and_return_conditional_losses_161739

inputs
dense1_161722
dense1_161724
critic_161727
critic_161729
actor_161732
actor_161734
identity

identity_1??Actor/StatefulPartitionedCall?Critic/StatefulPartitionedCall?Dense1/StatefulPartitionedCall?
Dense1/StatefulPartitionedCallStatefulPartitionedCallinputsdense1_161722dense1_161724*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Dense1_layer_call_and_return_conditional_losses_1615872 
Dense1/StatefulPartitionedCall?
Critic/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0critic_161727critic_161729*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Critic_layer_call_and_return_conditional_losses_1616132 
Critic/StatefulPartitionedCall?
Actor/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0actor_161732actor_161734*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Actor_layer_call_and_return_conditional_losses_1616392
Actor/StatefulPartitionedCall?
IdentityIdentity&Actor/StatefulPartitionedCall:output:0^Actor/StatefulPartitionedCall^Critic/StatefulPartitionedCall^Dense1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity'Critic/StatefulPartitionedCall:output:0^Actor/StatefulPartitionedCall^Critic/StatefulPartitionedCall^Dense1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:?????????::::::2>
Actor/StatefulPartitionedCallActor/StatefulPartitionedCall2@
Critic/StatefulPartitionedCallCritic/StatefulPartitionedCall2@
Dense1/StatefulPartitionedCallDense1/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
B__inference_Dense1_layer_call_and_return_conditional_losses_161587

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
7
input_1,
serving_default_input_1:0?????????3
output_1'
StatefulPartitionedCall:03
output_2'
StatefulPartitionedCall:1tensorflow/serving/predict:??
?
nn
trainable_variables
regularization_losses
	variables
	keras_api

signatures
K_default_save_signature
L__call__
*M&call_and_return_all_conditional_losses"?
_tf_keras_model?{"class_name": "a2c_model", "name": "a2c_model", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "a2c_model"}}
?
nn

signatures
#	_self_saveable_object_factories

trainable_variables
regularization_losses
	variables
	keras_api
N__call__
*O&call_and_return_all_conditional_losses"?
_tf_keras_model?{"class_name": "a2c_model", "name": "a2c_model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "a2c_model"}}
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
?
layer_regularization_losses
trainable_variables
regularization_losses
	variables
non_trainable_variables

layers
metrics
layer_metrics
L__call__
K_default_save_signature
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
,
Pserving_default"
signature_map
?!
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
#_self_saveable_object_factories
trainable_variables
regularization_losses
 	variables
!	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"?
_tf_keras_network?{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "Dense1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": 41}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Actor", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Actor", "inbound_nodes": [[["Dense1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Critic", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Critic", "inbound_nodes": [[["Dense1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["Actor", 0, 0], ["Critic", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "Dense1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": 41}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Actor", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Actor", "inbound_nodes": [[["Dense1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Critic", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Critic", "inbound_nodes": [[["Dense1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["Actor", 0, 0], ["Critic", 0, 0]]}}}
,
Sserving_default"
signature_map
 "
trackable_dict_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
?
"layer_regularization_losses

trainable_variables
regularization_losses
	variables
#non_trainable_variables

$layers
%metrics
&layer_metrics
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
 :	?2Dense1/kernel
:?2Dense1/bias
:	?2Actor/kernel
:2
Actor/bias
 :	?2Critic/kernel
:2Critic/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
#'_self_saveable_object_factories"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?

kernel
bias
#(_self_saveable_object_factories
)trainable_variables
*regularization_losses
+	variables
,	keras_api
T__call__
*U&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "Dense1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Dense1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": 41}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}}
?

kernel
bias
#-_self_saveable_object_factories
.trainable_variables
/regularization_losses
0	variables
1	keras_api
V__call__
*W&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "Actor", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Actor", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?

kernel
bias
#2_self_saveable_object_factories
3trainable_variables
4regularization_losses
5	variables
6	keras_api
X__call__
*Y&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "Critic", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Critic", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
 "
trackable_dict_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
?
7layer_regularization_losses
trainable_variables
regularization_losses
 	variables
8non_trainable_variables

9layers
:metrics
;layer_metrics
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
<layer_regularization_losses
)trainable_variables
*regularization_losses
+	variables
=non_trainable_variables

>layers
?metrics
@layer_metrics
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Alayer_regularization_losses
.trainable_variables
/regularization_losses
0	variables
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_metrics
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Flayer_regularization_losses
3trainable_variables
4regularization_losses
5	variables
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_metrics
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?2?
!__inference__wrapped_model_161435?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *"?
?
input_1?????????
?2?
*__inference_a2c_model_layer_call_fn_161908
*__inference_a2c_model_layer_call_fn_161813
*__inference_a2c_model_layer_call_fn_161889
*__inference_a2c_model_layer_call_fn_161832?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_a2c_model_layer_call_and_return_conditional_losses_161870
E__inference_a2c_model_layer_call_and_return_conditional_losses_161775
E__inference_a2c_model_layer_call_and_return_conditional_losses_161794
E__inference_a2c_model_layer_call_and_return_conditional_losses_161851?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_a2c_model_layer_call_fn_374
'__inference_a2c_model_layer_call_fn_413
'__inference_a2c_model_layer_call_fn_400
'__inference_a2c_model_layer_call_fn_387?
???
FullArgSpec
args?
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_a2c_model_layer_call_and_return_conditional_losses_536
B__inference_a2c_model_layer_call_and_return_conditional_losses_136
B__inference_a2c_model_layer_call_and_return_conditional_losses_598
B__inference_a2c_model_layer_call_and_return_conditional_losses_572?
???
FullArgSpec
args?
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
3B1
$__inference_signature_wrapper_161572input_1
?2?
-__inference_functional_1_layer_call_fn_161756
-__inference_functional_1_layer_call_fn_161975
-__inference_functional_1_layer_call_fn_161717
-__inference_functional_1_layer_call_fn_161994?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_functional_1_layer_call_and_return_conditional_losses_161932
H__inference_functional_1_layer_call_and_return_conditional_losses_161657
H__inference_functional_1_layer_call_and_return_conditional_losses_161956
H__inference_functional_1_layer_call_and_return_conditional_losses_161677?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
0B.
!__inference_signature_wrapper_637input_1
?2?
'__inference_Dense1_layer_call_fn_162014?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_Dense1_layer_call_and_return_conditional_losses_162005?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_Actor_layer_call_fn_162033?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_Actor_layer_call_and_return_conditional_losses_162024?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_Critic_layer_call_fn_162052?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_Critic_layer_call_and_return_conditional_losses_162043?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
A__inference_Actor_layer_call_and_return_conditional_losses_162024]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? z
&__inference_Actor_layer_call_fn_162033P0?-
&?#
!?
inputs??????????
? "???????????
B__inference_Critic_layer_call_and_return_conditional_losses_162043]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? {
'__inference_Critic_layer_call_fn_162052P0?-
&?#
!?
inputs??????????
? "???????????
B__inference_Dense1_layer_call_and_return_conditional_losses_162005]/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? {
'__inference_Dense1_layer_call_fn_162014P/?,
%?"
 ?
inputs?????????
? "????????????
!__inference__wrapped_model_161435?,?)
"?
?
input_1?????????
? "Q?N
%
output_1?
output_1
%
output_2?
output_2?
B__inference_a2c_model_layer_call_and_return_conditional_losses_136o*?'
 ?
?
x?????????
p 
? "9?6
/?,
?
0/0
?
0/1
? ?
E__inference_a2c_model_layer_call_and_return_conditional_losses_161775u0?-
&?#
?
input_1?????????
p
? "9?6
/?,
?
0/0
?
0/1
? ?
E__inference_a2c_model_layer_call_and_return_conditional_losses_161794u0?-
&?#
?
input_1?????????
p 
? "9?6
/?,
?
0/0
?
0/1
? ?
E__inference_a2c_model_layer_call_and_return_conditional_losses_161851o*?'
 ?
?
x?????????
p
? "9?6
/?,
?
0/0
?
0/1
? ?
E__inference_a2c_model_layer_call_and_return_conditional_losses_161870o*?'
 ?
?
x?????????
p 
? "9?6
/?,
?
0/0
?
0/1
? ?
B__inference_a2c_model_layer_call_and_return_conditional_losses_536o*?'
 ?
?
x?????????
p
? "9?6
/?,
?
0/0
?
0/1
? ?
B__inference_a2c_model_layer_call_and_return_conditional_losses_572u0?-
&?#
?
input_1?????????
p
? "9?6
/?,
?
0/0
?
0/1
? ?
B__inference_a2c_model_layer_call_and_return_conditional_losses_598u0?-
&?#
?
input_1?????????
p 
? "9?6
/?,
?
0/0
?
0/1
? ?
*__inference_a2c_model_layer_call_fn_161813g0?-
&?#
?
input_1?????????
p
? "+?(
?
0
?
1?
*__inference_a2c_model_layer_call_fn_161832g0?-
&?#
?
input_1?????????
p 
? "+?(
?
0
?
1?
*__inference_a2c_model_layer_call_fn_161889a*?'
 ?
?
x?????????
p
? "+?(
?
0
?
1?
*__inference_a2c_model_layer_call_fn_161908a*?'
 ?
?
x?????????
p 
? "+?(
?
0
?
1?
'__inference_a2c_model_layer_call_fn_374a*?'
 ?
?
x?????????
p
? "+?(
?
0
?
1?
'__inference_a2c_model_layer_call_fn_387g0?-
&?#
?
input_1?????????
p 
? "+?(
?
0
?
1?
'__inference_a2c_model_layer_call_fn_400g0?-
&?#
?
input_1?????????
p
? "+?(
?
0
?
1?
'__inference_a2c_model_layer_call_fn_413a*?'
 ?
?
x?????????
p 
? "+?(
?
0
?
1?
H__inference_functional_1_layer_call_and_return_conditional_losses_161657?8?5
.?+
!?
input_1?????????
p

 
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
H__inference_functional_1_layer_call_and_return_conditional_losses_161677?8?5
.?+
!?
input_1?????????
p 

 
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
H__inference_functional_1_layer_call_and_return_conditional_losses_161932?7?4
-?*
 ?
inputs?????????
p

 
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
H__inference_functional_1_layer_call_and_return_conditional_losses_161956?7?4
-?*
 ?
inputs?????????
p 

 
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
-__inference_functional_1_layer_call_fn_161717?8?5
.?+
!?
input_1?????????
p

 
? "=?:
?
0?????????
?
1??????????
-__inference_functional_1_layer_call_fn_161756?8?5
.?+
!?
input_1?????????
p 

 
? "=?:
?
0?????????
?
1??????????
-__inference_functional_1_layer_call_fn_161975?7?4
-?*
 ?
inputs?????????
p

 
? "=?:
?
0?????????
?
1??????????
-__inference_functional_1_layer_call_fn_161994?7?4
-?*
 ?
inputs?????????
p 

 
? "=?:
?
0?????????
?
1??????????
$__inference_signature_wrapper_161572?7?4
? 
-?*
(
input_1?
input_1?????????"Q?N
%
output_1?
output_1
%
output_2?
output_2?
!__inference_signature_wrapper_637?7?4
? 
-?*
(
input_1?
input_1?????????"Q?N
%
output_1?
output_1
%
output_2?
output_2