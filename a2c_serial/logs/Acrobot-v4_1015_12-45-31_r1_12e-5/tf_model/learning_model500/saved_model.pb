º
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
 "serve*2.3.02unknown8ýÞ
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

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ú
valueÐBÍ BÆ
j
nn
regularization_losses
	variables
trainable_variables
	keras_api

signatures

nn

signatures
#	_self_saveable_object_factories

regularization_losses
	variables
trainable_variables
	keras_api
 
*
0
1
2
3
4
5
*
0
1
2
3
4
5
­
layer_metrics
regularization_losses
metrics
	variables
non_trainable_variables
trainable_variables

layers
layer_regularization_losses
 

nn

signatures
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
 
 
 
*
0
1
2
3
4
5
*
0
1
2
3
4
5
­
 layer_metrics

regularization_losses
!metrics
	variables
"non_trainable_variables
trainable_variables

#layers
$layer_regularization_losses
IG
VARIABLE_VALUEDense1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEDense1/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEActor/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
FD
VARIABLE_VALUE
Actor/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUECritic/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUECritic/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

0
 
ù
%layer-0
&layer_with_weights-0
&layer-1
'layer_with_weights-1
'layer-2
(layer_with_weights-2
(layer-3
#)_self_saveable_object_factories
*regularization_losses
+	variables
,trainable_variables
-	keras_api
 
 
 
*
0
1
2
3
4
5
*
0
1
2
3
4
5
­
.layer_metrics
regularization_losses
/metrics
	variables
0non_trainable_variables
trainable_variables

1layers
2layer_regularization_losses
 
 
 

0
 
%
#3_self_saveable_object_factories


kernel
bias
#4_self_saveable_object_factories
5regularization_losses
6	variables
7trainable_variables
8	keras_api


kernel
bias
#9_self_saveable_object_factories
:regularization_losses
;	variables
<trainable_variables
=	keras_api


kernel
bias
#>_self_saveable_object_factories
?regularization_losses
@	variables
Atrainable_variables
B	keras_api
 
 
*
0
1
2
3
4
5
*
0
1
2
3
4
5
­
Clayer_metrics
*regularization_losses
Dmetrics
+	variables
Enon_trainable_variables
,trainable_variables

Flayers
Glayer_regularization_losses
 
 
 

0
 
 
 
 

0
1

0
1
­
Hlayer_metrics
5regularization_losses
Imetrics
6	variables
Jnon_trainable_variables
7trainable_variables

Klayers
Llayer_regularization_losses
 
 

0
1

0
1
­
Mlayer_metrics
:regularization_losses
Nmetrics
;	variables
Onon_trainable_variables
<trainable_variables

Players
Qlayer_regularization_losses
 
 

0
1

0
1
­
Rlayer_metrics
?regularization_losses
Smetrics
@	variables
Tnon_trainable_variables
Atrainable_variables

Ulayers
Vlayer_regularization_losses
 
 
 

%0
&1
'2
(3
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
&__inference_signature_wrapper_80413872
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
!__inference__traced_save_80414394
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
$__inference__traced_restore_80414422½­
Ü
}
(__inference_Actor_layer_call_fn_80414333

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
C__inference_Actor_layer_call_and_return_conditional_losses_804139392
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
ß
Ç
+__inference_restored_function_body_15881820
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1¢StatefulPartitionedCall
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
GPU 2J 8 *K
fFRD
B__inference_a2c_model_layer_call_and_return_conditional_losses_4742
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
Í

¥
G__inference_a2c_model_layer_call_and_return_conditional_losses_80413815
x
a2c_model_80413799
a2c_model_80413801
a2c_model_80413803
a2c_model_80413805
a2c_model_80413807
a2c_model_80413809
identity

identity_1¢!a2c_model/StatefulPartitionedCallØ
!a2c_model/StatefulPartitionedCallStatefulPartitionedCallxa2c_model_80413799a2c_model_80413801a2c_model_80413803a2c_model_80413805a2c_model_80413807a2c_model_80413809*
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
GPU 2J 8 *4
f/R-
+__inference_restored_function_body_158818202#
!a2c_model/StatefulPartitionedCall
IdentityIdentity*a2c_model/StatefulPartitionedCall:output:0"^a2c_model/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity

Identity_1Identity*a2c_model/StatefulPartitionedCall:output:1"^a2c_model/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::::2F
!a2c_model/StatefulPartitionedCall!a2c_model/StatefulPartitionedCall:F B
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
	
È
,__inference_a2c_model_layer_call_fn_80414189
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
G__inference_a2c_model_layer_call_and_return_conditional_losses_804138152
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
Ú
Â
&__inference_restored_function_body_656
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1¢StatefulPartitionedCall
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
GPU 2J 8 *K
fFRD
B__inference_a2c_model_layer_call_and_return_conditional_losses_6432
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
½


B__inference_a2c_model_layer_call_and_return_conditional_losses_669
input_1
a2c_model_161759
a2c_model_161761
a2c_model_161763
a2c_model_161765
a2c_model_161767
a2c_model_161769
identity

identity_1¢!a2c_model/StatefulPartitionedCallÍ
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
GPU 2J 8 */
f*R(
&__inference_restored_function_body_6562#
!a2c_model/StatefulPartitionedCall
IdentityIdentity*a2c_model/StatefulPartitionedCall:output:0"^a2c_model/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity

Identity_1Identity*a2c_model/StatefulPartitionedCall:output:1"^a2c_model/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::::2F
!a2c_model/StatefulPartitionedCall!a2c_model/StatefulPartitionedCall:L H
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Þ
~
)__inference_Dense1_layer_call_fn_80414314

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
D__inference_Dense1_layer_call_and_return_conditional_losses_804138872
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
	
Î
,__inference_a2c_model_layer_call_fn_80414132
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
G__inference_a2c_model_layer_call_and_return_conditional_losses_804138152
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
È	
Ñ
/__inference_functional_1_layer_call_fn_80414056
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
J__inference_functional_1_layer_call_and_return_conditional_losses_804140392
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
¨
ù
%__forward_restored_function_body_1170
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1
statefulpartitionedcall
statefulpartitionedcall_0
statefulpartitionedcall_1
statefulpartitionedcall_2
statefulpartitionedcall_3
statefulpartitionedcall_4¢StatefulPartitionedCallÎ
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout

2*m
_output_shapes[
Y:::	:	:	:	::ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__forward_a2c_model_layer_call_and_return_conditional_losses_11422
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

identity_1Identity_1:output:0";
statefulpartitionedcall StatefulPartitionedCall:output:2"=
statefulpartitionedcall_0 StatefulPartitionedCall:output:3"=
statefulpartitionedcall_1 StatefulPartitionedCall:output:4"=
statefulpartitionedcall_2 StatefulPartitionedCall:output:5"=
statefulpartitionedcall_3 StatefulPartitionedCall:output:6"=
statefulpartitionedcall_4 StatefulPartitionedCall:output:7*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::::*S
backward_function_name97__inference___backward_restored_function_body_1104_117122
StatefulPartitionedCallStatefulPartitionedCall:F B
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
	
É
'__inference_a2c_model_layer_call_fn_526
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1¢StatefulPartitionedCall©
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
GPU 2J 8 *K
fFRD
B__inference_a2c_model_layer_call_and_return_conditional_losses_5132
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
«


B__inference_a2c_model_layer_call_and_return_conditional_losses_682
x
a2c_model_161835
a2c_model_161837
a2c_model_161839
a2c_model_161841
a2c_model_161843
a2c_model_161845
identity

identity_1¢!a2c_model/StatefulPartitionedCallÇ
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
GPU 2J 8 */
f*R(
&__inference_restored_function_body_6562#
!a2c_model/StatefulPartitionedCall
IdentityIdentity*a2c_model/StatefulPartitionedCall:output:0"^a2c_model/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity

Identity_1Identity*a2c_model/StatefulPartitionedCall:output:1"^a2c_model/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::::2F
!a2c_model/StatefulPartitionedCall!a2c_model/StatefulPartitionedCall:F B
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex

ü
J__inference_functional_1_layer_call_and_return_conditional_losses_80414232

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
	
É
'__inference_a2c_model_layer_call_fn_289
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1¢StatefulPartitionedCall©
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
GPU 2J 8 *K
fFRD
B__inference_a2c_model_layer_call_and_return_conditional_losses_2632
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
«
×
J__inference_functional_1_layer_call_and_return_conditional_losses_80414000

inputs
dense1_80413983
dense1_80413985
critic_80413988
critic_80413990
actor_80413993
actor_80413995
identity

identity_1¢Actor/StatefulPartitionedCall¢Critic/StatefulPartitionedCall¢Dense1/StatefulPartitionedCall
Dense1/StatefulPartitionedCallStatefulPartitionedCallinputsdense1_80413983dense1_80413985*
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
D__inference_Dense1_layer_call_and_return_conditional_losses_804138872 
Dense1/StatefulPartitionedCall±
Critic/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0critic_80413988critic_80413990*
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
D__inference_Critic_layer_call_and_return_conditional_losses_804139132 
Critic/StatefulPartitionedCall¬
Actor/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0actor_80413993actor_80413995*
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
C__inference_Actor_layer_call_and_return_conditional_losses_804139392
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
È	
Ñ
/__inference_functional_1_layer_call_fn_80414017
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
J__inference_functional_1_layer_call_and_return_conditional_losses_804140002
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
ú
Ã
'__inference_a2c_model_layer_call_fn_315
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1¢StatefulPartitionedCall£
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
GPU 2J 8 *K
fFRD
B__inference_a2c_model_layer_call_and_return_conditional_losses_2632
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
Ä
÷
E__inference_functional_1_layer_call_and_return_conditional_losses_248

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

identity_1Identity_1:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ:::::::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¡
 
$__inference__traced_restore_80414422
file_prefix"
assignvariableop_dense1_kernel"
assignvariableop_1_dense1_bias#
assignvariableop_2_actor_kernel!
assignvariableop_3_actor_bias$
 assignvariableop_4_critic_kernel"
assignvariableop_5_critic_bias

identity_7¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*£
valueBB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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
Ä
Ã
B__inference_a2c_model_layer_call_and_return_conditional_losses_351
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
ß

«
G__inference_a2c_model_layer_call_and_return_conditional_losses_80414094
input_1
a2c_model_80414078
a2c_model_80414080
a2c_model_80414082
a2c_model_80414084
a2c_model_80414086
a2c_model_80414088
identity

identity_1¢!a2c_model/StatefulPartitionedCallÞ
!a2c_model/StatefulPartitionedCallStatefulPartitionedCallinput_1a2c_model_80414078a2c_model_80414080a2c_model_80414082a2c_model_80414084a2c_model_80414086a2c_model_80414088*
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
GPU 2J 8 *4
f/R-
+__inference_restored_function_body_158818202#
!a2c_model/StatefulPartitionedCall
IdentityIdentity*a2c_model/StatefulPartitionedCall:output:0"^a2c_model/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity

Identity_1Identity*a2c_model/StatefulPartitionedCall:output:1"^a2c_model/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::::2F
!a2c_model/StatefulPartitionedCall!a2c_model/StatefulPartitionedCall:L H
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
²
½
B__inference_a2c_model_layer_call_and_return_conditional_losses_448
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
Å	
Ð
/__inference_functional_1_layer_call_fn_80414294

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
J__inference_functional_1_layer_call_and_return_conditional_losses_804140392
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
Þ
~
)__inference_Critic_layer_call_fn_80414352

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
D__inference_Critic_layer_call_and_return_conditional_losses_804139132
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
ú
Ã
'__inference_a2c_model_layer_call_fn_276
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1¢StatefulPartitionedCall£
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
GPU 2J 8 *K
fFRD
B__inference_a2c_model_layer_call_and_return_conditional_losses_2632
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
	
Î
,__inference_a2c_model_layer_call_fn_80414113
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
G__inference_a2c_model_layer_call_and_return_conditional_losses_804138152
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
	
È
,__inference_a2c_model_layer_call_fn_80414208
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
G__inference_a2c_model_layer_call_and_return_conditional_losses_804138152
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
é
£
B__inference_a2c_model_layer_call_and_return_conditional_losses_263
x
functional_1_89352
functional_1_89354
functional_1_89356
functional_1_89358
functional_1_89360
functional_1_89362
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

ExpandDims
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
GPU 2J 8 *N
fIRG
E__inference_functional_1_layer_call_and_return_conditional_losses_2482&
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
ß
Ç
+__inference_restored_function_body_15881859
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1¢StatefulPartitionedCall
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
GPU 2J 8 *K
fFRD
B__inference_a2c_model_layer_call_and_return_conditional_losses_6692
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
Ä
Ã
B__inference_a2c_model_layer_call_and_return_conditional_losses_708
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
ß

«
G__inference_a2c_model_layer_call_and_return_conditional_losses_80414075
input_1
a2c_model_80414059
a2c_model_80414061
a2c_model_80414063
a2c_model_80414065
a2c_model_80414067
a2c_model_80414069
identity

identity_1¢!a2c_model/StatefulPartitionedCallÞ
!a2c_model/StatefulPartitionedCallStatefulPartitionedCallinput_1a2c_model_80414059a2c_model_80414061a2c_model_80414063a2c_model_80414065a2c_model_80414067a2c_model_80414069*
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
GPU 2J 8 *4
f/R-
+__inference_restored_function_body_158818592#
!a2c_model/StatefulPartitionedCall
IdentityIdentity*a2c_model/StatefulPartitionedCall:output:0"^a2c_model/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity

Identity_1Identity*a2c_model/StatefulPartitionedCall:output:1"^a2c_model/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::::2F
!a2c_model/StatefulPartitionedCall!a2c_model/StatefulPartitionedCall:L H
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
â
Ã
!__inference_signature_wrapper_500
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1¢StatefulPartitionedCall
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
GPU 2J 8 *'
f"R 
__inference__wrapped_model_4872
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

ú
!__inference__traced_save_80414394
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
value3B1 B+_temp_bef2223e7d914b56be17b8a4c7aabfeb/part2	
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
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*£
valueBB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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
	
É
'__inference_a2c_model_layer_call_fn_539
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1¢StatefulPartitionedCall©
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
GPU 2J 8 *K
fFRD
B__inference_a2c_model_layer_call_and_return_conditional_losses_5132
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

ü
J__inference_functional_1_layer_call_and_return_conditional_losses_80414256

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
Í

¥
G__inference_a2c_model_layer_call_and_return_conditional_losses_80414170
x
a2c_model_80414154
a2c_model_80414156
a2c_model_80414158
a2c_model_80414160
a2c_model_80414162
a2c_model_80414164
identity

identity_1¢!a2c_model/StatefulPartitionedCallØ
!a2c_model/StatefulPartitionedCallStatefulPartitionedCallxa2c_model_80414154a2c_model_80414156a2c_model_80414158a2c_model_80414160a2c_model_80414162a2c_model_80414164*
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
GPU 2J 8 *4
f/R-
+__inference_restored_function_body_158818202#
!a2c_model/StatefulPartitionedCall
IdentityIdentity*a2c_model/StatefulPartitionedCall:output:0"^a2c_model/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity

Identity_1Identity*a2c_model/StatefulPartitionedCall:output:1"^a2c_model/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::::2F
!a2c_model/StatefulPartitionedCall!a2c_model/StatefulPartitionedCall:F B
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
Ð
¬
D__inference_Critic_layer_call_and_return_conditional_losses_80413913

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
½


B__inference_a2c_model_layer_call_and_return_conditional_losses_578
input_1
a2c_model_161778
a2c_model_161780
a2c_model_161782
a2c_model_161784
a2c_model_161786
a2c_model_161788
identity

identity_1¢!a2c_model/StatefulPartitionedCallÍ
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
GPU 2J 8 */
f*R(
&__inference_restored_function_body_4612#
!a2c_model/StatefulPartitionedCall
IdentityIdentity*a2c_model/StatefulPartitionedCall:output:0"^a2c_model/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity

Identity_1Identity*a2c_model/StatefulPartitionedCall:output:1"^a2c_model/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::::2F
!a2c_model/StatefulPartitionedCall!a2c_model/StatefulPartitionedCall:L H
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ö!
ÿ
A__forward_a2c_model_layer_call_and_return_conditional_losses_1142
x_06
2functional_1_dense1_matmul_readvariableop_resource7
3functional_1_dense1_biasadd_readvariableop_resource6
2functional_1_critic_matmul_readvariableop_resource7
3functional_1_critic_biasadd_readvariableop_resource5
1functional_1_actor_matmul_readvariableop_resource6
2functional_1_actor_biasadd_readvariableop_resource
identity

identity_1,
(functional_1_actor_matmul_readvariableop
functional_1_dense1_relu-
)functional_1_critic_matmul_readvariableop-
)functional_1_dense1_matmul_readvariableop

expanddims
xb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dimm

ExpandDims
ExpandDimsx_0ExpandDims/dim:output:0*
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

Identity_1"!

expanddimsExpandDims:output:0"\
(functional_1_actor_matmul_readvariableop0functional_1/Actor/MatMul/ReadVariableOp:value:0"^
)functional_1_critic_matmul_readvariableop1functional_1/Critic/MatMul/ReadVariableOp:value:0"^
)functional_1_dense1_matmul_readvariableop1functional_1/Dense1/MatMul/ReadVariableOp:value:0"B
functional_1_dense1_relu&functional_1/Dense1/Relu:activations:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"
xx_0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::::*o
backward_function_nameUS__inference___backward_a2c_model_layer_call_and_return_conditional_losses_1110_1143:F B
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
«
×
J__inference_functional_1_layer_call_and_return_conditional_losses_80414039

inputs
dense1_80414022
dense1_80414024
critic_80414027
critic_80414029
actor_80414032
actor_80414034
identity

identity_1¢Actor/StatefulPartitionedCall¢Critic/StatefulPartitionedCall¢Dense1/StatefulPartitionedCall
Dense1/StatefulPartitionedCallStatefulPartitionedCallinputsdense1_80414022dense1_80414024*
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
D__inference_Dense1_layer_call_and_return_conditional_losses_804138872 
Dense1/StatefulPartitionedCall±
Critic/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0critic_80414027critic_80414029*
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
D__inference_Critic_layer_call_and_return_conditional_losses_804139132 
Critic/StatefulPartitionedCall¬
Actor/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0actor_80414032actor_80414034*
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
C__inference_Actor_layer_call_and_return_conditional_losses_804139392
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
ú
Ã
'__inference_a2c_model_layer_call_fn_565
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1¢StatefulPartitionedCall£
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
GPU 2J 8 *K
fFRD
B__inference_a2c_model_layer_call_and_return_conditional_losses_5132
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

Í
#__inference__wrapped_model_80413754
input_1 
a2c_model_a2c_model_80413738 
a2c_model_a2c_model_80413740 
a2c_model_a2c_model_80413742 
a2c_model_a2c_model_80413744 
a2c_model_a2c_model_80413746 
a2c_model_a2c_model_80413748
identity

identity_1¢+a2c_model/a2c_model/StatefulPartitionedCall®
+a2c_model/a2c_model/StatefulPartitionedCallStatefulPartitionedCallinput_1a2c_model_a2c_model_80413738a2c_model_a2c_model_80413740a2c_model_a2c_model_80413742a2c_model_a2c_model_80413744a2c_model_a2c_model_80413746a2c_model_a2c_model_80413748*
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
GPU 2J 8 *4
f/R-
+__inference_restored_function_body_158818202-
+a2c_model/a2c_model/StatefulPartitionedCall­
IdentityIdentity4a2c_model/a2c_model/StatefulPartitionedCall:output:0,^a2c_model/a2c_model/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity±

Identity_1Identity4a2c_model/a2c_model/StatefulPartitionedCall:output:1,^a2c_model/a2c_model/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::::2Z
+a2c_model/a2c_model/StatefulPartitionedCall+a2c_model/a2c_model/StatefulPartitionedCall:L H
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
²
½
B__inference_a2c_model_layer_call_and_return_conditional_losses_643
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
â
Ã
!__inference_signature_wrapper_617
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1¢StatefulPartitionedCall
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
GPU 2J 8 *'
f"R 
__inference__wrapped_model_6042
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
/__inference_functional_1_layer_call_fn_80414275

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
J__inference_functional_1_layer_call_and_return_conditional_losses_804140002
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
å 
Û
__inference__wrapped_model_604
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
Ú
Â
&__inference_restored_function_body_461
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1¢StatefulPartitionedCall
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
GPU 2J 8 *K
fFRD
B__inference_a2c_model_layer_call_and_return_conditional_losses_4482
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
D__inference_Critic_layer_call_and_return_conditional_losses_80414343

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
C__inference_Actor_layer_call_and_return_conditional_losses_80414324

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
®
Ø
J__inference_functional_1_layer_call_and_return_conditional_losses_80413957
input_1
dense1_80413898
dense1_80413900
critic_80413924
critic_80413926
actor_80413950
actor_80413952
identity

identity_1¢Actor/StatefulPartitionedCall¢Critic/StatefulPartitionedCall¢Dense1/StatefulPartitionedCall
Dense1/StatefulPartitionedCallStatefulPartitionedCallinput_1dense1_80413898dense1_80413900*
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
D__inference_Dense1_layer_call_and_return_conditional_losses_804138872 
Dense1/StatefulPartitionedCall±
Critic/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0critic_80413924critic_80413926*
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
D__inference_Critic_layer_call_and_return_conditional_losses_804139132 
Critic/StatefulPartitionedCall¬
Actor/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0actor_80413950actor_80413952*
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
C__inference_Actor_layer_call_and_return_conditional_losses_804139392
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
Ï
«
C__inference_Actor_layer_call_and_return_conditional_losses_80413939

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
«


B__inference_a2c_model_layer_call_and_return_conditional_losses_474
x
a2c_model_161854
a2c_model_161856
a2c_model_161858
a2c_model_161860
a2c_model_161862
a2c_model_161864
identity

identity_1¢!a2c_model/StatefulPartitionedCallÇ
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
GPU 2J 8 */
f*R(
&__inference_restored_function_body_4612#
!a2c_model/StatefulPartitionedCall
IdentityIdentity*a2c_model/StatefulPartitionedCall:output:0"^a2c_model/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity

Identity_1Identity*a2c_model/StatefulPartitionedCall:output:1"^a2c_model/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::::2F
!a2c_model/StatefulPartitionedCall!a2c_model/StatefulPartitionedCall:F B
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
¯
¬
D__inference_Dense1_layer_call_and_return_conditional_losses_80414305

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
ú
Ã
'__inference_a2c_model_layer_call_fn_552
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1¢StatefulPartitionedCall£
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
GPU 2J 8 *K
fFRD
B__inference_a2c_model_layer_call_and_return_conditional_losses_5132
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
ì
È
&__inference_signature_wrapper_80413872
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
#__inference__wrapped_model_804137542
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
ë
¼
__inference__wrapped_model_487
input_1
a2c_model_a2c_model_161419
a2c_model_a2c_model_161421
a2c_model_a2c_model_161423
a2c_model_a2c_model_161425
a2c_model_a2c_model_161427
a2c_model_a2c_model_161429
identity

identity_1¢+a2c_model/a2c_model/StatefulPartitionedCall
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
GPU 2J 8 */
f*R(
&__inference_restored_function_body_4612-
+a2c_model/a2c_model/StatefulPartitionedCall­
IdentityIdentity4a2c_model/a2c_model/StatefulPartitionedCall:output:0,^a2c_model/a2c_model/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity±

Identity_1Identity4a2c_model/a2c_model/StatefulPartitionedCall:output:1,^a2c_model/a2c_model/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::::2Z
+a2c_model/a2c_model/StatefulPartitionedCall+a2c_model/a2c_model/StatefulPartitionedCall:L H
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Í

¥
G__inference_a2c_model_layer_call_and_return_conditional_losses_80414151
x
a2c_model_80414135
a2c_model_80414137
a2c_model_80414139
a2c_model_80414141
a2c_model_80414143
a2c_model_80414145
identity

identity_1¢!a2c_model/StatefulPartitionedCallØ
!a2c_model/StatefulPartitionedCallStatefulPartitionedCallxa2c_model_80414135a2c_model_80414137a2c_model_80414139a2c_model_80414141a2c_model_80414143a2c_model_80414145*
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
GPU 2J 8 *4
f/R-
+__inference_restored_function_body_158818592#
!a2c_model/StatefulPartitionedCall
IdentityIdentity*a2c_model/StatefulPartitionedCall:output:0"^a2c_model/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity

Identity_1Identity*a2c_model/StatefulPartitionedCall:output:1"^a2c_model/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::::2F
!a2c_model/StatefulPartitionedCall!a2c_model/StatefulPartitionedCall:F B
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
«


B__inference_a2c_model_layer_call_and_return_conditional_losses_513
x
a2c_model_161499
a2c_model_161501
a2c_model_161503
a2c_model_161505
a2c_model_161507
a2c_model_161509
identity

identity_1¢!a2c_model/StatefulPartitionedCallÇ
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
GPU 2J 8 */
f*R(
&__inference_restored_function_body_4612#
!a2c_model/StatefulPartitionedCall
IdentityIdentity*a2c_model/StatefulPartitionedCall:output:0"^a2c_model/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity

Identity_1Identity*a2c_model/StatefulPartitionedCall:output:1"^a2c_model/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::::2F
!a2c_model/StatefulPartitionedCall!a2c_model/StatefulPartitionedCall:F B
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
¯
¬
D__inference_Dense1_layer_call_and_return_conditional_losses_80413887

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
®
Ø
J__inference_functional_1_layer_call_and_return_conditional_losses_80413977
input_1
dense1_80413960
dense1_80413962
critic_80413965
critic_80413967
actor_80413970
actor_80413972
identity

identity_1¢Actor/StatefulPartitionedCall¢Critic/StatefulPartitionedCall¢Dense1/StatefulPartitionedCall
Dense1/StatefulPartitionedCallStatefulPartitionedCallinput_1dense1_80413960dense1_80413962*
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
D__inference_Dense1_layer_call_and_return_conditional_losses_804138872 
Dense1/StatefulPartitionedCall±
Critic/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0critic_80413965critic_80413967*
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
D__inference_Critic_layer_call_and_return_conditional_losses_804139132 
Critic/StatefulPartitionedCall¬
Actor/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0actor_80413970actor_80413972*
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
C__inference_Actor_layer_call_and_return_conditional_losses_804139392
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
	
É
'__inference_a2c_model_layer_call_fn_302
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1¢StatefulPartitionedCall©
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
GPU 2J 8 *K
fFRD
B__inference_a2c_model_layer_call_and_return_conditional_losses_2632
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
StatefulPartitionedCall:1tensorflow/serving/predict:¿Ó
Á
nn
regularization_losses
	variables
trainable_variables
	keras_api

signatures
W_default_save_signature
X__call__
*Y&call_and_return_all_conditional_losses"ý
_tf_keras_modelã{"class_name": "a2c_model", "name": "a2c_model", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "a2c_model"}}
È
nn

signatures
#	_self_saveable_object_factories

regularization_losses
	variables
trainable_variables
	keras_api
Z__call__
*[&call_and_return_all_conditional_losses"ü
_tf_keras_modelâ{"class_name": "a2c_model", "name": "a2c_model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "a2c_model"}}
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
J
0
1
2
3
4
5"
trackable_list_wrapper
Ê
layer_metrics
regularization_losses
metrics
	variables
non_trainable_variables
trainable_variables

layers
layer_regularization_losses
X__call__
W_default_save_signature
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
,
\serving_default"
signature_map
È
nn

signatures
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
]__call__
*^&call_and_return_all_conditional_losses"ü
_tf_keras_modelâ{"class_name": "a2c_model", "name": "a2c_model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "a2c_model"}}
,
_serving_default"
signature_map
 "
trackable_dict_wrapper
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
J
0
1
2
3
4
5"
trackable_list_wrapper
­
 layer_metrics

regularization_losses
!metrics
	variables
"non_trainable_variables
trainable_variables

#layers
$layer_regularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
 :	2Dense1/kernel
:2Dense1/bias
:	2Actor/kernel
:2
Actor/bias
 :	2Critic/kernel
:2Critic/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
ò!
%layer-0
&layer_with_weights-0
&layer-1
'layer_with_weights-1
'layer-2
(layer_with_weights-2
(layer-3
#)_self_saveable_object_factories
*regularization_losses
+	variables
,trainable_variables
-	keras_api
`__call__
*a&call_and_return_all_conditional_losses"¼
_tf_keras_network {"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "Dense1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": 41}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Actor", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Actor", "inbound_nodes": [[["Dense1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Critic", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Critic", "inbound_nodes": [[["Dense1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["Actor", 0, 0], ["Critic", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "Dense1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": 41}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Actor", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Actor", "inbound_nodes": [[["Dense1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Critic", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Critic", "inbound_nodes": [[["Dense1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["Actor", 0, 0], ["Critic", 0, 0]]}}}
,
bserving_default"
signature_map
 "
trackable_dict_wrapper
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
J
0
1
2
3
4
5"
trackable_list_wrapper
­
.layer_metrics
regularization_losses
/metrics
	variables
0non_trainable_variables
trainable_variables

1layers
2layer_regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper

#3_self_saveable_object_factories"æ
_tf_keras_input_layerÆ{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}


kernel
bias
#4_self_saveable_object_factories
5regularization_losses
6	variables
7trainable_variables
8	keras_api
c__call__
*d&call_and_return_all_conditional_losses"Á
_tf_keras_layer§{"class_name": "Dense", "name": "Dense1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Dense1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": 41}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}}


kernel
bias
#9_self_saveable_object_factories
:regularization_losses
;	variables
<trainable_variables
=	keras_api
e__call__
*f&call_and_return_all_conditional_losses"Ê
_tf_keras_layer°{"class_name": "Dense", "name": "Actor", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Actor", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}


kernel
bias
#>_self_saveable_object_factories
?regularization_losses
@	variables
Atrainable_variables
B	keras_api
g__call__
*h&call_and_return_all_conditional_losses"Ì
_tf_keras_layer²{"class_name": "Dense", "name": "Critic", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Critic", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
 "
trackable_dict_wrapper
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
J
0
1
2
3
4
5"
trackable_list_wrapper
­
Clayer_metrics
*regularization_losses
Dmetrics
+	variables
Enon_trainable_variables
,trainable_variables

Flayers
Glayer_regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
Hlayer_metrics
5regularization_losses
Imetrics
6	variables
Jnon_trainable_variables
7trainable_variables

Klayers
Llayer_regularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
Mlayer_metrics
:regularization_losses
Nmetrics
;	variables
Onon_trainable_variables
<trainable_variables

Players
Qlayer_regularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
Rlayer_metrics
?regularization_losses
Smetrics
@	variables
Tnon_trainable_variables
Atrainable_variables

Ulayers
Vlayer_regularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
%0
&1
'2
(3"
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
Ý2Ú
#__inference__wrapped_model_80413754²
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
ì2é
,__inference_a2c_model_layer_call_fn_80414132
,__inference_a2c_model_layer_call_fn_80414113
,__inference_a2c_model_layer_call_fn_80414208
,__inference_a2c_model_layer_call_fn_80414189®
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
G__inference_a2c_model_layer_call_and_return_conditional_losses_80414094
G__inference_a2c_model_layer_call_and_return_conditional_losses_80414075
G__inference_a2c_model_layer_call_and_return_conditional_losses_80414170
G__inference_a2c_model_layer_call_and_return_conditional_losses_80414151®
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
Î2Ë
'__inference_a2c_model_layer_call_fn_565
'__inference_a2c_model_layer_call_fn_539
'__inference_a2c_model_layer_call_fn_552
'__inference_a2c_model_layer_call_fn_526¤
²
FullArgSpec
args
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
annotationsª *
 
º2·
B__inference_a2c_model_layer_call_and_return_conditional_losses_474
B__inference_a2c_model_layer_call_and_return_conditional_losses_669
B__inference_a2c_model_layer_call_and_return_conditional_losses_578
B__inference_a2c_model_layer_call_and_return_conditional_losses_682¤
²
FullArgSpec
args
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
annotationsª *
 
5B3
&__inference_signature_wrapper_80413872input_1
Î2Ë
'__inference_a2c_model_layer_call_fn_276
'__inference_a2c_model_layer_call_fn_315
'__inference_a2c_model_layer_call_fn_302
'__inference_a2c_model_layer_call_fn_289¤
²
FullArgSpec
args
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
annotationsª *
 
º2·
B__inference_a2c_model_layer_call_and_return_conditional_losses_643
B__inference_a2c_model_layer_call_and_return_conditional_losses_448
B__inference_a2c_model_layer_call_and_return_conditional_losses_708
B__inference_a2c_model_layer_call_and_return_conditional_losses_351¤
²
FullArgSpec
args
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
annotationsª *
 
0B.
!__inference_signature_wrapper_500input_1
2
/__inference_functional_1_layer_call_fn_80414056
/__inference_functional_1_layer_call_fn_80414017
/__inference_functional_1_layer_call_fn_80414294
/__inference_functional_1_layer_call_fn_80414275À
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
ö2ó
J__inference_functional_1_layer_call_and_return_conditional_losses_80413977
J__inference_functional_1_layer_call_and_return_conditional_losses_80414256
J__inference_functional_1_layer_call_and_return_conditional_losses_80414232
J__inference_functional_1_layer_call_and_return_conditional_losses_80413957À
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
0B.
!__inference_signature_wrapper_617input_1
Ó2Ð
)__inference_Dense1_layer_call_fn_80414314¢
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
D__inference_Dense1_layer_call_and_return_conditional_losses_80414305¢
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
(__inference_Actor_layer_call_fn_80414333¢
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
C__inference_Actor_layer_call_and_return_conditional_losses_80414324¢
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
)__inference_Critic_layer_call_fn_80414352¢
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
D__inference_Critic_layer_call_and_return_conditional_losses_80414343¢
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
C__inference_Actor_layer_call_and_return_conditional_losses_80414324]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
(__inference_Actor_layer_call_fn_80414333P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¥
D__inference_Critic_layer_call_and_return_conditional_losses_80414343]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
)__inference_Critic_layer_call_fn_80414352P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¥
D__inference_Dense1_layer_call_and_return_conditional_losses_80414305]/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 }
)__inference_Dense1_layer_call_fn_80414314P/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ±
#__inference__wrapped_model_80413754,¢)
"¢

input_1ÿÿÿÿÿÿÿÿÿ
ª "QªN
%
output_1
output_1
%
output_2
output_2»
B__inference_a2c_model_layer_call_and_return_conditional_losses_351u0¢-
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
 µ
B__inference_a2c_model_layer_call_and_return_conditional_losses_448o*¢'
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
 µ
B__inference_a2c_model_layer_call_and_return_conditional_losses_474o*¢'
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
 »
B__inference_a2c_model_layer_call_and_return_conditional_losses_578u0¢-
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
 µ
B__inference_a2c_model_layer_call_and_return_conditional_losses_643o*¢'
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
 »
B__inference_a2c_model_layer_call_and_return_conditional_losses_669u0¢-
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
 µ
B__inference_a2c_model_layer_call_and_return_conditional_losses_682o*¢'
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
 »
B__inference_a2c_model_layer_call_and_return_conditional_losses_708u0¢-
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
 À
G__inference_a2c_model_layer_call_and_return_conditional_losses_80414075u0¢-
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
G__inference_a2c_model_layer_call_and_return_conditional_losses_80414094u0¢-
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
G__inference_a2c_model_layer_call_and_return_conditional_losses_80414151o*¢'
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
G__inference_a2c_model_layer_call_and_return_conditional_losses_80414170o*¢'
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
 
'__inference_a2c_model_layer_call_fn_276a*¢'
 ¢

xÿÿÿÿÿÿÿÿÿ
p
ª "+(

0

1
'__inference_a2c_model_layer_call_fn_289g0¢-
&¢#

input_1ÿÿÿÿÿÿÿÿÿ
p 
ª "+(

0

1
'__inference_a2c_model_layer_call_fn_302g0¢-
&¢#

input_1ÿÿÿÿÿÿÿÿÿ
p
ª "+(

0

1
'__inference_a2c_model_layer_call_fn_315a*¢'
 ¢

xÿÿÿÿÿÿÿÿÿ
p 
ª "+(

0

1
'__inference_a2c_model_layer_call_fn_526g0¢-
&¢#

input_1ÿÿÿÿÿÿÿÿÿ
p 
ª "+(

0

1
'__inference_a2c_model_layer_call_fn_539g0¢-
&¢#

input_1ÿÿÿÿÿÿÿÿÿ
p
ª "+(

0

1
'__inference_a2c_model_layer_call_fn_552a*¢'
 ¢

xÿÿÿÿÿÿÿÿÿ
p
ª "+(

0

1
'__inference_a2c_model_layer_call_fn_565a*¢'
 ¢

xÿÿÿÿÿÿÿÿÿ
p 
ª "+(

0

1
,__inference_a2c_model_layer_call_fn_80414113g0¢-
&¢#

input_1ÿÿÿÿÿÿÿÿÿ
p
ª "+(

0

1
,__inference_a2c_model_layer_call_fn_80414132g0¢-
&¢#

input_1ÿÿÿÿÿÿÿÿÿ
p 
ª "+(

0

1
,__inference_a2c_model_layer_call_fn_80414189a*¢'
 ¢

xÿÿÿÿÿÿÿÿÿ
p
ª "+(

0

1
,__inference_a2c_model_layer_call_fn_80414208a*¢'
 ¢

xÿÿÿÿÿÿÿÿÿ
p 
ª "+(

0

1Þ
J__inference_functional_1_layer_call_and_return_conditional_losses_804139578¢5
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
J__inference_functional_1_layer_call_and_return_conditional_losses_804139778¢5
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
 Ý
J__inference_functional_1_layer_call_and_return_conditional_losses_804142327¢4
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
J__inference_functional_1_layer_call_and_return_conditional_losses_804142567¢4
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
/__inference_functional_1_layer_call_fn_804140178¢5
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
/__inference_functional_1_layer_call_fn_804140568¢5
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
1ÿÿÿÿÿÿÿÿÿ´
/__inference_functional_1_layer_call_fn_804142757¢4
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
/__inference_functional_1_layer_call_fn_804142947¢4
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
1ÿÿÿÿÿÿÿÿÿº
!__inference_signature_wrapper_5007¢4
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
output_2º
!__inference_signature_wrapper_6177¢4
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
output_2¿
&__inference_signature_wrapper_804138727¢4
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