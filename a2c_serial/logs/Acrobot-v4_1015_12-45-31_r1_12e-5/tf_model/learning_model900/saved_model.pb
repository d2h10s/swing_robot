îť
żŁ
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
ž
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
 "serve*2.3.02unknown8°ŕ
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
valueĐBÍ BĆ
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
ů
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
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙

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
GPU 2J 8 *0
f+R)
'__inference_signature_wrapper_145149426
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
î
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
GPU 2J 8 *+
f&R$
"__inference__traced_save_145149948
ń
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
GPU 2J 8 *.
f)R'
%__inference__traced_restore_145149976íŽ
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

identity_1˘StatefulPartitionedCall
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
':˙˙˙˙˙˙˙˙˙::::::22
StatefulPartitionedCallStatefulPartitionedCall:F B
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex
Ú

Ź
H__inference_a2c_model_layer_call_and_return_conditional_losses_145149705
x
a2c_model_145149689
a2c_model_145149691
a2c_model_145149693
a2c_model_145149695
a2c_model_145149697
a2c_model_145149699
identity

identity_1˘!a2c_model/StatefulPartitionedCallŢ
!a2c_model/StatefulPartitionedCallStatefulPartitionedCallxa2c_model_145149689a2c_model_145149691a2c_model_145149693a2c_model_145149695a2c_model_145149697a2c_model_145149699*
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
':˙˙˙˙˙˙˙˙˙::::::2F
!a2c_model/StatefulPartitionedCall!a2c_model/StatefulPartitionedCall:F B
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex
Ú

Ź
H__inference_a2c_model_layer_call_and_return_conditional_losses_145149724
x
a2c_model_145149708
a2c_model_145149710
a2c_model_145149712
a2c_model_145149714
a2c_model_145149716
a2c_model_145149718
identity

identity_1˘!a2c_model/StatefulPartitionedCallŢ
!a2c_model/StatefulPartitionedCallStatefulPartitionedCallxa2c_model_145149708a2c_model_145149710a2c_model_145149712a2c_model_145149714a2c_model_145149716a2c_model_145149718*
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
':˙˙˙˙˙˙˙˙˙::::::2F
!a2c_model/StatefulPartitionedCall!a2c_model/StatefulPartitionedCall:F B
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex
˘
Ą
%__inference__traced_restore_145149976
file_prefix"
assignvariableop_dense1_kernel"
assignvariableop_1_dense1_bias#
assignvariableop_2_actor_kernel!
assignvariableop_3_actor_bias$
 assignvariableop_4_critic_kernel"
assignvariableop_5_critic_bias

identity_7˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_2˘AssignVariableOp_3˘AssignVariableOp_4˘AssignVariableOp_5
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ł
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

Identity_1Ł
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

Identity_3˘
AssignVariableOp_3AssignVariableOpassignvariableop_3_actor_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ľ
AssignVariableOp_4AssignVariableOp assignvariableop_4_critic_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ł
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

identity_1˘StatefulPartitionedCall
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
':˙˙˙˙˙˙˙˙˙::::::22
StatefulPartitionedCallStatefulPartitionedCall:F B
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex
˝

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

identity_1˘!a2c_model/StatefulPartitionedCallÍ
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
':˙˙˙˙˙˙˙˙˙::::::2F
!a2c_model/StatefulPartitionedCall!a2c_model/StatefulPartitionedCall:L H
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1

ű
"__inference__traced_save_145149948
file_prefix,
(savev2_dense1_kernel_read_readvariableop*
&savev2_dense1_bias_read_readvariableop+
'savev2_actor_kernel_read_readvariableop)
%savev2_actor_bias_read_readvariableop,
(savev2_critic_kernel_read_readvariableop*
&savev2_critic_bias_read_readvariableop
savev2_const

identity_1˘MergeV2Checkpoints
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
value3B1 B+_temp_d1b6073f4f524d1388ddbd9f6d9864ef/part2	
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
ShardedFilename/shardŚ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ł
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
SaveV2ş
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesĄ
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
ž
ß
K__inference_functional_1_layer_call_and_return_conditional_losses_145149531
input_1
dense1_145149514
dense1_145149516
critic_145149519
critic_145149521
actor_145149524
actor_145149526
identity

identity_1˘Actor/StatefulPartitionedCall˘Critic/StatefulPartitionedCall˘Dense1/StatefulPartitionedCall
Dense1/StatefulPartitionedCallStatefulPartitionedCallinput_1dense1_145149514dense1_145149516*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_Dense1_layer_call_and_return_conditional_losses_1451494412 
Dense1/StatefulPartitionedCall´
Critic/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0critic_145149519critic_145149521*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_Critic_layer_call_and_return_conditional_losses_1451494672 
Critic/StatefulPartitionedCallŻ
Actor/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0actor_145149524actor_145149526*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_Actor_layer_call_and_return_conditional_losses_1451494932
Actor/StatefulPartitionedCallÜ
IdentityIdentity&Actor/StatefulPartitionedCall:output:0^Actor/StatefulPartitionedCall^Critic/StatefulPartitionedCall^Dense1/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identityá

Identity_1Identity'Critic/StatefulPartitionedCall:output:0^Actor/StatefulPartitionedCall^Critic/StatefulPartitionedCall^Dense1/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙::::::2>
Actor/StatefulPartitionedCallActor/StatefulPartitionedCall2@
Critic/StatefulPartitionedCallCritic/StatefulPartitionedCall2@
Dense1/StatefulPartitionedCallDense1/StatefulPartitionedCall:P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
î
É
'__inference_signature_wrapper_145149426
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1˘StatefulPartitionedCall
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
GPU 2J 8 *-
f(R&
$__inference__wrapped_model_1451493082
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
':˙˙˙˙˙˙˙˙˙::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
¨
ů
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
statefulpartitionedcall_4˘StatefulPartitionedCallÎ
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout

2*m
_output_shapes[
Y:::	:	:	:	::˙˙˙˙˙˙˙˙˙*(
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
':˙˙˙˙˙˙˙˙˙::::::*S
backward_function_name97__inference___backward_restored_function_body_1104_117122
StatefulPartitionedCallStatefulPartitionedCall:F B
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex
Ť

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

identity_1˘!a2c_model/StatefulPartitionedCallÇ
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
':˙˙˙˙˙˙˙˙˙::::::2F
!a2c_model/StatefulPartitionedCall!a2c_model/StatefulPartitionedCall:F B
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex

Ô
$__inference__wrapped_model_145149308
input_1!
a2c_model_a2c_model_145149292!
a2c_model_a2c_model_145149294!
a2c_model_a2c_model_145149296!
a2c_model_a2c_model_145149298!
a2c_model_a2c_model_145149300!
a2c_model_a2c_model_145149302
identity

identity_1˘+a2c_model/a2c_model/StatefulPartitionedCall´
+a2c_model/a2c_model/StatefulPartitionedCallStatefulPartitionedCallinput_1a2c_model_a2c_model_145149292a2c_model_a2c_model_145149294a2c_model_a2c_model_145149296a2c_model_a2c_model_145149298a2c_model_a2c_model_145149300a2c_model_a2c_model_145149302*
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

Identityą

Identity_1Identity4a2c_model/a2c_model/StatefulPartitionedCall:output:1,^a2c_model/a2c_model/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙::::::2Z
+a2c_model/a2c_model/StatefulPartitionedCall+a2c_model/a2c_model/StatefulPartitionedCall:L H
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
ŕ

*__inference_Dense1_layer_call_fn_145149868

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_Dense1_layer_call_and_return_conditional_losses_1451494412
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
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

identity_1˘StatefulPartitionedCallŠ
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
':˙˙˙˙˙˙˙˙˙::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
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

identity_1˘StatefulPartitionedCallŠ
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
':˙˙˙˙˙˙˙˙˙::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
ú
Ă
'__inference_a2c_model_layer_call_fn_315
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1˘StatefulPartitionedCallŁ
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
':˙˙˙˙˙˙˙˙˙::::::22
StatefulPartitionedCallStatefulPartitionedCall:F B
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex
Ä
Ă
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

ExpandDimsĘ
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
functional_1/Dense1/ReluĘ
)functional_1/Critic/MatMul/ReadVariableOpReadVariableOp2functional_1_critic_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02+
)functional_1/Critic/MatMul/ReadVariableOpĆ
functional_1/Critic/MatMulMatMul&functional_1/Dense1/Relu:activations:01functional_1/Critic/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Critic/MatMulČ
*functional_1/Critic/BiasAdd/ReadVariableOpReadVariableOp3functional_1_critic_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*functional_1/Critic/BiasAdd/ReadVariableOpČ
functional_1/Critic/BiasAddBiasAdd$functional_1/Critic/MatMul:product:02functional_1/Critic/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Critic/BiasAddÇ
(functional_1/Actor/MatMul/ReadVariableOpReadVariableOp1functional_1_actor_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02*
(functional_1/Actor/MatMul/ReadVariableOpĂ
functional_1/Actor/MatMulMatMul&functional_1/Dense1/Relu:activations:00functional_1/Actor/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Actor/MatMulĹ
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
':˙˙˙˙˙˙˙˙˙:::::::L H
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
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

identity_1Ł
Dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
Dense1/MatMul/ReadVariableOp
Dense1/MatMulMatMulinputs$Dense1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
Dense1/MatMul˘
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
Dense1/ReluŁ
Critic/MatMul/ReadVariableOpReadVariableOp%critic_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
Critic/MatMul/ReadVariableOp
Critic/MatMulMatMulDense1/Relu:activations:0$Critic/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Critic/MatMulĄ
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
+:˙˙˙˙˙˙˙˙˙:::::::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ě

˛
H__inference_a2c_model_layer_call_and_return_conditional_losses_145149648
input_1
a2c_model_145149632
a2c_model_145149634
a2c_model_145149636
a2c_model_145149638
a2c_model_145149640
a2c_model_145149642
identity

identity_1˘!a2c_model/StatefulPartitionedCallä
!a2c_model/StatefulPartitionedCallStatefulPartitionedCallinput_1a2c_model_145149632a2c_model_145149634a2c_model_145149636a2c_model_145149638a2c_model_145149640a2c_model_145149642*
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
':˙˙˙˙˙˙˙˙˙::::::2F
!a2c_model/StatefulPartitionedCall!a2c_model/StatefulPartitionedCall:L H
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
Đ
Ź
D__inference_Actor_layer_call_and_return_conditional_losses_145149493

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
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
˛
˝
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

ExpandDimsĘ
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
functional_1/Dense1/ReluĘ
)functional_1/Critic/MatMul/ReadVariableOpReadVariableOp2functional_1_critic_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02+
)functional_1/Critic/MatMul/ReadVariableOpĆ
functional_1/Critic/MatMulMatMul&functional_1/Dense1/Relu:activations:01functional_1/Critic/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Critic/MatMulČ
*functional_1/Critic/BiasAdd/ReadVariableOpReadVariableOp3functional_1_critic_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*functional_1/Critic/BiasAdd/ReadVariableOpČ
functional_1/Critic/BiasAddBiasAdd$functional_1/Critic/MatMul:product:02functional_1/Critic/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Critic/BiasAddÇ
(functional_1/Actor/MatMul/ReadVariableOpReadVariableOp1functional_1_actor_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02*
(functional_1/Actor/MatMul/ReadVariableOpĂ
functional_1/Actor/MatMulMatMul&functional_1/Dense1/Relu:activations:00functional_1/Actor/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Actor/MatMulĹ
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
':˙˙˙˙˙˙˙˙˙:::::::F B
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex
Ç	
Ń
0__inference_functional_1_layer_call_fn_145149848

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1˘StatefulPartitionedCallĂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_functional_1_layer_call_and_return_conditional_losses_1451495932
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ú
Ă
'__inference_a2c_model_layer_call_fn_276
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1˘StatefulPartitionedCallŁ
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
':˙˙˙˙˙˙˙˙˙::::::22
StatefulPartitionedCallStatefulPartitionedCall:F B
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex
°
­
E__inference_Dense1_layer_call_and_return_conditional_losses_145149859

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
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙:::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
	
Ď
-__inference_a2c_model_layer_call_fn_145149667
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1˘StatefulPartitionedCallŻ
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
GPU 2J 8 *Q
fLRJ
H__inference_a2c_model_layer_call_and_return_conditional_losses_1451493692
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
':˙˙˙˙˙˙˙˙˙::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
	
É
-__inference_a2c_model_layer_call_fn_145149743
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1˘StatefulPartitionedCallŠ
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
GPU 2J 8 *Q
fLRJ
H__inference_a2c_model_layer_call_and_return_conditional_losses_1451493692
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
':˙˙˙˙˙˙˙˙˙::::::22
StatefulPartitionedCallStatefulPartitionedCall:F B
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex
	
É
-__inference_a2c_model_layer_call_fn_145149762
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1˘StatefulPartitionedCallŠ
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
GPU 2J 8 *Q
fLRJ
H__inference_a2c_model_layer_call_and_return_conditional_losses_1451493692
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
':˙˙˙˙˙˙˙˙˙::::::22
StatefulPartitionedCallStatefulPartitionedCall:F B
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex
é
Ł
B__inference_a2c_model_layer_call_and_return_conditional_losses_263
x
functional_1_89352
functional_1_89354
functional_1_89356
functional_1_89358
functional_1_89360
functional_1_89362
identity

identity_1˘$functional_1/StatefulPartitionedCallb
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
:˙˙˙˙˙˙˙˙˙2

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

IdentityŁ

Identity_1Identity-functional_1/StatefulPartitionedCall:output:1%^functional_1/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙::::::2L
$functional_1/StatefulPartitionedCall$functional_1/StatefulPartitionedCall:F B
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex
Ä
Ă
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

ExpandDimsĘ
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
functional_1/Dense1/ReluĘ
)functional_1/Critic/MatMul/ReadVariableOpReadVariableOp2functional_1_critic_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02+
)functional_1/Critic/MatMul/ReadVariableOpĆ
functional_1/Critic/MatMulMatMul&functional_1/Dense1/Relu:activations:01functional_1/Critic/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Critic/MatMulČ
*functional_1/Critic/BiasAdd/ReadVariableOpReadVariableOp3functional_1_critic_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*functional_1/Critic/BiasAdd/ReadVariableOpČ
functional_1/Critic/BiasAddBiasAdd$functional_1/Critic/MatMul:product:02functional_1/Critic/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Critic/BiasAddÇ
(functional_1/Actor/MatMul/ReadVariableOpReadVariableOp1functional_1_actor_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02*
(functional_1/Actor/MatMul/ReadVariableOpĂ
functional_1/Actor/MatMulMatMul&functional_1/Dense1/Relu:activations:00functional_1/Actor/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Actor/MatMulĹ
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
':˙˙˙˙˙˙˙˙˙:::::::L H
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
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

identity_1˘StatefulPartitionedCall
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
':˙˙˙˙˙˙˙˙˙::::::22
StatefulPartitionedCallStatefulPartitionedCall:F B
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex
°
­
E__inference_Dense1_layer_call_and_return_conditional_losses_145149441

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
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙:::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
â
Ă
!__inference_signature_wrapper_500
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1˘StatefulPartitionedCall
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
':˙˙˙˙˙˙˙˙˙::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
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

identity_1˘StatefulPartitionedCallŠ
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
':˙˙˙˙˙˙˙˙˙::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
Ę	
Ň
0__inference_functional_1_layer_call_fn_145149571
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1˘StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_functional_1_layer_call_and_return_conditional_losses_1451495542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1

ý
K__inference_functional_1_layer_call_and_return_conditional_losses_145149810

inputs)
%dense1_matmul_readvariableop_resource*
&dense1_biasadd_readvariableop_resource)
%critic_matmul_readvariableop_resource*
&critic_biasadd_readvariableop_resource(
$actor_matmul_readvariableop_resource)
%actor_biasadd_readvariableop_resource
identity

identity_1Ł
Dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
Dense1/MatMul/ReadVariableOp
Dense1/MatMulMatMulinputs$Dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Dense1/MatMul˘
Dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
Dense1/BiasAdd/ReadVariableOp
Dense1/BiasAddBiasAddDense1/MatMul:product:0%Dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Dense1/BiasAddn
Dense1/ReluReluDense1/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Dense1/ReluŁ
Critic/MatMul/ReadVariableOpReadVariableOp%critic_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
Critic/MatMul/ReadVariableOp
Critic/MatMulMatMulDense1/Relu:activations:0$Critic/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Critic/MatMulĄ
Critic/BiasAdd/ReadVariableOpReadVariableOp&critic_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
Critic/BiasAdd/ReadVariableOp
Critic/BiasAddBiasAddCritic/MatMul:product:0%Critic/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Critic/BiasAdd 
Actor/MatMul/ReadVariableOpReadVariableOp$actor_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
Actor/MatMul/ReadVariableOp
Actor/MatMulMatMulDense1/Relu:activations:0#Actor/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Actor/MatMul
Actor/BiasAdd/ReadVariableOpReadVariableOp%actor_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
Actor/BiasAdd/ReadVariableOp
Actor/BiasAddBiasAddActor/MatMul:product:0$Actor/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Actor/BiasAddj
IdentityIdentityActor/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identityo

Identity_1IdentityCritic/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙:::::::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ě

˛
H__inference_a2c_model_layer_call_and_return_conditional_losses_145149629
input_1
a2c_model_145149613
a2c_model_145149615
a2c_model_145149617
a2c_model_145149619
a2c_model_145149621
a2c_model_145149623
identity

identity_1˘!a2c_model/StatefulPartitionedCallä
!a2c_model/StatefulPartitionedCallStatefulPartitionedCallinput_1a2c_model_145149613a2c_model_145149615a2c_model_145149617a2c_model_145149619a2c_model_145149621a2c_model_145149623*
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
':˙˙˙˙˙˙˙˙˙::::::2F
!a2c_model/StatefulPartitionedCall!a2c_model/StatefulPartitionedCall:L H
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
Ţ
~
)__inference_Actor_layer_call_fn_145149887

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_Actor_layer_call_and_return_conditional_losses_1451494932
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
˝

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

identity_1˘!a2c_model/StatefulPartitionedCallÍ
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
':˙˙˙˙˙˙˙˙˙::::::2F
!a2c_model/StatefulPartitionedCall!a2c_model/StatefulPartitionedCall:L H
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
ö!
˙
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

ExpandDimsĘ
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
functional_1/Dense1/ReluĘ
)functional_1/Critic/MatMul/ReadVariableOpReadVariableOp2functional_1_critic_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02+
)functional_1/Critic/MatMul/ReadVariableOpĆ
functional_1/Critic/MatMulMatMul&functional_1/Dense1/Relu:activations:01functional_1/Critic/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Critic/MatMulČ
*functional_1/Critic/BiasAdd/ReadVariableOpReadVariableOp3functional_1_critic_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*functional_1/Critic/BiasAdd/ReadVariableOpČ
functional_1/Critic/BiasAddBiasAdd$functional_1/Critic/MatMul:product:02functional_1/Critic/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Critic/BiasAddÇ
(functional_1/Actor/MatMul/ReadVariableOpReadVariableOp1functional_1_actor_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02*
(functional_1/Actor/MatMul/ReadVariableOpĂ
functional_1/Actor/MatMulMatMul&functional_1/Dense1/Relu:activations:00functional_1/Actor/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Actor/MatMulĹ
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
':˙˙˙˙˙˙˙˙˙::::::*o
backward_function_nameUS__inference___backward_a2c_model_layer_call_and_return_conditional_losses_1110_1143:F B
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex
ú
Ă
'__inference_a2c_model_layer_call_fn_565
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1˘StatefulPartitionedCallŁ
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
':˙˙˙˙˙˙˙˙˙::::::22
StatefulPartitionedCallStatefulPartitionedCall:F B
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex
	
Ď
-__inference_a2c_model_layer_call_fn_145149686
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1˘StatefulPartitionedCallŻ
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
GPU 2J 8 *Q
fLRJ
H__inference_a2c_model_layer_call_and_return_conditional_losses_1451493692
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
':˙˙˙˙˙˙˙˙˙::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
˛
˝
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

ExpandDimsĘ
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
functional_1/Dense1/ReluĘ
)functional_1/Critic/MatMul/ReadVariableOpReadVariableOp2functional_1_critic_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02+
)functional_1/Critic/MatMul/ReadVariableOpĆ
functional_1/Critic/MatMulMatMul&functional_1/Dense1/Relu:activations:01functional_1/Critic/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Critic/MatMulČ
*functional_1/Critic/BiasAdd/ReadVariableOpReadVariableOp3functional_1_critic_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*functional_1/Critic/BiasAdd/ReadVariableOpČ
functional_1/Critic/BiasAddBiasAdd$functional_1/Critic/MatMul:product:02functional_1/Critic/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Critic/BiasAddÇ
(functional_1/Actor/MatMul/ReadVariableOpReadVariableOp1functional_1_actor_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02*
(functional_1/Actor/MatMul/ReadVariableOpĂ
functional_1/Actor/MatMulMatMul&functional_1/Dense1/Relu:activations:00functional_1/Actor/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/Actor/MatMulĹ
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
':˙˙˙˙˙˙˙˙˙:::::::F B
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex
ť
Ţ
K__inference_functional_1_layer_call_and_return_conditional_losses_145149554

inputs
dense1_145149537
dense1_145149539
critic_145149542
critic_145149544
actor_145149547
actor_145149549
identity

identity_1˘Actor/StatefulPartitionedCall˘Critic/StatefulPartitionedCall˘Dense1/StatefulPartitionedCall
Dense1/StatefulPartitionedCallStatefulPartitionedCallinputsdense1_145149537dense1_145149539*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_Dense1_layer_call_and_return_conditional_losses_1451494412 
Dense1/StatefulPartitionedCall´
Critic/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0critic_145149542critic_145149544*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_Critic_layer_call_and_return_conditional_losses_1451494672 
Critic/StatefulPartitionedCallŻ
Actor/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0actor_145149547actor_145149549*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_Actor_layer_call_and_return_conditional_losses_1451494932
Actor/StatefulPartitionedCallÜ
IdentityIdentity&Actor/StatefulPartitionedCall:output:0^Actor/StatefulPartitionedCall^Critic/StatefulPartitionedCall^Dense1/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identityá

Identity_1Identity'Critic/StatefulPartitionedCall:output:0^Actor/StatefulPartitionedCall^Critic/StatefulPartitionedCall^Dense1/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙::::::2>
Actor/StatefulPartitionedCallActor/StatefulPartitionedCall2@
Critic/StatefulPartitionedCallCritic/StatefulPartitionedCall2@
Dense1/StatefulPartitionedCallDense1/StatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Đ
Ź
D__inference_Actor_layer_call_and_return_conditional_losses_145149878

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
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
â
Ă
!__inference_signature_wrapper_617
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1˘StatefulPartitionedCall
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
':˙˙˙˙˙˙˙˙˙::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
ĺ 
Ű
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
a2c_model/ExpandDimsč
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
4a2c_model/functional_1/Dense1/BiasAdd/ReadVariableOpń
%a2c_model/functional_1/Dense1/BiasAddBiasAdd.a2c_model/functional_1/Dense1/MatMul:product:0<a2c_model/functional_1/Dense1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2'
%a2c_model/functional_1/Dense1/BiasAddŞ
"a2c_model/functional_1/Dense1/ReluRelu.a2c_model/functional_1/Dense1/BiasAdd:output:0*
T0*
_output_shapes
:	2$
"a2c_model/functional_1/Dense1/Reluč
3a2c_model/functional_1/Critic/MatMul/ReadVariableOpReadVariableOp<a2c_model_functional_1_critic_matmul_readvariableop_resource*
_output_shapes
:	*
dtype025
3a2c_model/functional_1/Critic/MatMul/ReadVariableOpî
$a2c_model/functional_1/Critic/MatMulMatMul0a2c_model/functional_1/Dense1/Relu:activations:0;a2c_model/functional_1/Critic/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2&
$a2c_model/functional_1/Critic/MatMulć
4a2c_model/functional_1/Critic/BiasAdd/ReadVariableOpReadVariableOp=a2c_model_functional_1_critic_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4a2c_model/functional_1/Critic/BiasAdd/ReadVariableOpđ
%a2c_model/functional_1/Critic/BiasAddBiasAdd.a2c_model/functional_1/Critic/MatMul:product:0<a2c_model/functional_1/Critic/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2'
%a2c_model/functional_1/Critic/BiasAddĺ
2a2c_model/functional_1/Actor/MatMul/ReadVariableOpReadVariableOp;a2c_model_functional_1_actor_matmul_readvariableop_resource*
_output_shapes
:	*
dtype024
2a2c_model/functional_1/Actor/MatMul/ReadVariableOpë
#a2c_model/functional_1/Actor/MatMulMatMul0a2c_model/functional_1/Dense1/Relu:activations:0:a2c_model/functional_1/Actor/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2%
#a2c_model/functional_1/Actor/MatMulă
3a2c_model/functional_1/Actor/BiasAdd/ReadVariableOpReadVariableOp<a2c_model_functional_1_actor_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3a2c_model/functional_1/Actor/BiasAdd/ReadVariableOpě
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
':˙˙˙˙˙˙˙˙˙:::::::L H
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
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

identity_1˘StatefulPartitionedCall
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
':˙˙˙˙˙˙˙˙˙::::::22
StatefulPartitionedCallStatefulPartitionedCall:F B
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex
Ú

Ź
H__inference_a2c_model_layer_call_and_return_conditional_losses_145149369
x
a2c_model_145149353
a2c_model_145149355
a2c_model_145149357
a2c_model_145149359
a2c_model_145149361
a2c_model_145149363
identity

identity_1˘!a2c_model/StatefulPartitionedCallŢ
!a2c_model/StatefulPartitionedCallStatefulPartitionedCallxa2c_model_145149353a2c_model_145149355a2c_model_145149357a2c_model_145149359a2c_model_145149361a2c_model_145149363*
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
':˙˙˙˙˙˙˙˙˙::::::2F
!a2c_model/StatefulPartitionedCall!a2c_model/StatefulPartitionedCall:F B
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex
Ń
­
E__inference_Critic_layer_call_and_return_conditional_losses_145149467

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
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ť

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

identity_1˘!a2c_model/StatefulPartitionedCallÇ
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
':˙˙˙˙˙˙˙˙˙::::::2F
!a2c_model/StatefulPartitionedCall!a2c_model/StatefulPartitionedCall:F B
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex
ť
Ţ
K__inference_functional_1_layer_call_and_return_conditional_losses_145149593

inputs
dense1_145149576
dense1_145149578
critic_145149581
critic_145149583
actor_145149586
actor_145149588
identity

identity_1˘Actor/StatefulPartitionedCall˘Critic/StatefulPartitionedCall˘Dense1/StatefulPartitionedCall
Dense1/StatefulPartitionedCallStatefulPartitionedCallinputsdense1_145149576dense1_145149578*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_Dense1_layer_call_and_return_conditional_losses_1451494412 
Dense1/StatefulPartitionedCall´
Critic/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0critic_145149581critic_145149583*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_Critic_layer_call_and_return_conditional_losses_1451494672 
Critic/StatefulPartitionedCallŻ
Actor/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0actor_145149586actor_145149588*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_Actor_layer_call_and_return_conditional_losses_1451494932
Actor/StatefulPartitionedCallÜ
IdentityIdentity&Actor/StatefulPartitionedCall:output:0^Actor/StatefulPartitionedCall^Critic/StatefulPartitionedCall^Dense1/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identityá

Identity_1Identity'Critic/StatefulPartitionedCall:output:0^Actor/StatefulPartitionedCall^Critic/StatefulPartitionedCall^Dense1/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙::::::2>
Actor/StatefulPartitionedCallActor/StatefulPartitionedCall2@
Critic/StatefulPartitionedCallCritic/StatefulPartitionedCall2@
Dense1/StatefulPartitionedCallDense1/StatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ú
Ă
'__inference_a2c_model_layer_call_fn_552
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1˘StatefulPartitionedCallŁ
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
':˙˙˙˙˙˙˙˙˙::::::22
StatefulPartitionedCallStatefulPartitionedCall:F B
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex

ý
K__inference_functional_1_layer_call_and_return_conditional_losses_145149786

inputs)
%dense1_matmul_readvariableop_resource*
&dense1_biasadd_readvariableop_resource)
%critic_matmul_readvariableop_resource*
&critic_biasadd_readvariableop_resource(
$actor_matmul_readvariableop_resource)
%actor_biasadd_readvariableop_resource
identity

identity_1Ł
Dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
Dense1/MatMul/ReadVariableOp
Dense1/MatMulMatMulinputs$Dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Dense1/MatMul˘
Dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
Dense1/BiasAdd/ReadVariableOp
Dense1/BiasAddBiasAddDense1/MatMul:product:0%Dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Dense1/BiasAddn
Dense1/ReluReluDense1/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Dense1/ReluŁ
Critic/MatMul/ReadVariableOpReadVariableOp%critic_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
Critic/MatMul/ReadVariableOp
Critic/MatMulMatMulDense1/Relu:activations:0$Critic/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Critic/MatMulĄ
Critic/BiasAdd/ReadVariableOpReadVariableOp&critic_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
Critic/BiasAdd/ReadVariableOp
Critic/BiasAddBiasAddCritic/MatMul:product:0%Critic/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Critic/BiasAdd 
Actor/MatMul/ReadVariableOpReadVariableOp$actor_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
Actor/MatMul/ReadVariableOp
Actor/MatMulMatMulDense1/Relu:activations:0#Actor/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Actor/MatMul
Actor/BiasAdd/ReadVariableOpReadVariableOp%actor_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
Actor/BiasAdd/ReadVariableOp
Actor/BiasAddBiasAddActor/MatMul:product:0$Actor/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Actor/BiasAddj
IdentityIdentityActor/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identityo

Identity_1IdentityCritic/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙:::::::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ę	
Ň
0__inference_functional_1_layer_call_fn_145149610
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1˘StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_functional_1_layer_call_and_return_conditional_losses_1451495932
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
ë
ź
__inference__wrapped_model_487
input_1
a2c_model_a2c_model_161419
a2c_model_a2c_model_161421
a2c_model_a2c_model_161423
a2c_model_a2c_model_161425
a2c_model_a2c_model_161427
a2c_model_a2c_model_161429
identity

identity_1˘+a2c_model/a2c_model/StatefulPartitionedCall
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

Identityą

Identity_1Identity4a2c_model/a2c_model/StatefulPartitionedCall:output:1,^a2c_model/a2c_model/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙::::::2Z
+a2c_model/a2c_model/StatefulPartitionedCall+a2c_model/a2c_model/StatefulPartitionedCall:L H
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
ž
ß
K__inference_functional_1_layer_call_and_return_conditional_losses_145149511
input_1
dense1_145149452
dense1_145149454
critic_145149478
critic_145149480
actor_145149504
actor_145149506
identity

identity_1˘Actor/StatefulPartitionedCall˘Critic/StatefulPartitionedCall˘Dense1/StatefulPartitionedCall
Dense1/StatefulPartitionedCallStatefulPartitionedCallinput_1dense1_145149452dense1_145149454*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_Dense1_layer_call_and_return_conditional_losses_1451494412 
Dense1/StatefulPartitionedCall´
Critic/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0critic_145149478critic_145149480*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_Critic_layer_call_and_return_conditional_losses_1451494672 
Critic/StatefulPartitionedCallŻ
Actor/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0actor_145149504actor_145149506*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_Actor_layer_call_and_return_conditional_losses_1451494932
Actor/StatefulPartitionedCallÜ
IdentityIdentity&Actor/StatefulPartitionedCall:output:0^Actor/StatefulPartitionedCall^Critic/StatefulPartitionedCall^Dense1/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identityá

Identity_1Identity'Critic/StatefulPartitionedCall:output:0^Actor/StatefulPartitionedCall^Critic/StatefulPartitionedCall^Dense1/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙::::::2>
Actor/StatefulPartitionedCallActor/StatefulPartitionedCall2@
Critic/StatefulPartitionedCallCritic/StatefulPartitionedCall2@
Dense1/StatefulPartitionedCallDense1/StatefulPartitionedCall:P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
Ť

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

identity_1˘!a2c_model/StatefulPartitionedCallÇ
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
':˙˙˙˙˙˙˙˙˙::::::2F
!a2c_model/StatefulPartitionedCall!a2c_model/StatefulPartitionedCall:F B
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex
Ç	
Ń
0__inference_functional_1_layer_call_fn_145149829

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1˘StatefulPartitionedCallĂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_functional_1_layer_call_and_return_conditional_losses_1451495542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ŕ

*__inference_Critic_layer_call_fn_145149906

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallő
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_Critic_layer_call_and_return_conditional_losses_1451494672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ń
­
E__inference_Critic_layer_call_and_return_conditional_losses_145149897

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
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
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

identity_1˘StatefulPartitionedCallŠ
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
':˙˙˙˙˙˙˙˙˙::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ó
serving_defaultż
7
input_1,
serving_default_input_1:0˙˙˙˙˙˙˙˙˙3
output_1'
StatefulPartitionedCall:03
output_2'
StatefulPartitionedCall:1tensorflow/serving/predict:ďÓ
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
_tf_keras_modelă{"class_name": "a2c_model", "name": "a2c_model", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "a2c_model"}}
Č
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
Ę
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
Č
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
ň!
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
*a&call_and_return_all_conditional_losses"ź
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
#3_self_saveable_object_factories"ć
_tf_keras_input_layerĆ{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
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
*f&call_and_return_all_conditional_losses"Ę
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
*h&call_and_return_all_conditional_losses"Ě
_tf_keras_layer˛{"class_name": "Dense", "name": "Critic", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Critic", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
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
Ţ2Ű
$__inference__wrapped_model_145149308˛
˛
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
annotationsŞ *"˘

input_1˙˙˙˙˙˙˙˙˙
đ2í
-__inference_a2c_model_layer_call_fn_145149686
-__inference_a2c_model_layer_call_fn_145149743
-__inference_a2c_model_layer_call_fn_145149667
-__inference_a2c_model_layer_call_fn_145149762Ž
Ľ˛Ą
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
annotationsŞ *
 
Ü2Ů
H__inference_a2c_model_layer_call_and_return_conditional_losses_145149629
H__inference_a2c_model_layer_call_and_return_conditional_losses_145149648
H__inference_a2c_model_layer_call_and_return_conditional_losses_145149705
H__inference_a2c_model_layer_call_and_return_conditional_losses_145149724Ž
Ľ˛Ą
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
annotationsŞ *
 
Î2Ë
'__inference_a2c_model_layer_call_fn_565
'__inference_a2c_model_layer_call_fn_539
'__inference_a2c_model_layer_call_fn_552
'__inference_a2c_model_layer_call_fn_526¤
˛
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
annotationsŞ *
 
ş2ˇ
B__inference_a2c_model_layer_call_and_return_conditional_losses_474
B__inference_a2c_model_layer_call_and_return_conditional_losses_669
B__inference_a2c_model_layer_call_and_return_conditional_losses_578
B__inference_a2c_model_layer_call_and_return_conditional_losses_682¤
˛
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
annotationsŞ *
 
6B4
'__inference_signature_wrapper_145149426input_1
Î2Ë
'__inference_a2c_model_layer_call_fn_276
'__inference_a2c_model_layer_call_fn_315
'__inference_a2c_model_layer_call_fn_302
'__inference_a2c_model_layer_call_fn_289¤
˛
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
annotationsŞ *
 
ş2ˇ
B__inference_a2c_model_layer_call_and_return_conditional_losses_643
B__inference_a2c_model_layer_call_and_return_conditional_losses_448
B__inference_a2c_model_layer_call_and_return_conditional_losses_708
B__inference_a2c_model_layer_call_and_return_conditional_losses_351¤
˛
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
annotationsŞ *
 
0B.
!__inference_signature_wrapper_500input_1
2
0__inference_functional_1_layer_call_fn_145149571
0__inference_functional_1_layer_call_fn_145149829
0__inference_functional_1_layer_call_fn_145149848
0__inference_functional_1_layer_call_fn_145149610Ŕ
ˇ˛ł
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
kwonlydefaultsŞ 
annotationsŞ *
 
ú2÷
K__inference_functional_1_layer_call_and_return_conditional_losses_145149786
K__inference_functional_1_layer_call_and_return_conditional_losses_145149810
K__inference_functional_1_layer_call_and_return_conditional_losses_145149511
K__inference_functional_1_layer_call_and_return_conditional_losses_145149531Ŕ
ˇ˛ł
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
kwonlydefaultsŞ 
annotationsŞ *
 
0B.
!__inference_signature_wrapper_617input_1
Ô2Ń
*__inference_Dense1_layer_call_fn_145149868˘
˛
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
annotationsŞ *
 
ď2ě
E__inference_Dense1_layer_call_and_return_conditional_losses_145149859˘
˛
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
annotationsŞ *
 
Ó2Đ
)__inference_Actor_layer_call_fn_145149887˘
˛
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
annotationsŞ *
 
î2ë
D__inference_Actor_layer_call_and_return_conditional_losses_145149878˘
˛
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
annotationsŞ *
 
Ô2Ń
*__inference_Critic_layer_call_fn_145149906˘
˛
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
annotationsŞ *
 
ď2ě
E__inference_Critic_layer_call_and_return_conditional_losses_145149897˘
˛
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
annotationsŞ *
 Ľ
D__inference_Actor_layer_call_and_return_conditional_losses_145149878]0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 }
)__inference_Actor_layer_call_fn_145149887P0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙Ś
E__inference_Critic_layer_call_and_return_conditional_losses_145149897]0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 ~
*__inference_Critic_layer_call_fn_145149906P0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙Ś
E__inference_Dense1_layer_call_and_return_conditional_losses_145149859]/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 ~
*__inference_Dense1_layer_call_fn_145149868P/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙˛
$__inference__wrapped_model_145149308,˘)
"˘

input_1˙˙˙˙˙˙˙˙˙
Ş "QŞN
%
output_1
output_1
%
output_2
output_2Á
H__inference_a2c_model_layer_call_and_return_conditional_losses_145149629u0˘-
&˘#

input_1˙˙˙˙˙˙˙˙˙
p
Ş "9˘6
/,

0/0

0/1
 Á
H__inference_a2c_model_layer_call_and_return_conditional_losses_145149648u0˘-
&˘#

input_1˙˙˙˙˙˙˙˙˙
p 
Ş "9˘6
/,

0/0

0/1
 ť
H__inference_a2c_model_layer_call_and_return_conditional_losses_145149705o*˘'
 ˘

x˙˙˙˙˙˙˙˙˙
p
Ş "9˘6
/,

0/0

0/1
 ť
H__inference_a2c_model_layer_call_and_return_conditional_losses_145149724o*˘'
 ˘

x˙˙˙˙˙˙˙˙˙
p 
Ş "9˘6
/,

0/0

0/1
 ť
B__inference_a2c_model_layer_call_and_return_conditional_losses_351u0˘-
&˘#

input_1˙˙˙˙˙˙˙˙˙
p
Ş "9˘6
/,

0/0

0/1
 ľ
B__inference_a2c_model_layer_call_and_return_conditional_losses_448o*˘'
 ˘

x˙˙˙˙˙˙˙˙˙
p 
Ş "9˘6
/,

0/0

0/1
 ľ
B__inference_a2c_model_layer_call_and_return_conditional_losses_474o*˘'
 ˘

x˙˙˙˙˙˙˙˙˙
p 
Ş "9˘6
/,

0/0

0/1
 ť
B__inference_a2c_model_layer_call_and_return_conditional_losses_578u0˘-
&˘#

input_1˙˙˙˙˙˙˙˙˙
p 
Ş "9˘6
/,

0/0

0/1
 ľ
B__inference_a2c_model_layer_call_and_return_conditional_losses_643o*˘'
 ˘

x˙˙˙˙˙˙˙˙˙
p
Ş "9˘6
/,

0/0

0/1
 ť
B__inference_a2c_model_layer_call_and_return_conditional_losses_669u0˘-
&˘#

input_1˙˙˙˙˙˙˙˙˙
p
Ş "9˘6
/,

0/0

0/1
 ľ
B__inference_a2c_model_layer_call_and_return_conditional_losses_682o*˘'
 ˘

x˙˙˙˙˙˙˙˙˙
p
Ş "9˘6
/,

0/0

0/1
 ť
B__inference_a2c_model_layer_call_and_return_conditional_losses_708u0˘-
&˘#

input_1˙˙˙˙˙˙˙˙˙
p 
Ş "9˘6
/,

0/0

0/1
 
-__inference_a2c_model_layer_call_fn_145149667g0˘-
&˘#

input_1˙˙˙˙˙˙˙˙˙
p
Ş "+(

0

1
-__inference_a2c_model_layer_call_fn_145149686g0˘-
&˘#

input_1˙˙˙˙˙˙˙˙˙
p 
Ş "+(

0

1
-__inference_a2c_model_layer_call_fn_145149743a*˘'
 ˘

x˙˙˙˙˙˙˙˙˙
p
Ş "+(

0

1
-__inference_a2c_model_layer_call_fn_145149762a*˘'
 ˘

x˙˙˙˙˙˙˙˙˙
p 
Ş "+(

0

1
'__inference_a2c_model_layer_call_fn_276a*˘'
 ˘

x˙˙˙˙˙˙˙˙˙
p
Ş "+(

0

1
'__inference_a2c_model_layer_call_fn_289g0˘-
&˘#

input_1˙˙˙˙˙˙˙˙˙
p 
Ş "+(

0

1
'__inference_a2c_model_layer_call_fn_302g0˘-
&˘#

input_1˙˙˙˙˙˙˙˙˙
p
Ş "+(

0

1
'__inference_a2c_model_layer_call_fn_315a*˘'
 ˘

x˙˙˙˙˙˙˙˙˙
p 
Ş "+(

0

1
'__inference_a2c_model_layer_call_fn_526g0˘-
&˘#

input_1˙˙˙˙˙˙˙˙˙
p 
Ş "+(

0

1
'__inference_a2c_model_layer_call_fn_539g0˘-
&˘#

input_1˙˙˙˙˙˙˙˙˙
p
Ş "+(

0

1
'__inference_a2c_model_layer_call_fn_552a*˘'
 ˘

x˙˙˙˙˙˙˙˙˙
p
Ş "+(

0

1
'__inference_a2c_model_layer_call_fn_565a*˘'
 ˘

x˙˙˙˙˙˙˙˙˙
p 
Ş "+(

0

1ß
K__inference_functional_1_layer_call_and_return_conditional_losses_1451495118˘5
.˘+
!
input_1˙˙˙˙˙˙˙˙˙
p

 
Ş "K˘H
A>

0/0˙˙˙˙˙˙˙˙˙

0/1˙˙˙˙˙˙˙˙˙
 ß
K__inference_functional_1_layer_call_and_return_conditional_losses_1451495318˘5
.˘+
!
input_1˙˙˙˙˙˙˙˙˙
p 

 
Ş "K˘H
A>

0/0˙˙˙˙˙˙˙˙˙

0/1˙˙˙˙˙˙˙˙˙
 Ţ
K__inference_functional_1_layer_call_and_return_conditional_losses_1451497867˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş "K˘H
A>

0/0˙˙˙˙˙˙˙˙˙

0/1˙˙˙˙˙˙˙˙˙
 Ţ
K__inference_functional_1_layer_call_and_return_conditional_losses_1451498107˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş "K˘H
A>

0/0˙˙˙˙˙˙˙˙˙

0/1˙˙˙˙˙˙˙˙˙
 ś
0__inference_functional_1_layer_call_fn_1451495718˘5
.˘+
!
input_1˙˙˙˙˙˙˙˙˙
p

 
Ş "=:

0˙˙˙˙˙˙˙˙˙

1˙˙˙˙˙˙˙˙˙ś
0__inference_functional_1_layer_call_fn_1451496108˘5
.˘+
!
input_1˙˙˙˙˙˙˙˙˙
p 

 
Ş "=:

0˙˙˙˙˙˙˙˙˙

1˙˙˙˙˙˙˙˙˙ľ
0__inference_functional_1_layer_call_fn_1451498297˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş "=:

0˙˙˙˙˙˙˙˙˙

1˙˙˙˙˙˙˙˙˙ľ
0__inference_functional_1_layer_call_fn_1451498487˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş "=:

0˙˙˙˙˙˙˙˙˙

1˙˙˙˙˙˙˙˙˙Ŕ
'__inference_signature_wrapper_1451494267˘4
˘ 
-Ş*
(
input_1
input_1˙˙˙˙˙˙˙˙˙"QŞN
%
output_1
output_1
%
output_2
output_2ş
!__inference_signature_wrapper_5007˘4
˘ 
-Ş*
(
input_1
input_1˙˙˙˙˙˙˙˙˙"QŞN
%
output_1
output_1
%
output_2
output_2ş
!__inference_signature_wrapper_6177˘4
˘ 
-Ş*
(
input_1
input_1˙˙˙˙˙˙˙˙˙"QŞN
%
output_1
output_1
%
output_2
output_2