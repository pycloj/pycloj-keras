(ns keras.layers.convolutional-recurrent.ConvLSTM2D
  "Convolutional LSTM.

    It is similar to an LSTM layer, but the input transformations
    and recurrent transformations are both convolutional.

    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution).
        kernel_size: An integer or tuple/list of n integers, specifying the
            dimensions of the convolution window.
        strides: An integer or tuple/list of n integers,
            specifying the strides of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: One of `\"valid\"` or `\"same\"` (case-insensitive).
        data_format: A string,
            one of `\"channels_last\"` (default) or `\"channels_first\"`.
            The ordering of the dimensions in the inputs.
            `\"channels_last\"` corresponds to inputs with shape
            `(batch, time, ..., channels)`
            while `\"channels_first\"` corresponds to
            inputs with shape `(batch, time, channels, ...)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be `\"channels_last\"`.
        dilation_rate: An integer or tuple/list of n integers, specifying
            the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. \"linear\" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        unit_forget_bias: Boolean.
            If True, add 1 to the bias of the forget gate at initialization.
            Use in combination with `bias_initializer=\"zeros\"`.
            This is recommended in [Jozefowicz et al.]
            (http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its \"activation\").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.

    # Input shape
        - if data_format='channels_first'
            5D tensor with shape:
            `(samples, time, channels, rows, cols)`
        - if data_format='channels_last'
            5D tensor with shape:
            `(samples, time, rows, cols, channels)`

    # Output shape
        - if `return_sequences`
             - if data_format='channels_first'
                5D tensor with shape:
                `(samples, time, filters, output_row, output_col)`
             - if data_format='channels_last'
                5D tensor with shape:
                `(samples, time, output_row, output_col, filters)`
        - else
            - if data_format='channels_first'
                4D tensor with shape:
                `(samples, filters, output_row, output_col)`
            - if data_format='channels_last'
                4D tensor with shape:
                `(samples, output_row, output_col, filters)`
            where o_row and o_col depend on the shape of the filter and
            the padding

    # Raises
        ValueError: in case of invalid constructor arguments.

    # References
        - [Convolutional LSTM Network: A Machine Learning Approach for
        Precipitation Nowcasting](http://arxiv.org/abs/1506.04214v1)
        The current implementation does not include the feedback loop on the
        cells output
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce convolutional-recurrent (import-module "keras.layers.convolutional_recurrent"))

(defn ConvLSTM2D 
  "Convolutional LSTM.

    It is similar to an LSTM layer, but the input transformations
    and recurrent transformations are both convolutional.

    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution).
        kernel_size: An integer or tuple/list of n integers, specifying the
            dimensions of the convolution window.
        strides: An integer or tuple/list of n integers,
            specifying the strides of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: One of `\"valid\"` or `\"same\"` (case-insensitive).
        data_format: A string,
            one of `\"channels_last\"` (default) or `\"channels_first\"`.
            The ordering of the dimensions in the inputs.
            `\"channels_last\"` corresponds to inputs with shape
            `(batch, time, ..., channels)`
            while `\"channels_first\"` corresponds to
            inputs with shape `(batch, time, channels, ...)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be `\"channels_last\"`.
        dilation_rate: An integer or tuple/list of n integers, specifying
            the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. \"linear\" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        unit_forget_bias: Boolean.
            If True, add 1 to the bias of the forget gate at initialization.
            Use in combination with `bias_initializer=\"zeros\"`.
            This is recommended in [Jozefowicz et al.]
            (http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its \"activation\").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.

    # Input shape
        - if data_format='channels_first'
            5D tensor with shape:
            `(samples, time, channels, rows, cols)`
        - if data_format='channels_last'
            5D tensor with shape:
            `(samples, time, rows, cols, channels)`

    # Output shape
        - if `return_sequences`
             - if data_format='channels_first'
                5D tensor with shape:
                `(samples, time, filters, output_row, output_col)`
             - if data_format='channels_last'
                5D tensor with shape:
                `(samples, time, output_row, output_col, filters)`
        - else
            - if data_format='channels_first'
                4D tensor with shape:
                `(samples, filters, output_row, output_col)`
            - if data_format='channels_last'
                4D tensor with shape:
                `(samples, output_row, output_col, filters)`
            where o_row and o_col depend on the shape of the filter and
            the padding

    # Raises
        ValueError: in case of invalid constructor arguments.

    # References
        - [Convolutional LSTM Network: A Machine Learning Approach for
        Precipitation Nowcasting](http://arxiv.org/abs/1506.04214v1)
        The current implementation does not include the feedback loop on the
        cells output
    "
  [ & {:keys [filters kernel_size strides padding data_format dilation_rate activation recurrent_activation use_bias kernel_initializer recurrent_initializer bias_initializer unit_forget_bias kernel_regularizer recurrent_regularizer bias_regularizer activity_regularizer kernel_constraint recurrent_constraint bias_constraint return_sequences go_backwards stateful dropout recurrent_dropout]
       :or {strides (1, 1) padding "valid" dilation_rate (1, 1) activation "tanh" recurrent_activation "hard_sigmoid" use_bias true kernel_initializer "glorot_uniform" recurrent_initializer "orthogonal" bias_initializer "zeros" unit_forget_bias true return_sequences false go_backwards false stateful false dropout 0.0 recurrent_dropout 0.0}} ]
  
   (py/call-attr-kw convolutional-recurrent "ConvLSTM2D" [] {:filters filters :kernel_size kernel_size :strides strides :padding padding :data_format data_format :dilation_rate dilation_rate :activation activation :recurrent_activation recurrent_activation :use_bias use_bias :kernel_initializer kernel_initializer :recurrent_initializer recurrent_initializer :bias_initializer bias_initializer :unit_forget_bias unit_forget_bias :kernel_regularizer kernel_regularizer :recurrent_regularizer recurrent_regularizer :bias_regularizer bias_regularizer :activity_regularizer activity_regularizer :kernel_constraint kernel_constraint :recurrent_constraint recurrent_constraint :bias_constraint bias_constraint :return_sequences return_sequences :go_backwards go_backwards :stateful stateful :dropout dropout :recurrent_dropout recurrent_dropout }))

(defn activation 
  ""
  [ self ]
    (py/call-attr convolutional-recurrent "activation"  self))

(defn add-loss 
  "Adds losses to the layer.

        The loss may potentially be conditional on some inputs tensors,
        for instance activity losses are conditional on the layer's inputs.

        # Arguments
            losses: loss tensor or list of loss tensors
                to add to the layer.
            inputs: input tensor or list of inputs tensors to mark
                the losses as conditional on these inputs.
                If None is passed, the loss is assumed unconditional
                (e.g. L2 weight regularization, which only depends
                on the layer's weights variables, not on any inputs tensors).
        "
  [self  & {:keys [losses inputs]} ]
    (py/call-attr-kw convolutional-recurrent "add_loss" [self] {:losses losses :inputs inputs }))

(defn add-update 
  "Adds updates to the layer.

        The updates may potentially be conditional on some inputs tensors,
        for instance batch norm updates are conditional on the layer's inputs.

        # Arguments
            updates: update op or list of update ops
                to add to the layer.
            inputs: input tensor or list of inputs tensors to mark
                the updates as conditional on these inputs.
                If None is passed, the updates are assumed unconditional.
        "
  [self  & {:keys [updates inputs]} ]
    (py/call-attr-kw convolutional-recurrent "add_update" [self] {:updates updates :inputs inputs }))

(defn add-weight 
  "Adds a weight variable to the layer.

        # Arguments
            name: String, the name for the weight variable.
            shape: The shape tuple of the weight.
            dtype: The dtype of the weight.
            initializer: An Initializer instance (callable).
            regularizer: An optional Regularizer instance.
            trainable: A boolean, whether the weight should
                be trained via backprop or not (assuming
                that the layer itself is also trainable).
            constraint: An optional Constraint instance.

        # Returns
            The created weight variable.
        "
  [self & {:keys [name shape dtype initializer regularizer trainable constraint]
                       :or {trainable true}} ]
    (py/call-attr-kw convolutional-recurrent "add_weight" [] {:name name :shape shape :dtype dtype :initializer initializer :regularizer regularizer :trainable trainable :constraint constraint }))

(defn assert-input-compatibility 
  "Checks compatibility between the layer and provided inputs.

        This checks that the tensor(s) `input`
        verify the input assumptions of the layer
        (if any). If not, exceptions are raised.

        # Arguments
            inputs: input tensor or list of input tensors.

        # Raises
            ValueError: in case of mismatch between
                the provided inputs and the expectations of the layer.
        "
  [self  & {:keys [inputs]} ]
    (py/call-attr-kw convolutional-recurrent "assert_input_compatibility" [self] {:inputs inputs }))

(defn bias-constraint 
  ""
  [ self ]
    (py/call-attr convolutional-recurrent "bias_constraint"  self))

(defn bias-initializer 
  ""
  [ self ]
    (py/call-attr convolutional-recurrent "bias_initializer"  self))

(defn bias-regularizer 
  ""
  [ self ]
    (py/call-attr convolutional-recurrent "bias_regularizer"  self))

(defn build 
  ""
  [self  & {:keys [input_shape]} ]
    (py/call-attr-kw convolutional-recurrent "build" [self] {:input_shape input_shape }))

(defn built 
  ""
  [ self ]
    (py/call-attr convolutional-recurrent "built"  self))

(defn call 
  ""
  [self  & {:keys [inputs mask training initial_state]} ]
    (py/call-attr-kw convolutional-recurrent "call" [self] {:inputs inputs :mask mask :training training :initial_state initial_state }))

(defn compute-mask 
  ""
  [self  & {:keys [inputs mask]} ]
    (py/call-attr-kw convolutional-recurrent "compute_mask" [self] {:inputs inputs :mask mask }))

(defn compute-output-shape 
  ""
  [self  & {:keys [input_shape]} ]
    (py/call-attr-kw convolutional-recurrent "compute_output_shape" [self] {:input_shape input_shape }))

(defn count-params 
  "Counts the total number of scalars composing the weights.

        # Returns
            An integer count.

        # Raises
            RuntimeError: if the layer isn't yet built
                (in which case its weights aren't yet defined).
        "
  [ self ]
  (py/call-attr convolutional-recurrent "count_params"  self ))

(defn data-format 
  ""
  [ self ]
    (py/call-attr convolutional-recurrent "data_format"  self))

(defn dilation-rate 
  ""
  [ self ]
    (py/call-attr convolutional-recurrent "dilation_rate"  self))

(defn dropout 
  ""
  [ self ]
    (py/call-attr convolutional-recurrent "dropout"  self))

(defn filters 
  ""
  [ self ]
    (py/call-attr convolutional-recurrent "filters"  self))

(defn get-config 
  ""
  [ self ]
  (py/call-attr convolutional-recurrent "get_config"  self ))

(defn get-initial-state 
  ""
  [self  & {:keys [inputs]} ]
    (py/call-attr-kw convolutional-recurrent "get_initial_state" [self] {:inputs inputs }))

(defn get-input-at 
  "Retrieves the input tensor(s) of a layer at a given node.

        # Arguments
            node_index: Integer, index of the node
                from which to retrieve the attribute.
                E.g. `node_index=0` will correspond to the
                first time the layer was called.

        # Returns
            A tensor (or list of tensors if the layer has multiple inputs).
        "
  [self  & {:keys [node_index]} ]
    (py/call-attr-kw convolutional-recurrent "get_input_at" [self] {:node_index node_index }))

(defn get-input-mask-at 
  "Retrieves the input mask tensor(s) of a layer at a given node.

        # Arguments
            node_index: Integer, index of the node
                from which to retrieve the attribute.
                E.g. `node_index=0` will correspond to the
                first time the layer was called.

        # Returns
            A mask tensor
            (or list of tensors if the layer has multiple inputs).
        "
  [self  & {:keys [node_index]} ]
    (py/call-attr-kw convolutional-recurrent "get_input_mask_at" [self] {:node_index node_index }))

(defn get-input-shape-at 
  "Retrieves the input shape(s) of a layer at a given node.

        # Arguments
            node_index: Integer, index of the node
                from which to retrieve the attribute.
                E.g. `node_index=0` will correspond to the
                first time the layer was called.

        # Returns
            A shape tuple
            (or list of shape tuples if the layer has multiple inputs).
        "
  [self  & {:keys [node_index]} ]
    (py/call-attr-kw convolutional-recurrent "get_input_shape_at" [self] {:node_index node_index }))

(defn get-losses-for 
  ""
  [self  & {:keys [inputs]} ]
    (py/call-attr-kw convolutional-recurrent "get_losses_for" [self] {:inputs inputs }))

(defn get-output-at 
  "Retrieves the output tensor(s) of a layer at a given node.

        # Arguments
            node_index: Integer, index of the node
                from which to retrieve the attribute.
                E.g. `node_index=0` will correspond to the
                first time the layer was called.

        # Returns
            A tensor (or list of tensors if the layer has multiple outputs).
        "
  [self  & {:keys [node_index]} ]
    (py/call-attr-kw convolutional-recurrent "get_output_at" [self] {:node_index node_index }))

(defn get-output-mask-at 
  "Retrieves the output mask tensor(s) of a layer at a given node.

        # Arguments
            node_index: Integer, index of the node
                from which to retrieve the attribute.
                E.g. `node_index=0` will correspond to the
                first time the layer was called.

        # Returns
            A mask tensor
            (or list of tensors if the layer has multiple outputs).
        "
  [self  & {:keys [node_index]} ]
    (py/call-attr-kw convolutional-recurrent "get_output_mask_at" [self] {:node_index node_index }))

(defn get-output-shape-at 
  "Retrieves the output shape(s) of a layer at a given node.

        # Arguments
            node_index: Integer, index of the node
                from which to retrieve the attribute.
                E.g. `node_index=0` will correspond to the
                first time the layer was called.

        # Returns
            A shape tuple
            (or list of shape tuples if the layer has multiple outputs).
        "
  [self  & {:keys [node_index]} ]
    (py/call-attr-kw convolutional-recurrent "get_output_shape_at" [self] {:node_index node_index }))

(defn get-updates-for 
  ""
  [self  & {:keys [inputs]} ]
    (py/call-attr-kw convolutional-recurrent "get_updates_for" [self] {:inputs inputs }))

(defn get-weights 
  "Returns the current weights of the layer.

        # Returns
            Weights values as a list of numpy arrays.
        "
  [ self ]
  (py/call-attr convolutional-recurrent "get_weights"  self ))

(defn input 
  "Retrieves the input tensor(s) of a layer.

        Only applicable if the layer has exactly one inbound node,
        i.e. if it is connected to one incoming layer.

        # Returns
            Input tensor or list of input tensors.

        # Raises
            AttributeError: if the layer is connected to
            more than one incoming layers.
        "
  [ self ]
    (py/call-attr convolutional-recurrent "input"  self))

(defn input-mask 
  "Retrieves the input mask tensor(s) of a layer.

        Only applicable if the layer has exactly one inbound node,
        i.e. if it is connected to one incoming layer.

        # Returns
            Input mask tensor (potentially None) or list of input
            mask tensors.

        # Raises
            AttributeError: if the layer is connected to
            more than one incoming layers.
        "
  [ self ]
    (py/call-attr convolutional-recurrent "input_mask"  self))

(defn input-shape 
  "Retrieves the input shape tuple(s) of a layer.

        Only applicable if the layer has exactly one inbound node,
        i.e. if it is connected to one incoming layer.

        # Returns
            Input shape tuple
            (or list of input shape tuples, one tuple per input tensor).

        # Raises
            AttributeError: if the layer is connected to
            more than one incoming layers.
        "
  [ self ]
    (py/call-attr convolutional-recurrent "input_shape"  self))

(defn kernel-constraint 
  ""
  [ self ]
    (py/call-attr convolutional-recurrent "kernel_constraint"  self))

(defn kernel-initializer 
  ""
  [ self ]
    (py/call-attr convolutional-recurrent "kernel_initializer"  self))

(defn kernel-regularizer 
  ""
  [ self ]
    (py/call-attr convolutional-recurrent "kernel_regularizer"  self))

(defn kernel-size 
  ""
  [ self ]
    (py/call-attr convolutional-recurrent "kernel_size"  self))

(defn losses 
  ""
  [ self ]
    (py/call-attr convolutional-recurrent "losses"  self))

(defn non-trainable-weights 
  ""
  [ self ]
    (py/call-attr convolutional-recurrent "non_trainable_weights"  self))

(defn output 
  "Retrieves the output tensor(s) of a layer.

        Only applicable if the layer has exactly one inbound node,
        i.e. if it is connected to one incoming layer.

        # Returns
            Output tensor or list of output tensors.

        # Raises
            AttributeError: if the layer is connected to
            more than one incoming layers.
        "
  [ self ]
    (py/call-attr convolutional-recurrent "output"  self))

(defn output-mask 
  "Retrieves the output mask tensor(s) of a layer.

        Only applicable if the layer has exactly one inbound node,
        i.e. if it is connected to one incoming layer.

        # Returns
            Output mask tensor (potentially None) or list of output
            mask tensors.

        # Raises
            AttributeError: if the layer is connected to
            more than one incoming layers.
        "
  [ self ]
    (py/call-attr convolutional-recurrent "output_mask"  self))

(defn output-shape 
  "Retrieves the output shape tuple(s) of a layer.

        Only applicable if the layer has one inbound node,
        or if all inbound nodes have the same output shape.

        # Returns
            Output shape tuple
            (or list of input shape tuples, one tuple per output tensor).

        # Raises
            AttributeError: if the layer is connected to
            more than one incoming layers.
        "
  [ self ]
    (py/call-attr convolutional-recurrent "output_shape"  self))

(defn padding 
  ""
  [ self ]
    (py/call-attr convolutional-recurrent "padding"  self))

(defn recurrent-activation 
  ""
  [ self ]
    (py/call-attr convolutional-recurrent "recurrent_activation"  self))

(defn recurrent-constraint 
  ""
  [ self ]
    (py/call-attr convolutional-recurrent "recurrent_constraint"  self))

(defn recurrent-dropout 
  ""
  [ self ]
    (py/call-attr convolutional-recurrent "recurrent_dropout"  self))

(defn recurrent-initializer 
  ""
  [ self ]
    (py/call-attr convolutional-recurrent "recurrent_initializer"  self))

(defn recurrent-regularizer 
  ""
  [ self ]
    (py/call-attr convolutional-recurrent "recurrent_regularizer"  self))

(defn reset-states 
  ""
  [self  & {:keys [states]} ]
    (py/call-attr-kw convolutional-recurrent "reset_states" [self] {:states states }))

(defn set-weights 
  "Sets the weights of the layer, from Numpy arrays.

        # Arguments
            weights: a list of Numpy arrays. The number
                of arrays and their shape must match
                number of the dimensions of the weights
                of the layer (i.e. it should match the
                output of `get_weights`).

        # Raises
            ValueError: If the provided weights list does not match the
                layer's specifications.
        "
  [self  & {:keys [weights]} ]
    (py/call-attr-kw convolutional-recurrent "set_weights" [self] {:weights weights }))

(defn states 
  ""
  [ self ]
    (py/call-attr convolutional-recurrent "states"  self))

(defn strides 
  ""
  [ self ]
    (py/call-attr convolutional-recurrent "strides"  self))

(defn trainable-weights 
  ""
  [ self ]
    (py/call-attr convolutional-recurrent "trainable_weights"  self))

(defn unit-forget-bias 
  ""
  [ self ]
    (py/call-attr convolutional-recurrent "unit_forget_bias"  self))

(defn updates 
  ""
  [ self ]
    (py/call-attr convolutional-recurrent "updates"  self))

(defn use-bias 
  ""
  [ self ]
    (py/call-attr convolutional-recurrent "use_bias"  self))

(defn weights 
  ""
  [ self ]
    (py/call-attr convolutional-recurrent "weights"  self))
