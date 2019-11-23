(ns keras.layers.convolutional.Conv1D
  "1D convolution layer (e.g. temporal convolution).

    This layer creates a convolution kernel that is convolved
    with the layer input over a single spatial (or temporal) dimension
    to produce a tensor of outputs.
    If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`,
    it is applied to the outputs as well.

    When using this layer as the first layer in a model,
    provide an `input_shape` argument
    (tuple of integers or `None`, e.g.
    `(10, 128)` for sequences of 10 vectors of 128-dimensional vectors,
    or `(None, 128)` for variable-length sequences of 128-dimensional vectors.

    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of a single integer,
            specifying the length of the 1D convolution window.
        strides: An integer or tuple/list of a single integer,
            specifying the stride length of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: One of `\"valid\"`, `\"causal\"` or `\"same\"` (case-insensitive).
            `\"valid\"` means \"no padding\".
            `\"same\"` results in padding the input such that
            the output has the same length as the original input.
            `\"causal\"` results in causal (dilated) convolutions,
            e.g. `output[t]` does not depend on `input[t + 1:]`.
            A zero padding is used such that
            the output has the same length as the original input.
            Useful when modeling temporal data where the model
            should not violate the temporal order. See
            [WaveNet: A Generative Model for Raw Audio, section 2.1]
            (https://arxiv.org/abs/1609.03499).
        data_format: A string,
            one of `\"channels_last\"` (default) or `\"channels_first\"`.
            The ordering of the dimensions in the inputs.
            `\"channels_last\"` corresponds to inputs with shape
            `(batch, steps, channels)`
            (default format for temporal data in Keras)
            while `\"channels_first\"` corresponds to inputs
            with shape `(batch, channels, steps)`.
        dilation_rate: an integer or tuple/list of a single integer, specifying
            the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. \"linear\" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its \"activation\").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).

    # Input shape
        3D tensor with shape: `(batch, steps, channels)`

    # Output shape
        3D tensor with shape: `(batch, new_steps, filters)`
        `steps` value might have changed due to padding or strides.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce convolutional (import-module "keras.layers.convolutional"))

(defn Conv1D 
  "1D convolution layer (e.g. temporal convolution).

    This layer creates a convolution kernel that is convolved
    with the layer input over a single spatial (or temporal) dimension
    to produce a tensor of outputs.
    If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`,
    it is applied to the outputs as well.

    When using this layer as the first layer in a model,
    provide an `input_shape` argument
    (tuple of integers or `None`, e.g.
    `(10, 128)` for sequences of 10 vectors of 128-dimensional vectors,
    or `(None, 128)` for variable-length sequences of 128-dimensional vectors.

    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of a single integer,
            specifying the length of the 1D convolution window.
        strides: An integer or tuple/list of a single integer,
            specifying the stride length of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: One of `\"valid\"`, `\"causal\"` or `\"same\"` (case-insensitive).
            `\"valid\"` means \"no padding\".
            `\"same\"` results in padding the input such that
            the output has the same length as the original input.
            `\"causal\"` results in causal (dilated) convolutions,
            e.g. `output[t]` does not depend on `input[t + 1:]`.
            A zero padding is used such that
            the output has the same length as the original input.
            Useful when modeling temporal data where the model
            should not violate the temporal order. See
            [WaveNet: A Generative Model for Raw Audio, section 2.1]
            (https://arxiv.org/abs/1609.03499).
        data_format: A string,
            one of `\"channels_last\"` (default) or `\"channels_first\"`.
            The ordering of the dimensions in the inputs.
            `\"channels_last\"` corresponds to inputs with shape
            `(batch, steps, channels)`
            (default format for temporal data in Keras)
            while `\"channels_first\"` corresponds to inputs
            with shape `(batch, channels, steps)`.
        dilation_rate: an integer or tuple/list of a single integer, specifying
            the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. \"linear\" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its \"activation\").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).

    # Input shape
        3D tensor with shape: `(batch, steps, channels)`

    # Output shape
        3D tensor with shape: `(batch, new_steps, filters)`
        `steps` value might have changed due to padding or strides.
    "
  [ & {:keys [filters kernel_size strides padding data_format dilation_rate activation use_bias kernel_initializer bias_initializer kernel_regularizer bias_regularizer activity_regularizer kernel_constraint bias_constraint]
       :or {strides 1 padding "valid" data_format "channels_last" dilation_rate 1 use_bias true kernel_initializer "glorot_uniform" bias_initializer "zeros"}} ]
  
   (py/call-attr-kw convolutional "Conv1D" [] {:filters filters :kernel_size kernel_size :strides strides :padding padding :data_format data_format :dilation_rate dilation_rate :activation activation :use_bias use_bias :kernel_initializer kernel_initializer :bias_initializer bias_initializer :kernel_regularizer kernel_regularizer :bias_regularizer bias_regularizer :activity_regularizer activity_regularizer :kernel_constraint kernel_constraint :bias_constraint bias_constraint }))

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
    (py/call-attr-kw convolutional "add_loss" [self] {:losses losses :inputs inputs }))

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
    (py/call-attr-kw convolutional "add_update" [self] {:updates updates :inputs inputs }))

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
    (py/call-attr-kw convolutional "add_weight" [] {:name name :shape shape :dtype dtype :initializer initializer :regularizer regularizer :trainable trainable :constraint constraint }))

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
    (py/call-attr-kw convolutional "assert_input_compatibility" [self] {:inputs inputs }))

(defn build 
  ""
  [self  & {:keys [input_shape]} ]
    (py/call-attr-kw convolutional "build" [self] {:input_shape input_shape }))

(defn built 
  ""
  [ self ]
    (py/call-attr convolutional "built"  self))

(defn call 
  ""
  [self  & {:keys [inputs]} ]
    (py/call-attr-kw convolutional "call" [self] {:inputs inputs }))

(defn compute-mask 
  "Computes an output mask tensor.

        # Arguments
            inputs: Tensor or list of tensors.
            mask: Tensor or list of tensors.

        # Returns
            None or a tensor (or list of tensors,
                one per output tensor of the layer).
        "
  [self  & {:keys [inputs mask]} ]
    (py/call-attr-kw convolutional "compute_mask" [self] {:inputs inputs :mask mask }))

(defn compute-output-shape 
  ""
  [self  & {:keys [input_shape]} ]
    (py/call-attr-kw convolutional "compute_output_shape" [self] {:input_shape input_shape }))

(defn count-params 
  "Counts the total number of scalars composing the weights.

        # Returns
            An integer count.

        # Raises
            RuntimeError: if the layer isn't yet built
                (in which case its weights aren't yet defined).
        "
  [ self ]
  (py/call-attr convolutional "count_params"  self ))

(defn get-config 
  ""
  [ self ]
  (py/call-attr convolutional "get_config"  self ))

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
    (py/call-attr-kw convolutional "get_input_at" [self] {:node_index node_index }))

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
    (py/call-attr-kw convolutional "get_input_mask_at" [self] {:node_index node_index }))

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
    (py/call-attr-kw convolutional "get_input_shape_at" [self] {:node_index node_index }))

(defn get-losses-for 
  ""
  [self  & {:keys [inputs]} ]
    (py/call-attr-kw convolutional "get_losses_for" [self] {:inputs inputs }))

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
    (py/call-attr-kw convolutional "get_output_at" [self] {:node_index node_index }))

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
    (py/call-attr-kw convolutional "get_output_mask_at" [self] {:node_index node_index }))

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
    (py/call-attr-kw convolutional "get_output_shape_at" [self] {:node_index node_index }))

(defn get-updates-for 
  ""
  [self  & {:keys [inputs]} ]
    (py/call-attr-kw convolutional "get_updates_for" [self] {:inputs inputs }))

(defn get-weights 
  "Returns the current weights of the layer.

        # Returns
            Weights values as a list of numpy arrays.
        "
  [ self ]
  (py/call-attr convolutional "get_weights"  self ))

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
    (py/call-attr convolutional "input"  self))

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
    (py/call-attr convolutional "input_mask"  self))

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
    (py/call-attr convolutional "input_shape"  self))

(defn losses 
  ""
  [ self ]
    (py/call-attr convolutional "losses"  self))

(defn non-trainable-weights 
  ""
  [ self ]
    (py/call-attr convolutional "non_trainable_weights"  self))

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
    (py/call-attr convolutional "output"  self))

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
    (py/call-attr convolutional "output_mask"  self))

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
    (py/call-attr convolutional "output_shape"  self))

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
    (py/call-attr-kw convolutional "set_weights" [self] {:weights weights }))

(defn trainable-weights 
  ""
  [ self ]
    (py/call-attr convolutional "trainable_weights"  self))

(defn updates 
  ""
  [ self ]
    (py/call-attr convolutional "updates"  self))

(defn weights 
  ""
  [ self ]
    (py/call-attr convolutional "weights"  self))
