(ns keras.layers.pooling.Layer
  "Abstract base layer class.

    # Properties
        input, output: Input/output tensor(s). Note that if the layer
            is used more than once (shared layer), this is ill-defined
            and will raise an exception. In such cases, use
            `layer.get_input_at(node_index)`.
        input_mask, output_mask: Mask tensors. Same caveats apply as
            input, output.
        input_shape: Shape tuple. Provided for convenience, but note
            that there may be cases in which this attribute is
            ill-defined (e.g. a shared layer with multiple input
            shapes), in which case requesting `input_shape` will raise
            an Exception. Prefer using
            `layer.get_input_shape_at(node_index)`.
        input_spec: List of InputSpec class instances
            each entry describes one required input:
                - ndim
                - dtype
            A layer with `n` input tensors must have
            an `input_spec` of length `n`.
        name: String, must be unique within a model.
        non_trainable_weights: List of variables.
        output_shape: Shape tuple. See `input_shape`.
        stateful: Boolean indicating whether the layer carries
            additional non-weight state. Used in, for instance, RNN
            cells to carry information between batches.
        supports_masking: Boolean indicator of whether the layer
            supports masking, typically for unused timesteps in a
            sequence.
        trainable: Boolean, whether the layer weights
            will be updated during training.
        trainable_weights: List of variables.
        uses_learning_phase: Whether any operation
            of the layer uses `K.in_training_phase()`
            or `K.in_test_phase()`.
        weights: The concatenation of the lists trainable_weights and
            non_trainable_weights (in this order).


    # Methods
        call(x, mask=None): Where the layer's logic lives.
        __call__(x, mask=None): Wrapper around the layer logic (`call`).
            If x is a Keras tensor:
                - Connect current layer with last layer from tensor:
                    `self._add_inbound_node(last_layer)`
                - Add layer to tensor history
            If layer is not built:
                - Build from x._keras_shape
        compute_mask(x, mask)
        compute_output_shape(input_shape)
        count_params()
        get_config()
        get_input_at(node_index)
        get_input_mask_at(node_index)
        get_input_shape_at(node_index)
        get_output_at(node_index)
        get_output_mask_at(node_index)
        get_output_shape_at(node_index)
        get_weights()
        set_weights(weights)

    # Class Methods
        from_config(config)

    # Internal methods:
        _add_inbound_node(layer, index=0)
        assert_input_compatibility()
        build(input_shape)
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce pooling (import-module "keras.layers.pooling"))

(defn Layer 
  "Abstract base layer class.

    # Properties
        input, output: Input/output tensor(s). Note that if the layer
            is used more than once (shared layer), this is ill-defined
            and will raise an exception. In such cases, use
            `layer.get_input_at(node_index)`.
        input_mask, output_mask: Mask tensors. Same caveats apply as
            input, output.
        input_shape: Shape tuple. Provided for convenience, but note
            that there may be cases in which this attribute is
            ill-defined (e.g. a shared layer with multiple input
            shapes), in which case requesting `input_shape` will raise
            an Exception. Prefer using
            `layer.get_input_shape_at(node_index)`.
        input_spec: List of InputSpec class instances
            each entry describes one required input:
                - ndim
                - dtype
            A layer with `n` input tensors must have
            an `input_spec` of length `n`.
        name: String, must be unique within a model.
        non_trainable_weights: List of variables.
        output_shape: Shape tuple. See `input_shape`.
        stateful: Boolean indicating whether the layer carries
            additional non-weight state. Used in, for instance, RNN
            cells to carry information between batches.
        supports_masking: Boolean indicator of whether the layer
            supports masking, typically for unused timesteps in a
            sequence.
        trainable: Boolean, whether the layer weights
            will be updated during training.
        trainable_weights: List of variables.
        uses_learning_phase: Whether any operation
            of the layer uses `K.in_training_phase()`
            or `K.in_test_phase()`.
        weights: The concatenation of the lists trainable_weights and
            non_trainable_weights (in this order).


    # Methods
        call(x, mask=None): Where the layer's logic lives.
        __call__(x, mask=None): Wrapper around the layer logic (`call`).
            If x is a Keras tensor:
                - Connect current layer with last layer from tensor:
                    `self._add_inbound_node(last_layer)`
                - Add layer to tensor history
            If layer is not built:
                - Build from x._keras_shape
        compute_mask(x, mask)
        compute_output_shape(input_shape)
        count_params()
        get_config()
        get_input_at(node_index)
        get_input_mask_at(node_index)
        get_input_shape_at(node_index)
        get_output_at(node_index)
        get_output_mask_at(node_index)
        get_output_shape_at(node_index)
        get_weights()
        set_weights(weights)

    # Class Methods
        from_config(config)

    # Internal methods:
        _add_inbound_node(layer, index=0)
        assert_input_compatibility()
        build(input_shape)
    "
  [  ]
  (py/call-attr pooling "Layer"   ))

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
    (py/call-attr-kw pooling "add_loss" [self] {:losses losses :inputs inputs }))

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
    (py/call-attr-kw pooling "add_update" [self] {:updates updates :inputs inputs }))

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
    (py/call-attr-kw pooling "add_weight" [] {:name name :shape shape :dtype dtype :initializer initializer :regularizer regularizer :trainable trainable :constraint constraint }))

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
    (py/call-attr-kw pooling "assert_input_compatibility" [self] {:inputs inputs }))

(defn build 
  "Creates the layer weights.

        Must be implemented on all layers that have weights.

        # Arguments
            input_shape: Keras tensor (future input to layer)
                or list/tuple of Keras tensors to reference
                for weight shape computations.
        "
  [self  & {:keys [input_shape]} ]
    (py/call-attr-kw pooling "build" [self] {:input_shape input_shape }))

(defn built 
  ""
  [ self ]
    (py/call-attr pooling "built"  self))

(defn call 
  "This is where the layer's logic lives.

        # Arguments
            inputs: Input tensor, or list/tuple of input tensors.
            **kwargs: Additional keyword arguments.

        # Returns
            A tensor or list/tuple of tensors.
        "
  [self  & {:keys [inputs]} ]
    (py/call-attr-kw pooling "call" [self] {:inputs inputs }))

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
    (py/call-attr-kw pooling "compute_mask" [self] {:inputs inputs :mask mask }))

(defn compute-output-shape 
  "Computes the output shape of the layer.

        Assumes that the layer will be built
        to match that input shape provided.

        # Arguments
            input_shape: Shape tuple (tuple of integers)
                or list of shape tuples (one per output tensor of the layer).
                Shape tuples can include None for free dimensions,
                instead of an integer.

        # Returns
            An input shape tuple.
        "
  [self  & {:keys [input_shape]} ]
    (py/call-attr-kw pooling "compute_output_shape" [self] {:input_shape input_shape }))

(defn count-params 
  "Counts the total number of scalars composing the weights.

        # Returns
            An integer count.

        # Raises
            RuntimeError: if the layer isn't yet built
                (in which case its weights aren't yet defined).
        "
  [ self ]
  (py/call-attr pooling "count_params"  self ))

(defn get-config 
  "Returns the config of the layer.

        A layer config is a Python dictionary (serializable)
        containing the configuration of a layer.
        The same layer can be reinstantiated later
        (without its trained weights) from this configuration.

        The config of a layer does not include connectivity
        information, nor the layer class name. These are handled
        by `Network` (one layer of abstraction above).

        # Returns
            Python dictionary.
        "
  [ self ]
  (py/call-attr pooling "get_config"  self ))

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
    (py/call-attr-kw pooling "get_input_at" [self] {:node_index node_index }))

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
    (py/call-attr-kw pooling "get_input_mask_at" [self] {:node_index node_index }))

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
    (py/call-attr-kw pooling "get_input_shape_at" [self] {:node_index node_index }))

(defn get-losses-for 
  ""
  [self  & {:keys [inputs]} ]
    (py/call-attr-kw pooling "get_losses_for" [self] {:inputs inputs }))

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
    (py/call-attr-kw pooling "get_output_at" [self] {:node_index node_index }))

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
    (py/call-attr-kw pooling "get_output_mask_at" [self] {:node_index node_index }))

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
    (py/call-attr-kw pooling "get_output_shape_at" [self] {:node_index node_index }))

(defn get-updates-for 
  ""
  [self  & {:keys [inputs]} ]
    (py/call-attr-kw pooling "get_updates_for" [self] {:inputs inputs }))

(defn get-weights 
  "Returns the current weights of the layer.

        # Returns
            Weights values as a list of numpy arrays.
        "
  [ self ]
  (py/call-attr pooling "get_weights"  self ))

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
    (py/call-attr pooling "input"  self))

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
    (py/call-attr pooling "input_mask"  self))

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
    (py/call-attr pooling "input_shape"  self))

(defn losses 
  ""
  [ self ]
    (py/call-attr pooling "losses"  self))

(defn non-trainable-weights 
  ""
  [ self ]
    (py/call-attr pooling "non_trainable_weights"  self))

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
    (py/call-attr pooling "output"  self))

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
    (py/call-attr pooling "output_mask"  self))

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
    (py/call-attr pooling "output_shape"  self))

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
    (py/call-attr-kw pooling "set_weights" [self] {:weights weights }))

(defn trainable-weights 
  ""
  [ self ]
    (py/call-attr pooling "trainable_weights"  self))

(defn updates 
  ""
  [ self ]
    (py/call-attr pooling "updates"  self))

(defn weights 
  ""
  [ self ]
    (py/call-attr pooling "weights"  self))
