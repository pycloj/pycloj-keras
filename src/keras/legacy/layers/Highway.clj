(ns keras.legacy.layers.Highway
  "Densely connected highway network.
    Highway layers are a natural extension of LSTMs to feedforward networks.
    # Arguments
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. \"linear\" activation: a(x) = x).
        weights: list of Numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, output_dim)`
            and (output_dim,) for weights and biases respectively.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        bias: whether to include a bias
            (i.e. make the layer affine rather than linear).
        input_dim: dimensionality of the input (integer). This argument
            (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.
    # Output shape
        2D tensor with shape: `(nb_samples, input_dim)`.
    # References
        - [Highway Networks](http://arxiv.org/abs/1505.00387v2)
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce layers (import-module "keras.legacy.layers"))

(defn Highway 
  "Densely connected highway network.
    Highway layers are a natural extension of LSTMs to feedforward networks.
    # Arguments
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. \"linear\" activation: a(x) = x).
        weights: list of Numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, output_dim)`
            and (output_dim,) for weights and biases respectively.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        bias: whether to include a bias
            (i.e. make the layer affine rather than linear).
        input_dim: dimensionality of the input (integer). This argument
            (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.
    # Output shape
        2D tensor with shape: `(nb_samples, input_dim)`.
    # References
        - [Highway Networks](http://arxiv.org/abs/1505.00387v2)
    "
  [ & {:keys [init activation weights W_regularizer b_regularizer activity_regularizer W_constraint b_constraint bias input_dim]
       :or {init "glorot_uniform" bias true}} ]
  
   (py/call-attr-kw layers "Highway" [] {:init init :activation activation :weights weights :W_regularizer W_regularizer :b_regularizer b_regularizer :activity_regularizer activity_regularizer :W_constraint W_constraint :b_constraint b_constraint :bias bias :input_dim input_dim }))

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
    (py/call-attr-kw layers "add_loss" [self] {:losses losses :inputs inputs }))

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
    (py/call-attr-kw layers "add_update" [self] {:updates updates :inputs inputs }))

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
    (py/call-attr-kw layers "add_weight" [] {:name name :shape shape :dtype dtype :initializer initializer :regularizer regularizer :trainable trainable :constraint constraint }))

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
    (py/call-attr-kw layers "assert_input_compatibility" [self] {:inputs inputs }))

(defn build 
  ""
  [self  & {:keys [input_shape]} ]
    (py/call-attr-kw layers "build" [self] {:input_shape input_shape }))

(defn built 
  ""
  [ self ]
    (py/call-attr layers "built"  self))

(defn call 
  ""
  [self  & {:keys [x]} ]
    (py/call-attr-kw layers "call" [self] {:x x }))

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
    (py/call-attr-kw layers "compute_mask" [self] {:inputs inputs :mask mask }))

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
    (py/call-attr-kw layers "compute_output_shape" [self] {:input_shape input_shape }))

(defn count-params 
  "Counts the total number of scalars composing the weights.

        # Returns
            An integer count.

        # Raises
            RuntimeError: if the layer isn't yet built
                (in which case its weights aren't yet defined).
        "
  [ self ]
  (py/call-attr layers "count_params"  self ))

(defn get-config 
  ""
  [ self ]
  (py/call-attr layers "get_config"  self ))

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
    (py/call-attr-kw layers "get_input_at" [self] {:node_index node_index }))

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
    (py/call-attr-kw layers "get_input_mask_at" [self] {:node_index node_index }))

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
    (py/call-attr-kw layers "get_input_shape_at" [self] {:node_index node_index }))

(defn get-losses-for 
  ""
  [self  & {:keys [inputs]} ]
    (py/call-attr-kw layers "get_losses_for" [self] {:inputs inputs }))

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
    (py/call-attr-kw layers "get_output_at" [self] {:node_index node_index }))

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
    (py/call-attr-kw layers "get_output_mask_at" [self] {:node_index node_index }))

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
    (py/call-attr-kw layers "get_output_shape_at" [self] {:node_index node_index }))

(defn get-updates-for 
  ""
  [self  & {:keys [inputs]} ]
    (py/call-attr-kw layers "get_updates_for" [self] {:inputs inputs }))

(defn get-weights 
  "Returns the current weights of the layer.

        # Returns
            Weights values as a list of numpy arrays.
        "
  [ self ]
  (py/call-attr layers "get_weights"  self ))

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
    (py/call-attr layers "input"  self))

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
    (py/call-attr layers "input_mask"  self))

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
    (py/call-attr layers "input_shape"  self))

(defn losses 
  ""
  [ self ]
    (py/call-attr layers "losses"  self))

(defn non-trainable-weights 
  ""
  [ self ]
    (py/call-attr layers "non_trainable_weights"  self))

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
    (py/call-attr layers "output"  self))

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
    (py/call-attr layers "output_mask"  self))

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
    (py/call-attr layers "output_shape"  self))

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
    (py/call-attr-kw layers "set_weights" [self] {:weights weights }))

(defn trainable-weights 
  ""
  [ self ]
    (py/call-attr layers "trainable_weights"  self))

(defn updates 
  ""
  [ self ]
    (py/call-attr layers "updates"  self))

(defn weights 
  ""
  [ self ]
    (py/call-attr layers "weights"  self))
