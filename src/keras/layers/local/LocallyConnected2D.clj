(ns keras.layers.local.LocallyConnected2D
  "Locally-connected layer for 2D inputs.

    The `LocallyConnected2D` layer works similarly
    to the `Conv2D` layer, except that weights are unshared,
    that is, a different set of filters is applied at each
    different patch of the input.

    # Examples
    ```python
        # apply a 3x3 unshared weights convolution with 64 output filters
        # on a 32x32 image with `data_format=\"channels_last\"`:
        model = Sequential()
        model.add(LocallyConnected2D(64, (3, 3), input_shape=(32, 32, 3)))
        # now model.output_shape == (None, 30, 30, 64)
        # notice that this layer will consume (30*30)*(3*3*3*64)
        # + (30*30)*64 parameters

        # add a 3x3 unshared weights convolution on top, with 32 output filters:
        model.add(LocallyConnected2D(32, (3, 3)))
        # now model.output_shape == (None, 28, 28, 32)
    ```

    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        padding: Currently only support `\"valid\"` (case-insensitive).
            `\"same\"` will be supported in future.
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be \"channels_last\".
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
        4D tensor with shape:
        `(samples, channels, rows, cols)` if `data_format='channels_first'`
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if `data_format='channels_last'`.

    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce local (import-module "keras.layers.local"))

(defn LocallyConnected2D 
  "Locally-connected layer for 2D inputs.

    The `LocallyConnected2D` layer works similarly
    to the `Conv2D` layer, except that weights are unshared,
    that is, a different set of filters is applied at each
    different patch of the input.

    # Examples
    ```python
        # apply a 3x3 unshared weights convolution with 64 output filters
        # on a 32x32 image with `data_format=\"channels_last\"`:
        model = Sequential()
        model.add(LocallyConnected2D(64, (3, 3), input_shape=(32, 32, 3)))
        # now model.output_shape == (None, 30, 30, 64)
        # notice that this layer will consume (30*30)*(3*3*3*64)
        # + (30*30)*64 parameters

        # add a 3x3 unshared weights convolution on top, with 32 output filters:
        model.add(LocallyConnected2D(32, (3, 3)))
        # now model.output_shape == (None, 28, 28, 32)
    ```

    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        padding: Currently only support `\"valid\"` (case-insensitive).
            `\"same\"` will be supported in future.
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be \"channels_last\".
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
        4D tensor with shape:
        `(samples, channels, rows, cols)` if `data_format='channels_first'`
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if `data_format='channels_last'`.

    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    "
  [filters kernel_size & {:keys [strides padding data_format activation use_bias kernel_initializer bias_initializer kernel_regularizer bias_regularizer activity_regularizer kernel_constraint bias_constraint]
                       :or {strides (1, 1) padding "valid" use_bias true kernel_initializer "glorot_uniform" bias_initializer "zeros"}} ]
    (py/call-attr-kw local "LocallyConnected2D" [filters kernel_size] {:strides strides :padding padding :data_format data_format :activation activation :use_bias use_bias :kernel_initializer kernel_initializer :bias_initializer bias_initializer :kernel_regularizer kernel_regularizer :bias_regularizer bias_regularizer :activity_regularizer activity_regularizer :kernel_constraint kernel_constraint :bias_constraint bias_constraint }))
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
  [self losses  & {:keys [inputs]} ]
    (py/call-attr-kw self "add_loss" [losses] {:inputs inputs }))
(defn add-metric 
  "Adds metric tensor to the layer.

        # Arguments
            value: Metric tensor.
            name: String metric name.
        "
  [self value  & {:keys [name]} ]
    (py/call-attr-kw self "add_metric" [value] {:name name }))
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
  [self updates  & {:keys [inputs]} ]
    (py/call-attr-kw self "add_update" [updates] {:inputs inputs }))

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
  [self  & {:keys [name shape dtype initializer regularizer trainable constraint]
                       :or {trainable true}} ]
    (py/call-attr-kw self "add_weight" [] {:name name :shape shape :dtype dtype :initializer initializer :regularizer regularizer :trainable trainable :constraint constraint }))

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
  [ self inputs ]
  (py/call-attr self "assert_input_compatibility"  self inputs ))

(defn build 
  ""
  [ self input_shape ]
  (py/call-attr self "build"  self input_shape ))

(defn built 
  ""
  [ self ]
    (py/call-attr self "built"))

(defn call 
  ""
  [ self inputs ]
  (py/call-attr self "call"  self inputs ))
(defn compute-mask 
  "Computes an output mask tensor.

        # Arguments
            inputs: Tensor or list of tensors.
            mask: Tensor or list of tensors.

        # Returns
            None or a tensor (or list of tensors,
                one per output tensor of the layer).
        "
  [self inputs  & {:keys [mask]} ]
    (py/call-attr-kw self "compute_mask" [inputs] {:mask mask }))

(defn compute-output-shape 
  ""
  [ self input_shape ]
  (py/call-attr self "compute_output_shape"  self input_shape ))

(defn count-params 
  "Counts the total number of scalars composing the weights.

        # Returns
            An integer count.

        # Raises
            RuntimeError: if the layer isn't yet built
                (in which case its weights aren't yet defined).
        "
  [ self  ]
  (py/call-attr self "count_params"  self  ))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))

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
  [ self node_index ]
  (py/call-attr self "get_input_at"  self node_index ))

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
  [ self node_index ]
  (py/call-attr self "get_input_mask_at"  self node_index ))

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
  [ self node_index ]
  (py/call-attr self "get_input_shape_at"  self node_index ))

(defn get-losses-for 
  ""
  [ self inputs ]
  (py/call-attr self "get_losses_for"  self inputs ))

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
  [ self node_index ]
  (py/call-attr self "get_output_at"  self node_index ))

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
  [ self node_index ]
  (py/call-attr self "get_output_mask_at"  self node_index ))

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
  [ self node_index ]
  (py/call-attr self "get_output_shape_at"  self node_index ))

(defn get-updates-for 
  ""
  [ self inputs ]
  (py/call-attr self "get_updates_for"  self inputs ))

(defn get-weights 
  "Returns the current weights of the layer.

        # Returns
            Weights values as a list of numpy arrays.
        "
  [ self  ]
  (py/call-attr self "get_weights"  self  ))

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
    (py/call-attr self "input"))

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
    (py/call-attr self "input_mask"))

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
    (py/call-attr self "input_shape"))

(defn losses 
  ""
  [ self ]
    (py/call-attr self "losses"))

(defn metrics 
  ""
  [ self ]
    (py/call-attr self "metrics"))

(defn non-trainable-weights 
  ""
  [ self ]
    (py/call-attr self "non_trainable_weights"))

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
    (py/call-attr self "output"))

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
    (py/call-attr self "output_mask"))

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
    (py/call-attr self "output_shape"))

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
  [ self weights ]
  (py/call-attr self "set_weights"  self weights ))

(defn trainable-weights 
  ""
  [ self ]
    (py/call-attr self "trainable_weights"))

(defn updates 
  ""
  [ self ]
    (py/call-attr self "updates"))

(defn weights 
  ""
  [ self ]
    (py/call-attr self "weights"))
