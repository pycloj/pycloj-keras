(ns keras.layers.wrappers.Wrapper
  "Abstract wrapper base class.

    Wrappers take another layer and augment it in various ways.
    Do not use this class as a layer, it is only an abstract base class.
    Two usable wrappers are the `TimeDistributed` and `Bidirectional` wrappers.

    # Arguments
        layer: The layer to be wrapped.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce wrappers (import-module "keras.layers.wrappers"))

(defn Wrapper 
  "Abstract wrapper base class.

    Wrappers take another layer and augment it in various ways.
    Do not use this class as a layer, it is only an abstract base class.
    Two usable wrappers are the `TimeDistributed` and `Bidirectional` wrappers.

    # Arguments
        layer: The layer to be wrapped.
    "
  [  ]
  (py/call-attr wrappers "Wrapper"  ))

(defn activity-regularizer 
  ""
  [ self ]
    (py/call-attr self "activity_regularizer"))
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
  [self   & {:keys [input_shape]} ]
    (py/call-attr-kw self "build" [] {:input_shape input_shape }))

(defn built 
  ""
  [ self ]
    (py/call-attr self "built"))

(defn call 
  "This is where the layer's logic lives.

        # Arguments
            inputs: Input tensor, or list/tuple of input tensors.
            **kwargs: Additional keyword arguments.

        # Returns
            A tensor or list/tuple of tensors.
        "
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
  "Computes the output shape of the layer.

        Assumes that the layer will be built
        to match that input shape provided.

        # Arguments
            input_shape: Shape tuple (tuple of integers)
                or list of shape tuples (one per output tensor of the layer).
                Shape tuples can include None for free dimensions,
                instead of an integer.

        # Returns
            An output shape tuple.
        "
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
  [self   & {:keys [inputs]} ]
    (py/call-attr-kw self "get_losses_for" [] {:inputs inputs }))

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
  [self   & {:keys [inputs]} ]
    (py/call-attr-kw self "get_updates_for" [] {:inputs inputs }))

(defn get-weights 
  ""
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
  ""
  [ self weights ]
  (py/call-attr self "set_weights"  self weights ))

(defn trainable 
  ""
  [ self ]
    (py/call-attr self "trainable"))

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
