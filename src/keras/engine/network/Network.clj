(ns keras.engine.network.Network
  "A Network is a directed acyclic graph of layers.

    It is the topological form of a \"model\". A Model
    is simply a Network with added training routines.

    # Properties
        name
        inputs
        outputs
        layers
        input_spec (list of class instances)
            each entry describes one required input:
                - ndim
                - dtype
        trainable (boolean)
        dtype
        input_shape
        output_shape
        weights (list of variables)
        trainable_weights (list of variables)
        non_trainable_weights (list of variables)
        losses
        updates
        state_updates
        stateful

    # Methods
        __call__
        summary
        get_layer
        get_weights
        set_weights
        get_config
        compute_output_shape
        save
        add_loss
        add_update
        get_losses_for
        get_updates_for
        to_json
        to_yaml
        reset_states

    # Class Methods
        from_config

    # Raises
        TypeError: if input tensors are not Keras tensors
            (tensors returned by `Input`).
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce network (import-module "keras.engine.network"))

(defn Network 
  "A Network is a directed acyclic graph of layers.

    It is the topological form of a \"model\". A Model
    is simply a Network with added training routines.

    # Properties
        name
        inputs
        outputs
        layers
        input_spec (list of class instances)
            each entry describes one required input:
                - ndim
                - dtype
        trainable (boolean)
        dtype
        input_shape
        output_shape
        weights (list of variables)
        trainable_weights (list of variables)
        non_trainable_weights (list of variables)
        losses
        updates
        state_updates
        stateful

    # Methods
        __call__
        summary
        get_layer
        get_weights
        set_weights
        get_config
        compute_output_shape
        save
        add_loss
        add_update
        get_losses_for
        get_updates_for
        to_json
        to_yaml
        reset_states

    # Class Methods
        from_config

    # Raises
        TypeError: if input tensors are not Keras tensors
            (tensors returned by `Input`).
    "
  [  ]
  (py/call-attr network "Network"  ))
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
  "Creates the layer weights.

        Must be implemented on all layers that have weights.

        # Arguments
            input_shape: Keras tensor (future input to layer)
                or list/tuple of Keras tensors to reference
                for weight shape computations.
        "
  [ self input_shape ]
  (py/call-attr self "build"  self input_shape ))

(defn built 
  ""
  [ self ]
    (py/call-attr self "built"))
(defn call 
  "Calls the model on new inputs.

        In this case `call` just reapplies
        all ops in the graph to the new inputs
        (e.g. build a new computational graph from the provided inputs).

        A model is callable on non-Keras tensors.

        # Arguments
            inputs: A tensor or list of tensors.
            mask: A mask or list of masks. A mask can be
                either a tensor or None (no mask).

        # Returns
            A tensor if there is a single output, or
            a list of tensors if there are more than one outputs.
        "
  [self inputs  & {:keys [mask]} ]
    (py/call-attr-kw self "call" [inputs] {:mask mask }))

(defn compute-mask 
  ""
  [ self inputs mask ]
  (py/call-attr self "compute_mask"  self inputs mask ))

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
(defn get-layer 
  "Retrieves a layer based on either its name (unique) or index.

        If `name` and `index` are both provided, `index` will take precedence.

        Indices are based on order of horizontal graph traversal (bottom-up).

        # Arguments
            name: String, name of layer.
            index: Integer, index of layer.

        # Returns
            A layer instance.

        # Raises
            ValueError: In case of invalid layer name or index.
        "
  [self   & {:keys [name index]} ]
    (py/call-attr-kw self "get_layer" [] {:name name :index index }))

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
  "Retrieves the weights of the model.

        # Returns
            A flat list of Numpy arrays.
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

(defn input-spec 
  "Gets the model's input specs.

        # Returns
            A list of `InputSpec` instances (one per input to the model)
                or a single instance if the model has only one input.
        "
  [ self ]
    (py/call-attr self "input_spec"))

(defn layers 
  ""
  [ self ]
    (py/call-attr self "layers"))

(defn load-weights 
  "Loads all layer weights from a HDF5 save file.

        If `by_name` is False (default) weights are loaded
        based on the network's topology, meaning the architecture
        should be the same as when the weights were saved.
        Note that layers that don't have weights are not taken
        into account in the topological ordering, so adding or
        removing layers is fine as long as they don't have weights.

        If `by_name` is True, weights are loaded into layers
        only if they share the same name. This is useful
        for fine-tuning or transfer-learning models where
        some of the layers have changed.

        # Arguments
            filepath: String, path to the weights file to load.
            by_name: Boolean, whether to load weights by name
                or by topological order.
            skip_mismatch: Boolean, whether to skip loading of layers
                where there is a mismatch in the number of weights,
                or a mismatch in the shape of the weight
                (only valid when `by_name`=True).
            reshape: Reshape weights to fit the layer when the correct number
                of weight arrays is present but their shape does not match.


        # Raises
            ImportError: If h5py is not available.
        "
  [self filepath & {:keys [by_name skip_mismatch reshape]
                       :or {by_name false skip_mismatch false reshape false}} ]
    (py/call-attr-kw self "load_weights" [filepath] {:by_name by_name :skip_mismatch skip_mismatch :reshape reshape }))

(defn losses 
  "Retrieves the model's losses.

        Will only include losses that are either
        unconditional, or conditional on inputs to this model
        (e.g. will not include losses that depend on tensors
        that aren't inputs to this model).

        # Returns
            A list of loss tensors.
        "
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

(defn reset-states 
  ""
  [ self  ]
  (py/call-attr self "reset_states"  self  ))
(defn run-internal-graph 
  "Computes output tensors for new inputs.

        # Note:
            - Expects `inputs` to be a list (potentially with 1 element).
            - Can be run on non-Keras tensors.

        # Arguments
            inputs: List of tensors
            masks: List of masks (tensors or None).

        # Returns
            Three lists: output_tensors, output_masks, output_shapes
        "
  [self inputs  & {:keys [masks]} ]
    (py/call-attr-kw self "run_internal_graph" [inputs] {:masks masks }))

(defn save 
  "Saves the model to a single HDF5 file.

        The savefile includes:
            - The model architecture, allowing to re-instantiate the model.
            - The model weights.
            - The state of the optimizer, allowing to resume training
                exactly where you left off.

        This allows you to save the entirety of the state of a model
        in a single file.

        Saved models can be reinstantiated via `keras.models.load_model`.
        The model returned by `load_model`
        is a compiled model ready to be used (unless the saved model
        was never compiled in the first place).

        # Arguments
            filepath: one of the following:
                - string, path to the file to save the model to
                - h5py.File or h5py.Group object where to save the model
                - any file-like object implementing the method `write` that accepts
                    `bytes` data (e.g. `io.BytesIO`).
            overwrite: Whether to silently overwrite any existing file at the
                target location, or provide the user with a manual prompt.
            include_optimizer: If True, save optimizer's state together.

        # Example

        ```python
        from keras.models import load_model

        model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
        del model  # deletes the existing model

        # returns a compiled model
        # identical to the previous one
        model = load_model('my_model.h5')
        ```
        "
  [self filepath & {:keys [overwrite include_optimizer]
                       :or {overwrite true include_optimizer true}} ]
    (py/call-attr-kw self "save" [filepath] {:overwrite overwrite :include_optimizer include_optimizer }))

(defn save-weights 
  "Dumps all layer weights to a HDF5 file.

        The weight file has:
            - `layer_names` (attribute), a list of strings
                (ordered names of model layers).
            - For every layer, a `group` named `layer.name`
                - For every such layer group, a group attribute `weight_names`,
                    a list of strings
                    (ordered names of weights tensor of the layer).
                - For every weight in the layer, a dataset
                    storing the weight value, named after the weight tensor.

        # Arguments
            filepath: String, path to the file to save the weights to.
            overwrite: Whether to silently overwrite any existing file at the
                target location, or provide the user with a manual prompt.

        # Raises
            ImportError: If h5py is not available.
        "
  [self filepath & {:keys [overwrite]
                       :or {overwrite true}} ]
    (py/call-attr-kw self "save_weights" [filepath] {:overwrite overwrite }))

(defn set-weights 
  "Sets the weights of the model.

        # Arguments
            weights: A list of Numpy arrays with shapes and types matching
                the output of `model.get_weights()`.
        "
  [ self weights ]
  (py/call-attr self "set_weights"  self weights ))

(defn state-updates 
  "Returns the `updates` from all layers that are stateful.

        This is useful for separating training updates and
        state updates, e.g. when we need to update a layer's internal state
        during prediction.

        # Returns
            A list of update ops.
        "
  [ self ]
    (py/call-attr self "state_updates"))

(defn stateful 
  ""
  [ self ]
    (py/call-attr self "stateful"))
(defn summary 
  "Prints a string summary of the network.

        # Arguments
            line_length: Total length of printed lines
                (e.g. set this to adapt the display to different
                terminal window sizes).
            positions: Relative or absolute positions of log elements
                in each line. If not provided,
                defaults to `[.33, .55, .67, 1.]`.
            print_fn: Print function to use.
                It will be called on each line of the summary.
                You can set it to a custom function
                in order to capture the string summary.
                It defaults to `print` (prints to stdout).
        "
  [self   & {:keys [line_length positions print_fn]} ]
    (py/call-attr-kw self "summary" [] {:line_length line_length :positions positions :print_fn print_fn }))

(defn to-json 
  "Returns a JSON string containing the network configuration.

        To load a network from a JSON save file, use
        `keras.models.model_from_json(json_string, custom_objects={})`.

        # Arguments
            **kwargs: Additional keyword arguments
                to be passed to `json.dumps()`.

        # Returns
            A JSON string.
        "
  [ self  ]
  (py/call-attr self "to_json"  self  ))

(defn to-yaml 
  "Returns a yaml string containing the network configuration.

        To load a network from a yaml save file, use
        `keras.models.model_from_yaml(yaml_string, custom_objects={})`.

        `custom_objects` should be a dictionary mapping
        the names of custom losses / layers / etc to the corresponding
        functions / classes.

        # Arguments
            **kwargs: Additional keyword arguments
                to be passed to `yaml.dump()`.

        # Returns
            A YAML string.
        "
  [ self  ]
  (py/call-attr self "to_yaml"  self  ))

(defn trainable-weights 
  ""
  [ self ]
    (py/call-attr self "trainable_weights"))

(defn updates 
  "Retrieves the model's updates.

        Will only include updates that are either
        unconditional, or conditional on inputs to this model
        (e.g. will not include updates that depend on tensors
        that aren't inputs to this model).

        # Returns
            A list of update ops.
        "
  [ self ]
    (py/call-attr self "updates"))

(defn uses-learning-phase 
  ""
  [ self ]
    (py/call-attr self "uses_learning_phase"))

(defn weights 
  ""
  [ self ]
    (py/call-attr self "weights"))
