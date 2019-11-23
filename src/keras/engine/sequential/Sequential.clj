(ns keras.engine.sequential.Sequential
  "Linear stack of layers.

    # Arguments
        layers: list of layers to add to the model.

    # Example

    ```python
    # Optionally, the first layer can receive an `input_shape` argument:
    model = Sequential()
    model.add(Dense(32, input_shape=(500,)))

    # Afterwards, we do automatic shape inference:
    model.add(Dense(32))

    # This is identical to the following:
    model = Sequential()
    model.add(Dense(32, input_dim=500))

    # And to the following:
    model = Sequential()
    model.add(Dense(32, batch_input_shape=(None, 500)))

    # Note that you can also omit the `input_shape` argument:
    # In that case the model gets built the first time you call `fit` (or other
    # training and evaluation methods).
    model = Sequential()
    model.add(Dense(32))
    model.add(Dense(32))
    model.compile(optimizer=optimizer, loss=loss)

    # This builds the model for the first time:
    model.fit(x, y, batch_size=32, epochs=10)

    # Note that when using this delayed-build pattern
    # (no input shape specified),
    # the model doesn't have any weights until the first call
    # to a training/evaluation method (since it isn't yet built):
    model = Sequential()
    model.add(Dense(32))
    model.add(Dense(32))
    model.weights  # returns []

    # Whereas if you specify the input shape, the model gets built continuously
    # as you are adding layers:
    model = Sequential()
    model.add(Dense(32, input_shape=(500,)))
    model.add(Dense(32))
    model.weights  # returns list of length 4

    # When using the delayed-build pattern (no input shape specified), you can
    # choose to manually build your model by calling
    # `build(batch_input_shape)`:
    model = Sequential()
    model.add(Dense(32))
    model.add(Dense(32))
    model.build((None, 500))
    model.weights  # returns list of length 4
    ```
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce sequential (import-module "keras.engine.sequential"))

(defn Sequential 
  "Linear stack of layers.

    # Arguments
        layers: list of layers to add to the model.

    # Example

    ```python
    # Optionally, the first layer can receive an `input_shape` argument:
    model = Sequential()
    model.add(Dense(32, input_shape=(500,)))

    # Afterwards, we do automatic shape inference:
    model.add(Dense(32))

    # This is identical to the following:
    model = Sequential()
    model.add(Dense(32, input_dim=500))

    # And to the following:
    model = Sequential()
    model.add(Dense(32, batch_input_shape=(None, 500)))

    # Note that you can also omit the `input_shape` argument:
    # In that case the model gets built the first time you call `fit` (or other
    # training and evaluation methods).
    model = Sequential()
    model.add(Dense(32))
    model.add(Dense(32))
    model.compile(optimizer=optimizer, loss=loss)

    # This builds the model for the first time:
    model.fit(x, y, batch_size=32, epochs=10)

    # Note that when using this delayed-build pattern
    # (no input shape specified),
    # the model doesn't have any weights until the first call
    # to a training/evaluation method (since it isn't yet built):
    model = Sequential()
    model.add(Dense(32))
    model.add(Dense(32))
    model.weights  # returns []

    # Whereas if you specify the input shape, the model gets built continuously
    # as you are adding layers:
    model = Sequential()
    model.add(Dense(32, input_shape=(500,)))
    model.add(Dense(32))
    model.weights  # returns list of length 4

    # When using the delayed-build pattern (no input shape specified), you can
    # choose to manually build your model by calling
    # `build(batch_input_shape)`:
    model = Sequential()
    model.add(Dense(32))
    model.add(Dense(32))
    model.build((None, 500))
    model.weights  # returns list of length 4
    ```
    "
  [ & {:keys [layers name]} ]
   (py/call-attr-kw sequential "Sequential" [] {:layers layers :name name }))

(defn add 
  "Adds a layer instance on top of the layer stack.

        # Arguments
            layer: layer instance.

        # Raises
            TypeError: If `layer` is not a layer instance.
            ValueError: In case the `layer` argument does not
                know its input shape.
            ValueError: In case the `layer` argument has
                multiple output tensors, or is already connected
                somewhere else (forbidden in `Sequential` models).
        "
  [self  & {:keys [layer]} ]
    (py/call-attr-kw sequential "add" [self] {:layer layer }))

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
    (py/call-attr-kw sequential "add_loss" [self] {:losses losses :inputs inputs }))

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
    (py/call-attr-kw sequential "add_update" [self] {:updates updates :inputs inputs }))

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
    (py/call-attr-kw sequential "add_weight" [] {:name name :shape shape :dtype dtype :initializer initializer :regularizer regularizer :trainable trainable :constraint constraint }))

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
    (py/call-attr-kw sequential "assert_input_compatibility" [self] {:inputs inputs }))

(defn build 
  ""
  [self  & {:keys [input_shape]} ]
    (py/call-attr-kw sequential "build" [self] {:input_shape input_shape }))

(defn built 
  ""
  [ self ]
    (py/call-attr sequential "built"  self))

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
  [self  & {:keys [inputs mask]} ]
    (py/call-attr-kw sequential "call" [self] {:inputs inputs :mask mask }))

(defn compile 
  "Configures the model for training.

        # Arguments
            optimizer: String (name of optimizer) or optimizer instance.
                See [optimizers](/optimizers).
            loss: String (name of objective function) or objective function.
                See [losses](/losses).
                If the model has multiple outputs, you can use a different loss
                on each output by passing a dictionary or a list of losses.
                The loss value that will be minimized by the model
                will then be the sum of all individual losses.
            metrics: List of metrics to be evaluated by the model
                during training and testing.
                Typically you will use `metrics=['accuracy']`.
                To specify different metrics for different outputs of a
                multi-output model, you could also pass a dictionary,
                such as `metrics={'output_a': 'accuracy'}`.
            loss_weights: Optional list or dictionary specifying scalar
                coefficients (Python floats) to weight the loss contributions
                of different model outputs.
                The loss value that will be minimized by the model
                will then be the *weighted sum* of all individual losses,
                weighted by the `loss_weights` coefficients.
                If a list, it is expected to have a 1:1 mapping
                to the model's outputs. If a tensor, it is expected to map
                output names (strings) to scalar coefficients.
            sample_weight_mode: If you need to do timestep-wise
                sample weighting (2D weights), set this to `\"temporal\"`.
                `None` defaults to sample-wise weights (1D).
                If the model has multiple outputs, you can use a different
                `sample_weight_mode` on each output by passing a
                dictionary or a list of modes.
            weighted_metrics: List of metrics to be evaluated and weighted
                by sample_weight or class_weight during training and testing.
            target_tensors: By default, Keras will create placeholders for the
                model's target, which will be fed with the target data during
                training. If instead you would like to use your own
                target tensors (in turn, Keras will not expect external
                Numpy data for these targets at training time), you
                can specify them via the `target_tensors` argument. It can be
                a single tensor (for a single-output model), a list of tensors,
                or a dict mapping output names to target tensors.
            **kwargs: When using the Theano/CNTK backends, these arguments
                are passed into `K.function`.
                When using the TensorFlow backend,
                these arguments are passed into `tf.Session.run`.

        # Raises
            ValueError: In case of invalid arguments for
                `optimizer`, `loss`, `metrics` or `sample_weight_mode`.
        "
  [self  & {:keys [optimizer loss metrics loss_weights sample_weight_mode weighted_metrics target_tensors]} ]
    (py/call-attr-kw sequential "compile" [self] {:optimizer optimizer :loss loss :metrics metrics :loss_weights loss_weights :sample_weight_mode sample_weight_mode :weighted_metrics weighted_metrics :target_tensors target_tensors }))

(defn compute-mask 
  ""
  [self  & {:keys [inputs mask]} ]
    (py/call-attr-kw sequential "compute_mask" [self] {:inputs inputs :mask mask }))

(defn compute-output-shape 
  ""
  [self  & {:keys [input_shape]} ]
    (py/call-attr-kw sequential "compute_output_shape" [self] {:input_shape input_shape }))

(defn count-params 
  "Counts the total number of scalars composing the weights.

        # Returns
            An integer count.

        # Raises
            RuntimeError: if the layer isn't yet built
                (in which case its weights aren't yet defined).
        "
  [ self ]
  (py/call-attr sequential "count_params"  self ))

(defn evaluate 
  "Returns the loss value & metrics values for the model in test mode.

        Computation is done in batches.

        # Arguments
            x: Numpy array of test data (if the model has a single input),
                or list of Numpy arrays (if the model has multiple inputs).
                If input layers in the model are named, you can also pass a
                dictionary mapping input names to Numpy arrays.
                `x` can be `None` (default) if feeding from
                framework-native tensors (e.g. TensorFlow data tensors).
            y: Numpy array of target (label) data
                (if the model has a single output),
                or list of Numpy arrays (if the model has multiple outputs).
                If output layers in the model are named, you can also pass a
                dictionary mapping output names to Numpy arrays.
                `y` can be `None` (default) if feeding from
                framework-native tensors (e.g. TensorFlow data tensors).
            batch_size: Integer or `None`.
                Number of samples per evaluation step.
                If unspecified, `batch_size` will default to 32.
            verbose: 0 or 1. Verbosity mode.
                0 = silent, 1 = progress bar.
            sample_weight: Optional Numpy array of weights for
                the test samples, used for weighting the loss function.
                You can either pass a flat (1D)
                Numpy array with the same length as the input samples
                (1:1 mapping between weights and samples),
                or in the case of temporal data,
                you can pass a 2D array with shape
                `(samples, sequence_length)`,
                to apply a different weight to every timestep of every sample.
                In this case you should make sure to specify
                `sample_weight_mode=\"temporal\"` in `compile()`.
            steps: Integer or `None`.
                Total number of steps (batches of samples)
                before declaring the evaluation round finished.
                Ignored with the default value of `None`.

        # Returns
            Scalar test loss (if the model has a single output and no metrics)
            or list of scalars (if the model has multiple outputs
            and/or metrics). The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.
        "
  [self & {:keys [x y batch_size verbose sample_weight steps]
                       :or {verbose 1}} ]
    (py/call-attr-kw sequential "evaluate" [] {:x x :y y :batch_size batch_size :verbose verbose :sample_weight sample_weight :steps steps }))

(defn evaluate-generator 
  "Evaluates the model on a data generator.

        The generator should return the same kind of data
        as accepted by `test_on_batch`.

        # Arguments
            generator: Generator yielding tuples (inputs, targets)
                or (inputs, targets, sample_weights)
                or an instance of Sequence (keras.utils.Sequence)
                object in order to avoid duplicate data
                when using multiprocessing.
            steps: Total number of steps (batches of samples)
                to yield from `generator` before stopping.
                Optional for `Sequence`: if unspecified, will use
                the `len(generator)` as a number of steps.
            max_queue_size: maximum size for the generator queue
            workers: Integer. Maximum number of processes to spin up
                when using process based threading.
                If unspecified, `workers` will default to 1. If 0, will
                execute the generator on the main thread.
            use_multiprocessing: if True, use process based threading.
                Note that because
                this implementation relies on multiprocessing,
                you should not pass
                non picklable arguments to the generator
                as they can't be passed
                easily to children processes.
            verbose: verbosity mode, 0 or 1.

        # Returns
            Scalar test loss (if the model has a single output and no metrics)
            or list of scalars (if the model has multiple outputs
            and/or metrics). The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.

        # Raises
            ValueError: In case the generator yields
                data in an invalid format.
        "
  [self & {:keys [generator steps max_queue_size workers use_multiprocessing verbose]
                       :or {max_queue_size 10 workers 1 use_multiprocessing false verbose 0}} ]
    (py/call-attr-kw sequential "evaluate_generator" [] {:generator generator :steps steps :max_queue_size max_queue_size :workers workers :use_multiprocessing use_multiprocessing :verbose verbose }))

(defn fit 
  "Trains the model for a given number of epochs (iterations on a dataset).

        # Arguments
            x: Numpy array of training data (if the model has a single input),
                or list of Numpy arrays (if the model has multiple inputs).
                If input layers in the model are named, you can also pass a
                dictionary mapping input names to Numpy arrays.
                `x` can be `None` (default) if feeding from
                framework-native tensors (e.g. TensorFlow data tensors).
            y: Numpy array of target (label) data
                (if the model has a single output),
                or list of Numpy arrays (if the model has multiple outputs).
                If output layers in the model are named, you can also pass a
                dictionary mapping output names to Numpy arrays.
                `y` can be `None` (default) if feeding from
                framework-native tensors (e.g. TensorFlow data tensors).
            batch_size: Integer or `None`.
                Number of samples per gradient update.
                If unspecified, `batch_size` will default to 32.
            epochs: Integer. Number of epochs to train the model.
                An epoch is an iteration over the entire `x` and `y`
                data provided.
                Note that in conjunction with `initial_epoch`,
                `epochs` is to be understood as \"final epoch\".
                The model is not trained for a number of iterations
                given by `epochs`, but merely until the epoch
                of index `epochs` is reached.
            verbose: Integer. 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during training.
                See [callbacks](/callbacks).
            validation_split: Float between 0 and 1.
                Fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate
                the loss and any model metrics
                on this data at the end of each epoch.
                The validation data is selected from the last samples
                in the `x` and `y` data provided, before shuffling.
            validation_data: tuple `(x_val, y_val)` or tuple
                `(x_val, y_val, val_sample_weights)` on which to evaluate
                the loss and any model metrics at the end of each epoch.
                The model will not be trained on this data.
                `validation_data` will override `validation_split`.
            shuffle: Boolean (whether to shuffle the training data
                before each epoch) or str (for 'batch').
                'batch' is a special option for dealing with the
                limitations of HDF5 data; it shuffles in batch-sized chunks.
                Has no effect when `steps_per_epoch` is not `None`.
            class_weight: Optional dictionary mapping class indices (integers)
                to a weight (float) value, used for weighting the loss function
                (during training only).
                This can be useful to tell the model to
                \"pay more attention\" to samples from
                an under-represented class.
            sample_weight: Optional Numpy array of weights for
                the training samples, used for weighting the loss function
                (during training only). You can either pass a flat (1D)
                Numpy array with the same length as the input samples
                (1:1 mapping between weights and samples),
                or in the case of temporal data,
                you can pass a 2D array with shape
                `(samples, sequence_length)`,
                to apply a different weight to every timestep of every sample.
                In this case you should make sure to specify
                `sample_weight_mode=\"temporal\"` in `compile()`.
            initial_epoch: Integer.
                Epoch at which to start training
                (useful for resuming a previous training run).
            steps_per_epoch: Integer or `None`.
                Total number of steps (batches of samples)
                before declaring one epoch finished and starting the
                next epoch. When training with input tensors such as
                TensorFlow data tensors, the default `None` is equal to
                the number of samples in your dataset divided by
                the batch size, or 1 if that cannot be determined.
            validation_steps: Only relevant if `steps_per_epoch`
                is specified. Total number of steps (batches of samples)
                to validate before stopping.

        # Returns
            A `History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).

        # Raises
            RuntimeError: If the model was never compiled.
            ValueError: In case of mismatch between the provided input data
                and what the model expects.
        "
  [self & {:keys [x y batch_size epochs verbose callbacks validation_split validation_data shuffle class_weight sample_weight initial_epoch steps_per_epoch validation_steps]
                       :or {epochs 1 verbose 1 validation_split 0.0 shuffle true initial_epoch 0}} ]
    (py/call-attr-kw sequential "fit" [] {:x x :y y :batch_size batch_size :epochs epochs :verbose verbose :callbacks callbacks :validation_split validation_split :validation_data validation_data :shuffle shuffle :class_weight class_weight :sample_weight sample_weight :initial_epoch initial_epoch :steps_per_epoch steps_per_epoch :validation_steps validation_steps }))

(defn fit-generator 
  "Trains the model on data generated batch-by-batch by a Python generator
        (or an instance of `Sequence`).

        The generator is run in parallel to the model, for efficiency.
        For instance, this allows you to do real-time data augmentation
        on images on CPU in parallel to training your model on GPU.

        The use of `keras.utils.Sequence` guarantees the ordering
        and guarantees the single use of every input per epoch when
        using `use_multiprocessing=True`.

        # Arguments
            generator: A generator or an instance of `Sequence`
                (`keras.utils.Sequence`) object in order to avoid
                duplicate data when using multiprocessing.
                The output of the generator must be either
                - a tuple `(inputs, targets)`
                - a tuple `(inputs, targets, sample_weights)`.
                This tuple (a single output of the generator) makes a single
                batch. Therefore, all arrays in this tuple must have the same
                length (equal to the size of this batch). Different batches may
                have different sizes. For example, the last batch of the epoch
                is commonly smaller than the others, if the size of the dataset
                is not divisible by the batch size.
                The generator is expected to loop over its data
                indefinitely. An epoch finishes when `steps_per_epoch`
                batches have been seen by the model.
            steps_per_epoch: Integer.
                Total number of steps (batches of samples)
                to yield from `generator` before declaring one epoch
                finished and starting the next epoch. It should typically
                be equal to the number of samples of your dataset
                divided by the batch size.
                Optional for `Sequence`: if unspecified, will use
                the `len(generator)` as a number of steps.
            epochs: Integer. Number of epochs to train the model.
                An epoch is an iteration over the entire data provided,
                as defined by `steps_per_epoch`.
                Note that in conjunction with `initial_epoch`,
                `epochs` is to be understood as \"final epoch\".
                The model is not trained for a number of iterations
                given by `epochs`, but merely until the epoch
                of index `epochs` is reached.
            verbose: Integer. 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during training.
                See [callbacks](/callbacks).
            validation_data: This can be either
                - a generator or a `Sequence` object for the validation data
                - tuple `(x_val, y_val)`
                - tuple `(x_val, y_val, val_sample_weights)`
                on which to evaluate
                the loss and any model metrics at the end of each epoch.
                The model will not be trained on this data.
            validation_steps: Only relevant if `validation_data`
                is a generator. Total number of steps (batches of samples)
                to yield from `validation_data` generator before stopping
                at the end of every epoch. It should typically
                be equal to the number of samples of your
                validation dataset divided by the batch size.
                Optional for `Sequence`: if unspecified, will use
                the `len(validation_data)` as a number of steps.
            class_weight: Optional dictionary mapping class indices (integers)
                to a weight (float) value, used for weighting the loss function
                (during training only). This can be useful to tell the model to
                \"pay more attention\" to samples
                from an under-represented class.
            max_queue_size: Integer. Maximum size for the generator queue.
                If unspecified, `max_queue_size` will default to 10.
            workers: Integer. Maximum number of processes to spin up
                when using process-based threading.
                If unspecified, `workers` will default to 1. If 0, will
                execute the generator on the main thread.
            use_multiprocessing: Boolean.
                If `True`, use process-based threading.
                If unspecified, `use_multiprocessing` will default to `False`.
                Note that because this implementation
                relies on multiprocessing,
                you should not pass non-picklable arguments to the generator
                as they can't be passed easily to children processes.
            shuffle: Boolean. Whether to shuffle the order of the batches at
                the beginning of each epoch. Only used with instances
                of `Sequence` (`keras.utils.Sequence`).
                Has no effect when `steps_per_epoch` is not `None`.
            initial_epoch: Integer.
                Epoch at which to start training
                (useful for resuming a previous training run).

        # Returns
            A `History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).

        # Raises
            ValueError: In case the generator yields data in an invalid format.

        # Example

        ```python
        def generate_arrays_from_file(path):
            while True:
                with open(path) as f:
                    for line in f:
                        # create numpy arrays of input data
                        # and labels, from each line in the file
                        x1, x2, y = process_line(line)
                        yield ({'input_1': x1, 'input_2': x2}, {'output': y})

        model.fit_generator(generate_arrays_from_file('/my_file.txt'),
                            steps_per_epoch=10000, epochs=10)
        ```
        "
  [self & {:keys [generator steps_per_epoch epochs verbose callbacks validation_data validation_steps class_weight max_queue_size workers use_multiprocessing shuffle initial_epoch]
                       :or {epochs 1 verbose 1 max_queue_size 10 workers 1 use_multiprocessing false shuffle true initial_epoch 0}} ]
    (py/call-attr-kw sequential "fit_generator" [] {:generator generator :steps_per_epoch steps_per_epoch :epochs epochs :verbose verbose :callbacks callbacks :validation_data validation_data :validation_steps validation_steps :class_weight class_weight :max_queue_size max_queue_size :workers workers :use_multiprocessing use_multiprocessing :shuffle shuffle :initial_epoch initial_epoch }))

(defn get-config 
  ""
  [ self ]
  (py/call-attr sequential "get_config"  self ))

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
    (py/call-attr-kw sequential "get_input_at" [self] {:node_index node_index }))

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
    (py/call-attr-kw sequential "get_input_mask_at" [self] {:node_index node_index }))

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
    (py/call-attr-kw sequential "get_input_shape_at" [self] {:node_index node_index }))

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
  [self  & {:keys [name index]} ]
    (py/call-attr-kw sequential "get_layer" [self] {:name name :index index }))

(defn get-losses-for 
  ""
  [self  & {:keys [inputs]} ]
    (py/call-attr-kw sequential "get_losses_for" [self] {:inputs inputs }))

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
    (py/call-attr-kw sequential "get_output_at" [self] {:node_index node_index }))

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
    (py/call-attr-kw sequential "get_output_mask_at" [self] {:node_index node_index }))

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
    (py/call-attr-kw sequential "get_output_shape_at" [self] {:node_index node_index }))

(defn get-updates-for 
  ""
  [self  & {:keys [inputs]} ]
    (py/call-attr-kw sequential "get_updates_for" [self] {:inputs inputs }))

(defn get-weights 
  "Retrieves the weights of the model.

        # Returns
            A flat list of Numpy arrays.
        "
  [ self ]
  (py/call-attr sequential "get_weights"  self ))

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
    (py/call-attr sequential "input"  self))

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
    (py/call-attr sequential "input_mask"  self))

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
    (py/call-attr sequential "input_shape"  self))

(defn input-spec 
  "Gets the model's input specs.

        # Returns
            A list of `InputSpec` instances (one per input to the model)
                or a single instance if the model has only one input.
        "
  [ self ]
    (py/call-attr sequential "input_spec"  self))

(defn layers 
  ""
  [ self ]
    (py/call-attr sequential "layers"  self))

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
  [self & {:keys [filepath by_name skip_mismatch reshape]
                       :or {by_name false skip_mismatch false reshape false}} ]
    (py/call-attr-kw sequential "load_weights" [] {:filepath filepath :by_name by_name :skip_mismatch skip_mismatch :reshape reshape }))

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
    (py/call-attr sequential "losses"  self))

(defn model 
  ""
  [ self ]
    (py/call-attr sequential "model"  self))

(defn non-trainable-weights 
  ""
  [ self ]
    (py/call-attr sequential "non_trainable_weights"  self))

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
    (py/call-attr sequential "output"  self))

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
    (py/call-attr sequential "output_mask"  self))

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
    (py/call-attr sequential "output_shape"  self))

(defn pop 
  "Removes the last layer in the model.

        # Raises
            TypeError: if there are no layers in the model.
        "
  [ self ]
  (py/call-attr sequential "pop"  self ))

(defn predict 
  "Generates output predictions for the input samples.

        Computation is done in batches.

        # Arguments
            x: The input data, as a Numpy array
                (or list of Numpy arrays if the model has multiple inputs).
            batch_size: Integer. If unspecified, it will default to 32.
            verbose: Verbosity mode, 0 or 1.
            steps: Total number of steps (batches of samples)
                before declaring the prediction round finished.
                Ignored with the default value of `None`.

        # Returns
            Numpy array(s) of predictions.

        # Raises
            ValueError: In case of mismatch between the provided
                input data and the model's expectations,
                or in case a stateful model receives a number of samples
                that is not a multiple of the batch size.
        "
  [self & {:keys [x batch_size verbose steps]
                       :or {verbose 0}} ]
    (py/call-attr-kw sequential "predict" [] {:x x :batch_size batch_size :verbose verbose :steps steps }))

(defn predict-classes 
  "Generate class predictions for the input samples.

        The input samples are processed batch by batch.

        # Arguments
            x: input data, as a Numpy array or list of Numpy arrays
                (if the model has multiple inputs).
            batch_size: integer.
            verbose: verbosity mode, 0 or 1.

        # Returns:
            A numpy array of class predictions.
        "
  [self & {:keys [x batch_size verbose]
                       :or {batch_size 32 verbose 0}} ]
    (py/call-attr-kw sequential "predict_classes" [] {:x x :batch_size batch_size :verbose verbose }))

(defn predict-generator 
  "Generates predictions for the input samples from a data generator.

        The generator should return the same kind of data as accepted by
        `predict_on_batch`.

        # Arguments
            generator: Generator yielding batches of input samples
                or an instance of Sequence (keras.utils.Sequence)
                object in order to avoid duplicate data
                when using multiprocessing.
            steps: Total number of steps (batches of samples)
                to yield from `generator` before stopping.
                Optional for `Sequence`: if unspecified, will use
                the `len(generator)` as a number of steps.
            max_queue_size: Maximum size for the generator queue.
            workers: Integer. Maximum number of processes to spin up
                when using process based threading.
                If unspecified, `workers` will default to 1. If 0, will
                execute the generator on the main thread.
            use_multiprocessing: If `True`, use process based threading.
                Note that because
                this implementation relies on multiprocessing,
                you should not pass
                non picklable arguments to the generator
                as they can't be passed
                easily to children processes.
            verbose: verbosity mode, 0 or 1.

        # Returns
            Numpy array(s) of predictions.

        # Raises
            ValueError: In case the generator yields
                data in an invalid format.
        "
  [self & {:keys [generator steps max_queue_size workers use_multiprocessing verbose]
                       :or {max_queue_size 10 workers 1 use_multiprocessing false verbose 0}} ]
    (py/call-attr-kw sequential "predict_generator" [] {:generator generator :steps steps :max_queue_size max_queue_size :workers workers :use_multiprocessing use_multiprocessing :verbose verbose }))

(defn predict-on-batch 
  "Returns predictions for a single batch of samples.

        # Arguments
            x: Input samples, as a Numpy array.

        # Returns
            Numpy array(s) of predictions.
        "
  [self  & {:keys [x]} ]
    (py/call-attr-kw sequential "predict_on_batch" [self] {:x x }))

(defn predict-proba 
  "Generates class probability predictions for the input samples.

        The input samples are processed batch by batch.

        # Arguments
            x: input data, as a Numpy array or list of Numpy arrays
                (if the model has multiple inputs).
            batch_size: integer.
            verbose: verbosity mode, 0 or 1.

        # Returns
            A Numpy array of probability predictions.
        "
  [self & {:keys [x batch_size verbose]
                       :or {batch_size 32 verbose 0}} ]
    (py/call-attr-kw sequential "predict_proba" [] {:x x :batch_size batch_size :verbose verbose }))

(defn reset-states 
  ""
  [ self ]
  (py/call-attr sequential "reset_states"  self ))

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
  [self  & {:keys [inputs masks]} ]
    (py/call-attr-kw sequential "run_internal_graph" [self] {:inputs inputs :masks masks }))

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
            filepath: String, path to the file to save the weights to.
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
  [self & {:keys [filepath overwrite include_optimizer]
                       :or {overwrite true include_optimizer true}} ]
    (py/call-attr-kw sequential "save" [] {:filepath filepath :overwrite overwrite :include_optimizer include_optimizer }))

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
  [self & {:keys [filepath overwrite]
                       :or {overwrite true}} ]
    (py/call-attr-kw sequential "save_weights" [] {:filepath filepath :overwrite overwrite }))

(defn set-weights 
  "Sets the weights of the model.

        # Arguments
            weights: A list of Numpy arrays with shapes and types matching
                the output of `model.get_weights()`.
        "
  [self  & {:keys [weights]} ]
    (py/call-attr-kw sequential "set_weights" [self] {:weights weights }))

(defn state-updates 
  "Returns the `updates` from all layers that are stateful.

        This is useful for separating training updates and
        state updates, e.g. when we need to update a layer's internal state
        during prediction.

        # Returns
            A list of update ops.
        "
  [ self ]
    (py/call-attr sequential "state_updates"  self))

(defn stateful 
  ""
  [ self ]
    (py/call-attr sequential "stateful"  self))

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
  [self  & {:keys [line_length positions print_fn]} ]
    (py/call-attr-kw sequential "summary" [self] {:line_length line_length :positions positions :print_fn print_fn }))

(defn test-on-batch 
  "Test the model on a single batch of samples.

        # Arguments
            x: Numpy array of test data,
                or list of Numpy arrays if the model has multiple inputs.
                If all inputs in the model are named,
                you can also pass a dictionary
                mapping input names to Numpy arrays.
            y: Numpy array of target data,
                or list of Numpy arrays if the model has multiple outputs.
                If all outputs in the model are named,
                you can also pass a dictionary
                mapping output names to Numpy arrays.
            sample_weight: Optional array of the same length as x, containing
                weights to apply to the model's loss for each sample.
                In the case of temporal data, you can pass a 2D array
                with shape (samples, sequence_length),
                to apply a different weight to every timestep of every sample.
                In this case you should make sure to specify
                sample_weight_mode=\"temporal\" in compile().

        # Returns
            Scalar test loss (if the model has a single output and no metrics)
            or list of scalars (if the model has multiple outputs
            and/or metrics). The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.
        "
  [self  & {:keys [x y sample_weight]} ]
    (py/call-attr-kw sequential "test_on_batch" [self] {:x x :y y :sample_weight sample_weight }))

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
  [ self ]
  (py/call-attr sequential "to_json"  self ))

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
  [ self ]
  (py/call-attr sequential "to_yaml"  self ))

(defn train-on-batch 
  "Runs a single gradient update on a single batch of data.

        # Arguments
            x: Numpy array of training data,
                or list of Numpy arrays if the model has multiple inputs.
                If all inputs in the model are named,
                you can also pass a dictionary
                mapping input names to Numpy arrays.
            y: Numpy array of target data,
                or list of Numpy arrays if the model has multiple outputs.
                If all outputs in the model are named,
                you can also pass a dictionary
                mapping output names to Numpy arrays.
            sample_weight: Optional array of the same length as x, containing
                weights to apply to the model's loss for each sample.
                In the case of temporal data, you can pass a 2D array
                with shape (samples, sequence_length),
                to apply a different weight to every timestep of every sample.
                In this case you should make sure to specify
                sample_weight_mode=\"temporal\" in compile().
            class_weight: Optional dictionary mapping
                class indices (integers) to
                a weight (float) to apply to the model's loss for the samples
                from this class during training.
                This can be useful to tell the model to \"pay more attention\" to
                samples from an under-represented class.

        # Returns
            Scalar training loss
            (if the model has a single output and no metrics)
            or list of scalars (if the model has multiple outputs
            and/or metrics). The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.
        "
  [self  & {:keys [x y sample_weight class_weight]} ]
    (py/call-attr-kw sequential "train_on_batch" [self] {:x x :y y :sample_weight sample_weight :class_weight class_weight }))

(defn trainable-weights 
  ""
  [ self ]
    (py/call-attr sequential "trainable_weights"  self))

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
    (py/call-attr sequential "updates"  self))

(defn uses-learning-phase 
  ""
  [ self ]
    (py/call-attr sequential "uses_learning_phase"  self))

(defn weights 
  ""
  [ self ]
    (py/call-attr sequential "weights"  self))
