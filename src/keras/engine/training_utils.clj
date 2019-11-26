(ns keras.engine.training-utils
  "Training-related utilities.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce training-utils (import-module "keras.engine.training_utils"))

(defn batch-shuffle 
  "Shuffles an array in a batch-wise fashion.

    Useful for shuffling HDF5 arrays
    (where one cannot access arbitrary indices).

    # Arguments
        index_array: array of indices to be shuffled.
        batch_size: integer.

    # Returns
        The `index_array` array, shuffled in a batch-wise fashion.
    "
  [ index_array batch_size ]
  (py/call-attr training-utils "batch_shuffle"  index_array batch_size ))
(defn call-metric-function 
  "Invokes metric function and returns the metric result tensor."
  [metric_fn y_true  & {:keys [y_pred weights mask]} ]
    (py/call-attr-kw training-utils "call_metric_function" [metric_fn y_true] {:y_pred y_pred :weights weights :mask mask }))
(defn check-array-length-consistency 
  "Checks if batch axes are the same for Numpy arrays.

    # Arguments
        inputs: list of Numpy arrays of inputs.
        targets: list of Numpy arrays of targets.
        weights: list of Numpy arrays of sample weights.

    # Raises
        ValueError: in case of incorrectly formatted data.
    "
  [inputs targets  & {:keys [weights]} ]
    (py/call-attr-kw training-utils "check_array_length_consistency" [inputs targets] {:weights weights }))

(defn check-generator-arguments 
  "Validates arguments passed when using a generator."
  [ & {:keys [y sample_weight validation_split]} ]
   (py/call-attr-kw training-utils "check_generator_arguments" [] {:y y :sample_weight sample_weight :validation_split validation_split }))

(defn check-loss-and-target-compatibility 
  "Does validation on the compatibility of targets and loss functions.

    This helps prevent users from using loss functions incorrectly. This check
    is purely for UX purposes.

    # Arguments
        targets: list of Numpy arrays of targets.
        loss_fns: list of loss functions.
        output_shapes: list of shapes of model outputs.

    # Raises
        ValueError: if a loss function or target array
            is incompatible with an output.
    "
  [ targets loss_fns output_shapes ]
  (py/call-attr training-utils "check_loss_and_target_compatibility"  targets loss_fns output_shapes ))

(defn check-num-samples 
  "Checks the number of samples provided for training and evaluation.

    The number of samples is not defined when running with `steps`,
    in which case the number of samples is set to `None`.

    # Arguments
        ins: List of tensors to be fed to the Keras function.
        batch_size: Integer batch size or `None` if not defined.
        steps: Total number of steps (batches of samples)
            before declaring `predict_loop` finished.
            Ignored with the default value of `None`.
        steps_name: The public API's parameter name for `steps`.

    # Raises
        ValueError: when `steps` is `None` and the attribute `ins.shape`
        does not exist. Also raises ValueError when `steps` is not `None`
        and `batch_size` is not `None` because they are mutually
        exclusive.

    # Returns
        When `steps` is `None`, returns the number of samples to be
        processed based on the size of the first dimension of the
        first input Numpy array. When `steps` is not `None` and
        `batch_size` is `None`, returns `None`.

    # Raises
        ValueError: In case of invalid arguments.
    "
  [ins & {:keys [batch_size steps steps_name]
                       :or {steps_name "steps"}} ]
    (py/call-attr-kw training-utils "check_num_samples" [ins] {:batch_size batch_size :steps steps :steps_name steps_name }))

(defn collect-per-output-metric-info 
  "Maps metric names and functions to model outputs.

    # Arguments
        metrics: a list or a list of lists or a dict of metric functions.
        output_names: a list of the names (strings) of model outputs.
        output_shapes: a list of the shapes (strings) of model outputs.
        loss_fns: a list of the loss functions corresponding to the model outputs.
        is_weighted: Boolean indicating whether the given metrics are weighted.

    # Returns
        A list (one entry per model output) of dicts.
        For instance, if the model has 2 outputs, and for the first output
        we want to compute \"binary_accuracy\" and \"binary_crossentropy\",
        and just \"binary_accuracy\" for the second output,
        the list would look like: `[{
            'acc': binary_accuracy(),
            'ce': binary_crossentropy(),
        }, {
            'acc': binary_accuracy(),
        }]`

    # Raises
        TypeError: if an incorrect type is passed for the `metrics` argument.
    "
  [metrics output_names output_shapes loss_fns & {:keys [is_weighted]
                       :or {is_weighted false}} ]
    (py/call-attr-kw training-utils "collect_per_output_metric_info" [metrics output_names output_shapes loss_fns] {:is_weighted is_weighted }))

(defn get-input-shape-and-dtype 
  "Retrieves input shape and input dtype of layer if applicable.

    # Arguments
        layer: Layer (or model) instance.

    # Returns
        Tuple (input_shape, input_dtype). Both could be None if the layer
        does not have a defined input shape.

    # Raises
      ValueError: in case an empty Sequential or Functional model is passed.
    "
  [ layer ]
  (py/call-attr training-utils "get_input_shape_and_dtype"  layer ))

(defn get-loss-function 
  "Returns the loss corresponding to the loss input in `compile` API."
  [ loss ]
  (py/call-attr training-utils "get_loss_function"  loss ))
(defn get-metric-function 
  "Returns the metric function corresponding to the given metric input.

    # Arguments
        metric: Metric function name or reference.
        output_shape: The shape of the output that this metric will be calculated
            for.
        loss_fn: The loss function used.

    # Returns
        The metric function.
    "
  [metric  & {:keys [output_shape loss_fn]} ]
    (py/call-attr-kw training-utils "get_metric_function" [metric] {:output_shape output_shape :loss_fn loss_fn }))

(defn get-metric-name 
  "Returns the name corresponding to the given metric input.

    # Arguments
        metric: Metric function name or reference.
        weighted: Boolean indicating if the given metric is weighted.

    # Returns
        The metric name.
    "
  [metric & {:keys [weighted]
                       :or {weighted false}} ]
    (py/call-attr-kw training-utils "get_metric_name" [metric] {:weighted weighted }))

(defn get-output-sample-weight-and-mode 
  "Returns the sample weight and weight mode for a single output."
  [ skip_target_weighing_indices sample_weight_mode output_name output_index ]
  (py/call-attr training-utils "get_output_sample_weight_and_mode"  skip_target_weighing_indices sample_weight_mode output_name output_index ))

(defn get-static-batch-size 
  "Gets the static batch size of a Layer.

    # Arguments
        layer: a `Layer` instance.

    # Returns
        The static batch size of a Layer.
    "
  [ layer ]
  (py/call-attr training-utils "get_static_batch_size"  layer ))

(defn is-generator-or-sequence 
  "Check if `x` is a Keras generator type."
  [ x ]
  (py/call-attr training-utils "is_generator_or_sequence"  x ))

(defn is-sequence 
  "Determine if an object follows the Sequence API.

    # Arguments
        seq: a possible Sequence object

    # Returns
        boolean, whether the object follows the Sequence API.
    "
  [ seq ]
  (py/call-attr training-utils "is_sequence"  seq ))

(defn iter-sequence-infinite 
  "Iterate indefinitely over a Sequence.

    # Arguments
        seq: Sequence object

    # Returns
        Generator yielding batches.
    "
  [ seq ]
  (py/call-attr training-utils "iter_sequence_infinite"  seq ))

(defn make-batches 
  "Returns a list of batch indices (tuples of indices).

    # Arguments
        size: Integer, total size of the data to slice into batches.
        batch_size: Integer, batch size.

    # Returns
        A list of tuples of array indices.
    "
  [ size batch_size ]
  (py/call-attr training-utils "make_batches"  size batch_size ))

(defn prepare-loss-functions 
  "Converts loss to a list of loss functions.

    # Arguments
        loss: String (name of objective function), objective function or
            `Loss` instance. If the model has multiple outputs, you can use
            a different loss on each output by passing a dictionary or a
            list of losses. The loss value that will be minimized by the model
            will then be the sum of all individual losses.
        output_names: List of model output names.

    # Returns
        A list of loss objective functions.

    # Raises:
        ValueError: If loss is a dict with keys not in model output names,
            or if loss is a list with len not equal to model outputs.
    "
  [ loss output_names ]
  (py/call-attr training-utils "prepare_loss_functions"  loss output_names ))
(defn prepare-loss-weights 
  "Converts loss weights to a list of loss weights.

    # Arguments
        output_names: List of model output names.
        loss_weights: Optional list or dictionary specifying scalar coefficients
            (Python floats) to weight the loss contributions of different model
            outputs. The loss value that will be minimized by the model will then be
            the *weighted sum* of all individual losses, weighted by the
            `loss_weights` coefficients. If a list, it is expected to have a 1:1
            mapping to the model's outputs. If a dict, it is expected to map
            output names (strings) to scalar coefficients.

    # Returns
        A list of loss weights of python floats.

    # Raises
        ValueError: If loss weight is a dict with key not in model output names,
            or if loss is a list with len not equal to model outputs.
    "
  [output_names  & {:keys [loss_weights]} ]
    (py/call-attr-kw training-utils "prepare_loss_weights" [output_names] {:loss_weights loss_weights }))

(defn prepare-sample-weights 
  "Prepares sample weights for the model.

    # Arguments
        output_names: List of model output names.
        sample_weight_mode: sample weight mode user input passed from compile API.
        skip_target_weighing_indices: Indices of output for which sample weights
            should be skipped.

    # Returns
        A pair of list of sample weights and sample weight modes
            (one for each output).

    # Raises
        ValueError: In case of invalid `sample_weight_mode` input.
    "
  [ output_names sample_weight_mode skip_target_weighing_indices ]
  (py/call-attr training-utils "prepare_sample_weights"  output_names sample_weight_mode skip_target_weighing_indices ))

(defn should-run-validation 
  "Checks if validation should be run this epoch.

    # Arguments
        validation_freq: Integer or list. If an integer, specifies how many training
          epochs to run before a new validation run is performed. If a list,
          specifies the epochs on which to run validation.
        epoch: Integer, the number of the training epoch just completed.

    # Returns
        Bool, True if validation should be run.

    # Raises
        ValueError: if `validation_freq` is an Integer and less than 1, or if
        it is neither an Integer nor a Sequence.
    "
  [ validation_freq epoch ]
  (py/call-attr training-utils "should_run_validation"  validation_freq epoch ))

(defn standardize-class-weights 
  ""
  [ class_weight output_names ]
  (py/call-attr training-utils "standardize_class_weights"  class_weight output_names ))

(defn standardize-input-data 
  "Normalizes inputs and targets provided by users.

    Users may pass data as a list of arrays, dictionary of arrays,
    or as a single array. We normalize this to an ordered list of
    arrays (same order as `names`), while checking that the provided
    arrays have shapes that match the network's expectations.

    # Arguments
        data: User-provided input data (polymorphic).
        names: List of expected array names.
        shapes: Optional list of expected array shapes.
        check_batch_axis: Boolean; whether to check that
            the batch axis of the arrays matches the expected
            value found in `shapes`.
        exception_prefix: String prefix used for exception formatting.

    # Returns
        List of standardized input arrays (one array per model input).

    # Raises
        ValueError: in case of improperly formatted user-provided data.
    "
  [data names & {:keys [shapes check_batch_axis exception_prefix]
                       :or {check_batch_axis true exception_prefix ""}} ]
    (py/call-attr-kw training-utils "standardize_input_data" [data names] {:shapes shapes :check_batch_axis check_batch_axis :exception_prefix exception_prefix }))

(defn standardize-sample-or-class-weights 
  "Maps `sample_weight` or `class_weight` to model outputs.

    # Arguments
        x_weight: User-provided `sample_weight` or `class_weight` argument.
        output_names: List of output names (strings) in the model.
        weight_type: A string used purely for exception printing.

    # Returns
        A list of `sample_weight` or `class_weight` where there are exactly
            one element per model output.

    # Raises
        ValueError: In case of invalid user-provided argument.
    "
  [ x_weight output_names weight_type ]
  (py/call-attr training-utils "standardize_sample_or_class_weights"  x_weight output_names weight_type ))

(defn standardize-sample-weights 
  ""
  [ sample_weight output_names ]
  (py/call-attr training-utils "standardize_sample_weights"  sample_weight output_names ))

(defn standardize-single-array 
  ""
  [ x ]
  (py/call-attr training-utils "standardize_single_array"  x ))
(defn standardize-weights 
  "Performs sample weight validation and standardization.

    Everything gets normalized to a single sample-wise (or timestep-wise)
    weight array. If both `sample_weights` and `class_weights` are provided,
    the weights are multiplied together.

    # Arguments
        y: Numpy array of model targets to be weighted.
        sample_weight: User-provided `sample_weight` argument.
        class_weight: User-provided `class_weight` argument.
        sample_weight_mode: One of `None` or `\"temporal\"`.
            `\"temporal\"` indicated that we expect 2D weight data
            that will be applied to the last 2 dimensions of
            the targets (i.e. we are weighting timesteps, not samples).

    # Returns
        A Numpy array of target weights, one entry per sample to weight.

    # Raises
        ValueError: In case of invalid user-provided arguments.
    "
  [y  & {:keys [sample_weight class_weight sample_weight_mode]} ]
    (py/call-attr-kw training-utils "standardize_weights" [y] {:sample_weight sample_weight :class_weight class_weight :sample_weight_mode sample_weight_mode }))

(defn weighted-masked-objective 
  "Adds support for masking and sample-weighting to an objective function.

    It transforms an objective function `fn(y_true, y_pred)`
    into a sample-weighted, cost-masked objective function
    `fn(y_true, y_pred, weights, mask)`.

    # Arguments
        fn: The objective function to wrap,
            with signature `fn(y_true, y_pred)`.

    # Returns
        A function with signature `fn(y_true, y_pred, weights, mask)`.
    "
  [ fn ]
  (py/call-attr training-utils "weighted_masked_objective"  fn ))
