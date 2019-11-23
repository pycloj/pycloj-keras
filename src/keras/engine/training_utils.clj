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
  [ & {:keys [index_array batch_size]} ]
   (py/call-attr-kw training-utils "batch_shuffle" [] {:index_array index_array :batch_size batch_size }))

(defn check-array-length-consistency 
  "Checks if batch axes are the same for numpy arrays.

    # Arguments
        inputs: list of Numpy arrays of inputs.
        targets: list of Numpy arrays of targets.
        weights: list of Numpy arrays of sample weights.

    # Raises
        ValueError: in case of incorrectly formatted data.
    "
  [ & {:keys [inputs targets weights]} ]
   (py/call-attr-kw training-utils "check_array_length_consistency" [] {:inputs inputs :targets targets :weights weights }))

(defn check-loss-and-target-compatibility 
  "Does validation on the compatibility of targets and loss functions.

    This helps prevent users from using loss functions incorrectly.

    # Arguments
        targets: list of Numpy arrays of targets.
        loss_fns: list of loss functions.
        output_shapes: list of shapes of model outputs.

    # Raises
        ValueError: if a loss function or target array
            is incompatible with an output.
    "
  [ & {:keys [targets loss_fns output_shapes]} ]
   (py/call-attr-kw training-utils "check_loss_and_target_compatibility" [] {:targets targets :loss_fns loss_fns :output_shapes output_shapes }))

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
        When steps is `None`, returns the number of samples to be
        processed based on the size of the first dimension of the
        first input numpy array. When steps is not `None` and
        `batch_size` is `None`, returns `None`.

    # Raises
        ValueError: In case of invalid arguments.
    "
  [ & {:keys [ins batch_size steps steps_name]
       :or {steps_name "steps"}} ]
  
   (py/call-attr-kw training-utils "check_num_samples" [] {:ins ins :batch_size batch_size :steps steps :steps_name steps_name }))

(defn collect-metrics 
  "Maps metric functions to model outputs.

    # Arguments
        metrics: a list or dict of metric functions.
        output_names: a list of the names (strings) of model outputs.

    # Returns
        A list (one entry per model output) of lists of metric functions.
        For instance, if the model has 2 outputs, and for the first output
        we want to compute \"binary_accuracy\" and \"binary_crossentropy\",
        and just \"binary_accuracy\" for the second output,
        the list would look like:
            `[[binary_accuracy, binary_crossentropy], [binary_accuracy]]`

    # Raises
        TypeError: if an incorrect type is passed for the `metrics` argument.
    "
  [ & {:keys [metrics output_names]} ]
   (py/call-attr-kw training-utils "collect_metrics" [] {:metrics metrics :output_names output_names }))

(defn iter-sequence-infinite 
  "Iterate indefinitely over a Sequence.

    # Arguments
        seq: Sequence object

    # Returns
        Generator yielding batches.
    "
  [ & {:keys [seq]} ]
   (py/call-attr-kw training-utils "iter_sequence_infinite" [] {:seq seq }))

(defn make-batches 
  "Returns a list of batch indices (tuples of indices).

    # Arguments
        size: Integer, total size of the data to slice into batches.
        batch_size: Integer, batch size.

    # Returns
        A list of tuples of array indices.
    "
  [ & {:keys [size batch_size]} ]
   (py/call-attr-kw training-utils "make_batches" [] {:size size :batch_size batch_size }))

(defn standardize-class-weights 
  ""
  [ & {:keys [class_weight output_names]} ]
   (py/call-attr-kw training-utils "standardize_class_weights" [] {:class_weight class_weight :output_names output_names }))

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
  [ & {:keys [data names shapes check_batch_axis exception_prefix]
       :or {check_batch_axis true exception_prefix ""}} ]
  
   (py/call-attr-kw training-utils "standardize_input_data" [] {:data data :names names :shapes shapes :check_batch_axis check_batch_axis :exception_prefix exception_prefix }))

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
  [ & {:keys [x_weight output_names weight_type]} ]
   (py/call-attr-kw training-utils "standardize_sample_or_class_weights" [] {:x_weight x_weight :output_names output_names :weight_type weight_type }))

(defn standardize-sample-weights 
  ""
  [ & {:keys [sample_weight output_names]} ]
   (py/call-attr-kw training-utils "standardize_sample_weights" [] {:sample_weight sample_weight :output_names output_names }))

(defn standardize-single-array 
  ""
  [ & {:keys [x]} ]
   (py/call-attr-kw training-utils "standardize_single_array" [] {:x x }))

(defn standardize-weights 
  "Performs sample weight validation and standardization.

    Everything gets normalized to a single sample-wise (or timestep-wise)
    weight array.

    # Arguments
        y: Numpy array of model targets to be weighted.
        sample_weight: User-provided `sample_weight` argument.
        class_weight: User-provided `class_weight` argument.
        sample_weight_mode: One of `None` or `\"temporal\"`.
            `\"temporal\"` indicated that we expect 2D weight data
            that will be applied to the last 2 dimensions of
            the targets (i.e. we are weighting timesteps, not samples).

    # Returns
        A numpy array of target weights, one entry per sample to weight.

    # Raises
        ValueError: In case of invalid user-provided arguments.
    "
  [ & {:keys [y sample_weight class_weight sample_weight_mode]} ]
   (py/call-attr-kw training-utils "standardize_weights" [] {:y y :sample_weight sample_weight :class_weight class_weight :sample_weight_mode sample_weight_mode }))

(defn to-list 
  "Normalizes a list/tensor into a list.

    If a tensor is passed, we return
    a list of size 1 containing the tensor.

    # Arguments
        x: target object to be normalized.
        allow_tuple: If False and x is a tuple,
            it will be converted into a list
            with a single element (the tuple).
            Else converts the tuple to a list.

    # Returns
        A list.
    "
  [ & {:keys [x allow_tuple]
       :or {allow_tuple false}} ]
  
   (py/call-attr-kw training-utils "to_list" [] {:x x :allow_tuple allow_tuple }))

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
  [ & {:keys [fn]} ]
   (py/call-attr-kw training-utils "weighted_masked_objective" [] {:fn fn }))
