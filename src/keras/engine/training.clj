(ns keras.engine.training
  "Training-related part of the Keras engine.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce training (import-module "keras.engine.training"))

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
   (py/call-attr-kw training "check_array_length_consistency" [] {:inputs inputs :targets targets :weights weights }))

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
   (py/call-attr-kw training "check_loss_and_target_compatibility" [] {:targets targets :loss_fns loss_fns :output_shapes output_shapes }))

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
   (py/call-attr-kw training "collect_metrics" [] {:metrics metrics :output_names output_names }))

(defn slice-arrays 
  "Slices an array or list of arrays.

    This takes an array-like, or a list of
    array-likes, and outputs:
        - arrays[start:stop] if `arrays` is an array-like
        - [x[start:stop] for x in arrays] if `arrays` is a list

    Can also work on list/array of indices: `_slice_arrays(x, indices)`

    # Arguments
        arrays: Single array or list of arrays.
        start: can be an integer index (start index)
            or a list/array of indices
        stop: integer (stop index); should be None if
            `start` was a list.

    # Returns
        A slice of the array(s).
    "
  [ & {:keys [arrays start stop]} ]
   (py/call-attr-kw training "slice_arrays" [] {:arrays arrays :start start :stop stop }))

(defn standardize-class-weights 
  ""
  [ & {:keys [class_weight output_names]} ]
   (py/call-attr-kw training "standardize_class_weights" [] {:class_weight class_weight :output_names output_names }))

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
  
   (py/call-attr-kw training "standardize_input_data" [] {:data data :names names :shapes shapes :check_batch_axis check_batch_axis :exception_prefix exception_prefix }))

(defn standardize-sample-weights 
  ""
  [ & {:keys [sample_weight output_names]} ]
   (py/call-attr-kw training "standardize_sample_weights" [] {:sample_weight sample_weight :output_names output_names }))

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
   (py/call-attr-kw training "standardize_weights" [] {:y y :sample_weight sample_weight :class_weight class_weight :sample_weight_mode sample_weight_mode }))

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
  
   (py/call-attr-kw training "to_list" [] {:x x :allow_tuple allow_tuple }))

(defn unpack-singleton 
  "Gets the first element if the iterable has only one value.

    Otherwise return the iterable.

    # Argument:
        x: A list or tuple.

    # Returns:
        The same iterable or the first element.
    "
  [ & {:keys [x]} ]
   (py/call-attr-kw training "unpack_singleton" [] {:x x }))

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
   (py/call-attr-kw training "weighted_masked_objective" [] {:fn fn }))
