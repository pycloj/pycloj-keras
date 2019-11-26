(ns keras.engine.training-arrays
  "Part of the training engine related to plain array data (e.g. Numpy).
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce training-arrays (import-module "keras.engine.training_arrays"))

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
  (py/call-attr training-arrays "batch_shuffle"  index_array batch_size ))

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
    (py/call-attr-kw training-arrays "check_num_samples" [ins] {:batch_size batch_size :steps steps :steps_name steps_name }))

(defn fit-loop 
  "Abstract fit function for `fit_function(fit_inputs)`.

    Assumes that fit_function returns a list, labeled by out_labels.

    # Arguments
        model: Keras model instance.
        fit_function: Keras function returning a list of tensors
        fit_inputs: List of tensors to be fed to `fit_function`
        out_labels: List of strings, display names of
            the outputs of `fit_function`
        batch_size: Integer batch size or None if unknown.
        epochs: Number of times to iterate over the data
        verbose: Verbosity mode, 0, 1 or 2
        callbacks: List of callbacks to be called during training and validation
            (if `val_function` and `val_inputs` are not `None`).
        val_function: Keras function to call for validation
        val_inputs: List of tensors to be fed to `val_function`
        shuffle: Whether to shuffle the data at the beginning of each epoch
        initial_epoch: Epoch at which to start training
            (useful for resuming a previous training run)
        steps_per_epoch: Total number of steps (batches of samples)
            before declaring one epoch finished and starting the
            next epoch. Ignored with the default value of `None`.
        validation_steps: Number of steps to run validation for
            (only if doing validation from data tensors).
            Ignored with the default value of `None`.
        validation_freq: Only relevant if validation data is provided. Integer
            or list/tuple/set. If an integer, specifies how many training
            epochs to run before a new validation run is performed, e.g.
            validation_freq=2` runs validation every 2 epochs. If a list,
            tuple, or set, specifies the epochs on which to run validation,
            e.g. `validation_freq=[1, 2, 10]` runs validation at the end
            of the 1st, 2nd, and 10th epochs.

    # Returns
        `History` object.
    "
  [model fit_function fit_inputs & {:keys [out_labels batch_size epochs verbose callbacks val_function val_inputs shuffle initial_epoch steps_per_epoch validation_steps validation_freq]
                       :or {epochs 100 verbose 1 shuffle true initial_epoch 0 validation_freq 1}} ]
    (py/call-attr-kw training-arrays "fit_loop" [model fit_function fit_inputs] {:out_labels out_labels :batch_size batch_size :epochs epochs :verbose verbose :callbacks callbacks :val_function val_function :val_inputs val_inputs :shuffle shuffle :initial_epoch initial_epoch :steps_per_epoch steps_per_epoch :validation_steps validation_steps :validation_freq validation_freq }))

(defn issparse 
  "Is x of a sparse matrix type?

    Parameters
    ----------
    x
        object to check for being a sparse matrix

    Returns
    -------
    bool
        True if x is a sparse matrix, False otherwise

    Notes
    -----
    issparse and isspmatrix are aliases for the same function.

    Examples
    --------
    >>> from scipy.sparse import csr_matrix, isspmatrix
    >>> isspmatrix(csr_matrix([[5]]))
    True

    >>> from scipy.sparse import isspmatrix
    >>> isspmatrix(5)
    False
    "
  [ x ]
  (py/call-attr training-arrays "issparse"  x ))

(defn make-batches 
  "Returns a list of batch indices (tuples of indices).

    # Arguments
        size: Integer, total size of the data to slice into batches.
        batch_size: Integer, batch size.

    # Returns
        A list of tuples of array indices.
    "
  [ size batch_size ]
  (py/call-attr training-arrays "make_batches"  size batch_size ))

(defn predict-loop 
  "Abstract method to loop over some data in batches.

    # Arguments
        model: Keras model instance.
        f: Keras function returning a list of tensors.
        ins: list of tensors to be fed to `f`.
        batch_size: integer batch size.
        verbose: verbosity mode.
        steps: Total number of steps (batches of samples)
            before declaring `predict_loop` finished.
            Ignored with the default value of `None`.
        callbacks: List of callbacks or an instance of
            `keras.callbacks.CallbackList` to be called during prediction.

    # Returns
        Array of predictions (if the model has a single output)
        or list of arrays of predictions
        (if the model has multiple outputs).
    "
  [model f ins & {:keys [batch_size verbose steps callbacks]
                       :or {batch_size 32 verbose 0}} ]
    (py/call-attr-kw training-arrays "predict_loop" [model f ins] {:batch_size batch_size :verbose verbose :steps steps :callbacks callbacks }))

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
  (py/call-attr training-arrays "should_run_validation"  validation_freq epoch ))
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
  [arrays  & {:keys [start stop]} ]
    (py/call-attr-kw training-arrays "slice_arrays" [arrays] {:start start :stop stop }))

(defn test-loop 
  "Abstract method to loop over some data in batches.

    # Arguments
        model: Keras model instance.
        f: Keras function returning a list of tensors.
        ins: list of tensors to be fed to `f`.
        batch_size: integer batch size or `None`.
        verbose: verbosity mode.
        steps: Total number of steps (batches of samples)
            before declaring predictions finished.
            Ignored with the default value of `None`.
        callbacks: List of callbacks or an instance of
            `keras.callbacks.CallbackList` to be called during evaluation.

    # Returns
        Scalar loss (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs
        and/or metrics). The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.
    "
  [model f ins & {:keys [batch_size verbose steps callbacks]
                       :or {verbose 0}} ]
    (py/call-attr-kw training-arrays "test_loop" [model f ins] {:batch_size batch_size :verbose verbose :steps steps :callbacks callbacks }))

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
  [x & {:keys [allow_tuple]
                       :or {allow_tuple false}} ]
    (py/call-attr-kw training-arrays "to_list" [x] {:allow_tuple allow_tuple }))

(defn unpack-singleton 
  "Gets the first element if the iterable has only one value.

    Otherwise return the iterable.

    # Argument
        x: A list or tuple.

    # Returns
        The same iterable or the first element.
    "
  [ x ]
  (py/call-attr training-arrays "unpack_singleton"  x ))
