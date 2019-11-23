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
  [ & {:keys [index_array batch_size]} ]
   (py/call-attr-kw training-arrays "batch_shuffle" [] {:index_array index_array :batch_size batch_size }))

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
  
   (py/call-attr-kw training-arrays "check_num_samples" [] {:ins ins :batch_size batch_size :steps steps :steps_name steps_name }))

(defn fit-loop 
  "Abstract fit function for `f(ins)`.

    Assumes that f returns a list, labeled by out_labels.

    # Arguments
        model: Keras model instance.
        f: Keras function returning a list of tensors
        ins: List of tensors to be fed to `f`
        out_labels: List of strings, display names of
            the outputs of `f`
        batch_size: Integer batch size or None if unknown.
        epochs: Number of times to iterate over the data
        verbose: Verbosity mode, 0, 1 or 2
        callbacks: List of callbacks to be called during training
        val_f: Keras function to call for validation
        val_ins: List of tensors to be fed to `val_f`
        shuffle: Whether to shuffle the data at the beginning of each epoch
        callback_metrics: List of strings, the display names of the metrics
            passed to the callbacks. They should be the
            concatenation of list the display names of the outputs of
             `f` and the list of display names of the outputs of `f_val`.
        initial_epoch: Epoch at which to start training
            (useful for resuming a previous training run)
        steps_per_epoch: Total number of steps (batches of samples)
            before declaring one epoch finished and starting the
            next epoch. Ignored with the default value of `None`.
        validation_steps: Number of steps to run validation for
            (only if doing validation from data tensors).
            Ignored with the default value of `None`.

    # Returns
        `History` object.
    "
  [ & {:keys [model f ins out_labels batch_size epochs verbose callbacks val_f val_ins shuffle callback_metrics initial_epoch steps_per_epoch validation_steps]
       :or {epochs 100 verbose 1 shuffle true initial_epoch 0}} ]
  
   (py/call-attr-kw training-arrays "fit_loop" [] {:model model :f f :ins ins :out_labels out_labels :batch_size batch_size :epochs epochs :verbose verbose :callbacks callbacks :val_f val_f :val_ins val_ins :shuffle shuffle :callback_metrics callback_metrics :initial_epoch initial_epoch :steps_per_epoch steps_per_epoch :validation_steps validation_steps }))

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
  [ & {:keys [x]} ]
   (py/call-attr-kw training-arrays "issparse" [] {:x x }))

(defn make-batches 
  "Returns a list of batch indices (tuples of indices).

    # Arguments
        size: Integer, total size of the data to slice into batches.
        batch_size: Integer, batch size.

    # Returns
        A list of tuples of array indices.
    "
  [ & {:keys [size batch_size]} ]
   (py/call-attr-kw training-arrays "make_batches" [] {:size size :batch_size batch_size }))

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

    # Returns
        Array of predictions (if the model has a single output)
        or list of arrays of predictions
        (if the model has multiple outputs).
    "
  [ & {:keys [model f ins batch_size verbose steps]
       :or {batch_size 32 verbose 0}} ]
  
   (py/call-attr-kw training-arrays "predict_loop" [] {:model model :f f :ins ins :batch_size batch_size :verbose verbose :steps steps }))

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
   (py/call-attr-kw training-arrays "slice_arrays" [] {:arrays arrays :start start :stop stop }))

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

    # Returns
        Scalar loss (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs
        and/or metrics). The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.
    "
  [ & {:keys [model f ins batch_size verbose steps]
       :or {verbose 0}} ]
  
   (py/call-attr-kw training-arrays "test_loop" [] {:model model :f f :ins ins :batch_size batch_size :verbose verbose :steps steps }))

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
  
   (py/call-attr-kw training-arrays "to_list" [] {:x x :allow_tuple allow_tuple }))

(defn unpack-singleton 
  "Gets the first element if the iterable has only one value.

    Otherwise return the iterable.

    # Argument:
        x: A list or tuple.

    # Returns:
        The same iterable or the first element.
    "
  [ & {:keys [x]} ]
   (py/call-attr-kw training-arrays "unpack_singleton" [] {:x x }))
