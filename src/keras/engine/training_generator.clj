(ns keras.engine.training-generator
  "Part of the training engine related to Python generators of array data.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce training-generator (import-module "keras.engine.training_generator"))

(defn evaluate-generator 
  "See docstring for `Model.evaluate_generator`."
  [ & {:keys [model generator steps max_queue_size workers use_multiprocessing verbose]
       :or {max_queue_size 10 workers 1 use_multiprocessing false verbose 0}} ]
  
   (py/call-attr-kw training-generator "evaluate_generator" [] {:model model :generator generator :steps steps :max_queue_size max_queue_size :workers workers :use_multiprocessing use_multiprocessing :verbose verbose }))

(defn fit-generator 
  "See docstring for `Model.fit_generator`."
  [ & {:keys [model generator steps_per_epoch epochs verbose callbacks validation_data validation_steps class_weight max_queue_size workers use_multiprocessing shuffle initial_epoch]
       :or {epochs 1 verbose 1 max_queue_size 10 workers 1 use_multiprocessing false shuffle true initial_epoch 0}} ]
  
   (py/call-attr-kw training-generator "fit_generator" [] {:model model :generator generator :steps_per_epoch steps_per_epoch :epochs epochs :verbose verbose :callbacks callbacks :validation_data validation_data :validation_steps validation_steps :class_weight class_weight :max_queue_size max_queue_size :workers workers :use_multiprocessing use_multiprocessing :shuffle shuffle :initial_epoch initial_epoch }))

(defn iter-sequence-infinite 
  "Iterate indefinitely over a Sequence.

    # Arguments
        seq: Sequence object

    # Returns
        Generator yielding batches.
    "
  [ & {:keys [seq]} ]
   (py/call-attr-kw training-generator "iter_sequence_infinite" [] {:seq seq }))

(defn predict-generator 
  "See docstring for `Model.predict_generator`."
  [ & {:keys [model generator steps max_queue_size workers use_multiprocessing verbose]
       :or {max_queue_size 10 workers 1 use_multiprocessing false verbose 0}} ]
  
   (py/call-attr-kw training-generator "predict_generator" [] {:model model :generator generator :steps steps :max_queue_size max_queue_size :workers workers :use_multiprocessing use_multiprocessing :verbose verbose }))

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
  
   (py/call-attr-kw training-generator "to_list" [] {:x x :allow_tuple allow_tuple }))

(defn unpack-singleton 
  "Gets the first element if the iterable has only one value.

    Otherwise return the iterable.

    # Argument:
        x: A list or tuple.

    # Returns:
        The same iterable or the first element.
    "
  [ & {:keys [x]} ]
   (py/call-attr-kw training-generator "unpack_singleton" [] {:x x }))
