(ns keras.engine.training-generator.OrderedEnqueuer
  "Builds a Enqueuer from a Sequence.

    Used in `fit_generator`, `evaluate_generator`, `predict_generator`.

    # Arguments
        sequence: A `keras.utils.data_utils.Sequence` object.
        use_multiprocessing: use multiprocessing if True, otherwise threading
        shuffle: whether to shuffle the data at the beginning of each epoch
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

(defn OrderedEnqueuer 
  "Builds a Enqueuer from a Sequence.

    Used in `fit_generator`, `evaluate_generator`, `predict_generator`.

    # Arguments
        sequence: A `keras.utils.data_utils.Sequence` object.
        use_multiprocessing: use multiprocessing if True, otherwise threading
        shuffle: whether to shuffle the data at the beginning of each epoch
    "
  [ & {:keys [sequence use_multiprocessing shuffle]
       :or {use_multiprocessing false shuffle false}} ]
  
   (py/call-attr-kw training-generator "OrderedEnqueuer" [] {:sequence sequence :use_multiprocessing use_multiprocessing :shuffle shuffle }))

(defn get 
  "Creates a generator to extract data from the queue.

        Skip the data if it is `None`.

        # Yields
            The next element in the queue, i.e. a tuple
            `(inputs, targets)` or
            `(inputs, targets, sample_weights)`.
        "
  [ self ]
  (py/call-attr training-generator "get"  self ))

(defn is-running 
  ""
  [ self ]
  (py/call-attr training-generator "is_running"  self ))

(defn start 
  "Start the handler's workers.

        # Arguments
            workers: number of worker threads
            max_queue_size: queue size
                (when full, workers could block on `put()`)
        "
  [self & {:keys [workers max_queue_size]
                       :or {workers 1 max_queue_size 10}} ]
    (py/call-attr-kw training-generator "start" [] {:workers workers :max_queue_size max_queue_size }))

(defn stop 
  "Stops running threads and wait for them to exit, if necessary.

        Should be called by the same thread which called `start()`.

        # Arguments
            timeout: maximum time to wait on `thread.join()`
        "
  [self  & {:keys [timeout]} ]
    (py/call-attr-kw training-generator "stop" [self] {:timeout timeout }))
