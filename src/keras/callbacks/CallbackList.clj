(ns keras.callbacks.CallbackList
  "Container abstracting a list of callbacks.

    # Arguments
        callbacks: List of `Callback` instances.
        queue_length: Queue length for keeping
            running statistics over callback execution time.
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce callbacks (import-module "keras.callbacks"))

(defn CallbackList 
  "Container abstracting a list of callbacks.

    # Arguments
        callbacks: List of `Callback` instances.
        queue_length: Queue length for keeping
            running statistics over callback execution time.
    "
  [ & {:keys [callbacks queue_length]
       :or {queue_length 10}} ]
  
   (py/call-attr-kw callbacks "CallbackList" [] {:callbacks callbacks :queue_length queue_length }))

(defn append 
  ""
  [self  & {:keys [callback]} ]
    (py/call-attr-kw callbacks "append" [self] {:callback callback }))

(defn on-batch-begin 
  "Called right before processing a batch.

        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        "
  [self  & {:keys [batch logs]} ]
    (py/call-attr-kw callbacks "on_batch_begin" [self] {:batch batch :logs logs }))

(defn on-batch-end 
  "Called at the end of a batch.

        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        "
  [self  & {:keys [batch logs]} ]
    (py/call-attr-kw callbacks "on_batch_end" [self] {:batch batch :logs logs }))

(defn on-epoch-begin 
  "Called at the start of an epoch.

        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs.
        "
  [self  & {:keys [epoch logs]} ]
    (py/call-attr-kw callbacks "on_epoch_begin" [self] {:epoch epoch :logs logs }))

(defn on-epoch-end 
  "Called at the end of an epoch.

        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs.
        "
  [self  & {:keys [epoch logs]} ]
    (py/call-attr-kw callbacks "on_epoch_end" [self] {:epoch epoch :logs logs }))

(defn on-train-begin 
  "Called at the beginning of training.

        # Arguments
            logs: dictionary of logs.
        "
  [self  & {:keys [logs]} ]
    (py/call-attr-kw callbacks "on_train_begin" [self] {:logs logs }))

(defn on-train-end 
  "Called at the end of training.

        # Arguments
            logs: dictionary of logs.
        "
  [self  & {:keys [logs]} ]
    (py/call-attr-kw callbacks "on_train_end" [self] {:logs logs }))

(defn set-model 
  ""
  [self  & {:keys [model]} ]
    (py/call-attr-kw callbacks "set_model" [self] {:model model }))

(defn set-params 
  ""
  [self  & {:keys [params]} ]
    (py/call-attr-kw callbacks "set_params" [self] {:params params }))
