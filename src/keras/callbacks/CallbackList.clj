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
  [ self callback ]
  (py/call-attr self "append"  self callback ))
(defn on-batch-begin 
  ""
  [self batch  & {:keys [logs]} ]
    (py/call-attr-kw self "on_batch_begin" [batch] {:logs logs }))
(defn on-batch-end 
  ""
  [self batch  & {:keys [logs]} ]
    (py/call-attr-kw self "on_batch_end" [batch] {:logs logs }))
(defn on-epoch-begin 
  "Calls the `on_epoch_begin` methods of its callbacks.

        This function should only be called during train mode.

        # Arguments
            epoch: integer, index of epoch.
            logs: dict, Currently no data is passed to this argument for this method
                but that may change in the future.
        "
  [self epoch  & {:keys [logs]} ]
    (py/call-attr-kw self "on_epoch_begin" [epoch] {:logs logs }))
(defn on-epoch-end 
  "Calls the `on_epoch_end` methods of its callbacks.

        This function should only be called during train mode.

        # Arguments
            epoch: integer, index of epoch.
            logs: dict, metric results for this training epoch, and for the
                validation epoch if validation is performed. Validation result keys
                are prefixed with `val_`.
        "
  [self epoch  & {:keys [logs]} ]
    (py/call-attr-kw self "on_epoch_end" [epoch] {:logs logs }))
(defn on-predict-batch-begin 
  "Calls the `on_predict_batch_begin` methods of its callbacks.

        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dict, has keys `batch` and `size` representing the current
                batch number and the size of the batch.
        "
  [self batch  & {:keys [logs]} ]
    (py/call-attr-kw self "on_predict_batch_begin" [batch] {:logs logs }))
(defn on-predict-batch-end 
  "Calls the `on_predict_batch_end` methods of its callbacks.

        # Argument
            batch: integer, index of batch within the current epoch.
            logs: dict, metric results for this batch.
        "
  [self batch  & {:keys [logs]} ]
    (py/call-attr-kw self "on_predict_batch_end" [batch] {:logs logs }))
(defn on-predict-begin 
  "Calls the `on_predict_begin` methods of its callbacks.

        # Arguments
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        "
  [self   & {:keys [logs]} ]
    (py/call-attr-kw self "on_predict_begin" [] {:logs logs }))
(defn on-predict-end 
  "Calls the `on_predict_end` methods of its callbacks.

        # Arguments
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        "
  [self   & {:keys [logs]} ]
    (py/call-attr-kw self "on_predict_end" [] {:logs logs }))
(defn on-test-batch-begin 
  "Calls the `on_test_batch_begin` methods of its callbacks.

        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dict, has keys `batch` and `size` representing the current
                batch number and the size of the batch.
        "
  [self batch  & {:keys [logs]} ]
    (py/call-attr-kw self "on_test_batch_begin" [batch] {:logs logs }))
(defn on-test-batch-end 
  "Calls the `on_test_batch_end` methods of its callbacks.

        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dict, metric results for this batch.
        "
  [self batch  & {:keys [logs]} ]
    (py/call-attr-kw self "on_test_batch_end" [batch] {:logs logs }))
(defn on-test-begin 
  "Calls the `on_test_begin` methods of its callbacks.

        # Arguments
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        "
  [self   & {:keys [logs]} ]
    (py/call-attr-kw self "on_test_begin" [] {:logs logs }))
(defn on-test-end 
  "Calls the `on_test_end` methods of its callbacks.

        # Arguments
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        "
  [self   & {:keys [logs]} ]
    (py/call-attr-kw self "on_test_end" [] {:logs logs }))
(defn on-train-batch-begin 
  "Calls the `on_train_batch_begin` methods of its callbacks.

        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dict, has keys `batch` and `size` representing the current
                batch number and the size of the batch.
        "
  [self batch  & {:keys [logs]} ]
    (py/call-attr-kw self "on_train_batch_begin" [batch] {:logs logs }))
(defn on-train-batch-end 
  "Calls the `on_train_batch_end` methods of its callbacks.

        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dict, metric results for this batch.
        "
  [self batch  & {:keys [logs]} ]
    (py/call-attr-kw self "on_train_batch_end" [batch] {:logs logs }))
(defn on-train-begin 
  "Calls the `on_train_begin` methods of its callbacks.

        # Arguments
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        "
  [self   & {:keys [logs]} ]
    (py/call-attr-kw self "on_train_begin" [] {:logs logs }))
(defn on-train-end 
  "Calls the `on_train_end` methods of its callbacks.

        # Arguments
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        "
  [self   & {:keys [logs]} ]
    (py/call-attr-kw self "on_train_end" [] {:logs logs }))

(defn set-model 
  ""
  [ self model ]
  (py/call-attr self "set_model"  self model ))

(defn set-params 
  ""
  [ self params ]
  (py/call-attr self "set_params"  self params ))
