(ns keras.callbacks.ProgbarLogger
  "Callback that prints metrics to stdout.

    # Arguments
        count_mode: One of \"steps\" or \"samples\".
            Whether the progress bar should
            count samples seen or steps (batches) seen.
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over an epoch.
            Metrics in this list will be logged as-is.
            All others will be averaged over time (e.g. loss, etc).

    # Raises
        ValueError: In case of invalid `count_mode`.
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

(defn ProgbarLogger 
  "Callback that prints metrics to stdout.

    # Arguments
        count_mode: One of \"steps\" or \"samples\".
            Whether the progress bar should
            count samples seen or steps (batches) seen.
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over an epoch.
            Metrics in this list will be logged as-is.
            All others will be averaged over time (e.g. loss, etc).

    # Raises
        ValueError: In case of invalid `count_mode`.
    "
  [ & {:keys [count_mode stateful_metrics]
       :or {count_mode "samples"}} ]
  
   (py/call-attr-kw callbacks "ProgbarLogger" [] {:count_mode count_mode :stateful_metrics stateful_metrics }))
(defn on-batch-begin 
  ""
  [self batch  & {:keys [logs]} ]
    (py/call-attr-kw self "on_batch_begin" [batch] {:logs logs }))
(defn on-batch-end 
  ""
  [self batch  & {:keys [logs]} ]
    (py/call-attr-kw self "on_batch_end" [batch] {:logs logs }))
(defn on-epoch-begin 
  ""
  [self epoch  & {:keys [logs]} ]
    (py/call-attr-kw self "on_epoch_begin" [epoch] {:logs logs }))
(defn on-epoch-end 
  ""
  [self epoch  & {:keys [logs]} ]
    (py/call-attr-kw self "on_epoch_end" [epoch] {:logs logs }))
(defn on-predict-batch-begin 
  "Called at the beginning of a batch in `predict` methods.

        Subclasses should override for any actions to run.

        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dict, has keys `batch` and `size` representing the current
                batch number and the size of the batch.
        "
  [self batch  & {:keys [logs]} ]
    (py/call-attr-kw self "on_predict_batch_begin" [batch] {:logs logs }))
(defn on-predict-batch-end 
  "Called at the end of a batch in `predict` methods.

        Subclasses should override for any actions to run.

        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dict, metric results for this batch.
        "
  [self batch  & {:keys [logs]} ]
    (py/call-attr-kw self "on_predict_batch_end" [batch] {:logs logs }))
(defn on-predict-begin 
  "Called at the beginning of prediction.

        Subclasses should override for any actions to run.

        # Arguments
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        "
  [self   & {:keys [logs]} ]
    (py/call-attr-kw self "on_predict_begin" [] {:logs logs }))
(defn on-predict-end 
  "Called at the end of prediction.

        Subclasses should override for any actions to run.

        # Arguments
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        "
  [self   & {:keys [logs]} ]
    (py/call-attr-kw self "on_predict_end" [] {:logs logs }))
(defn on-test-batch-begin 
  "Called at the beginning of a batch in `evaluate` methods.

        Also called at the beginning of a validation batch in the `fit` methods,
        if validation data is provided.

        Subclasses should override for any actions to run.

        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dict, has keys `batch` and `size` representing the current
                batch number and the size of the batch.
        "
  [self batch  & {:keys [logs]} ]
    (py/call-attr-kw self "on_test_batch_begin" [batch] {:logs logs }))
(defn on-test-batch-end 
  "Called at the end of a batch in `evaluate` methods.

        Also called at the end of a validation batch in the `fit` methods,
        if validation data is provided.

        Subclasses should override for any actions to run.

        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dict, metric results for this batch.
        "
  [self batch  & {:keys [logs]} ]
    (py/call-attr-kw self "on_test_batch_end" [batch] {:logs logs }))
(defn on-test-begin 
  "Called at the beginning of evaluation or validation.

        Subclasses should override for any actions to run.

        # Arguments
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        "
  [self   & {:keys [logs]} ]
    (py/call-attr-kw self "on_test_begin" [] {:logs logs }))
(defn on-test-end 
  "Called at the end of evaluation or validation.

        Subclasses should override for any actions to run.

        # Arguments
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        "
  [self   & {:keys [logs]} ]
    (py/call-attr-kw self "on_test_end" [] {:logs logs }))
(defn on-train-batch-begin 
  "Called at the beginning of a training batch in `fit` methods.

        Subclasses should override for any actions to run.

        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dict, has keys `batch` and `size` representing the current
                batch number and the size of the batch.
        "
  [self batch  & {:keys [logs]} ]
    (py/call-attr-kw self "on_train_batch_begin" [batch] {:logs logs }))
(defn on-train-batch-end 
  "Called at the end of a training batch in `fit` methods.

        Subclasses should override for any actions to run.

        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dict, metric results for this batch.
        "
  [self batch  & {:keys [logs]} ]
    (py/call-attr-kw self "on_train_batch_end" [batch] {:logs logs }))
(defn on-train-begin 
  ""
  [self   & {:keys [logs]} ]
    (py/call-attr-kw self "on_train_begin" [] {:logs logs }))
(defn on-train-end 
  "Called at the end of training.

        Subclasses should override for any actions to run.

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
