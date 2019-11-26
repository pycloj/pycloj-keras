(ns keras.callbacks.EarlyStopping
  "Stop training when a monitored quantity has stopped improving.

    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs that produced the monitored
            quantity with no improvement after which training will
            be stopped.
            Validation quantities may not be produced for every
            epoch, if the validation frequency
            (`model.fit(validation_freq=5)`) is greater than one.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        baseline: Baseline value for the monitored quantity to reach.
            Training will stop if the model doesn't show improvement
            over the baseline.
        restore_best_weights: whether to restore model weights from
            the epoch with the best value of the monitored quantity.
            If False, the model weights obtained at the last step of
            training are used.
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

(defn EarlyStopping 
  "Stop training when a monitored quantity has stopped improving.

    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs that produced the monitored
            quantity with no improvement after which training will
            be stopped.
            Validation quantities may not be produced for every
            epoch, if the validation frequency
            (`model.fit(validation_freq=5)`) is greater than one.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        baseline: Baseline value for the monitored quantity to reach.
            Training will stop if the model doesn't show improvement
            over the baseline.
        restore_best_weights: whether to restore model weights from
            the epoch with the best value of the monitored quantity.
            If False, the model weights obtained at the last step of
            training are used.
    "
  [ & {:keys [monitor min_delta patience verbose mode baseline restore_best_weights]
       :or {monitor "val_loss" min_delta 0 patience 0 verbose 0 mode "auto" restore_best_weights false}} ]
  
   (py/call-attr-kw callbacks "EarlyStopping" [] {:monitor monitor :min_delta min_delta :patience patience :verbose verbose :mode mode :baseline baseline :restore_best_weights restore_best_weights }))

(defn get-monitor-value 
  ""
  [ self logs ]
  (py/call-attr self "get_monitor_value"  self logs ))
(defn on-batch-begin 
  "A backwards compatibility alias for `on_train_batch_begin`."
  [self batch  & {:keys [logs]} ]
    (py/call-attr-kw self "on_batch_begin" [batch] {:logs logs }))
(defn on-batch-end 
  "A backwards compatibility alias for `on_train_batch_end`."
  [self batch  & {:keys [logs]} ]
    (py/call-attr-kw self "on_batch_end" [batch] {:logs logs }))
(defn on-epoch-begin 
  "Called at the start of an epoch.

        Subclasses should override for any actions to run. This function should only
        be called during train mode.

        # Arguments
            epoch: integer, index of epoch.
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        "
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
  ""
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
