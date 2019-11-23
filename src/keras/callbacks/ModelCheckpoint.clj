(ns keras.callbacks.ModelCheckpoint
  "Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
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

(defn ModelCheckpoint 
  "Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    "
  [ & {:keys [filepath monitor verbose save_best_only save_weights_only mode period]
       :or {monitor "val_loss" verbose 0 save_best_only false save_weights_only false mode "auto" period 1}} ]
  
   (py/call-attr-kw callbacks "ModelCheckpoint" [] {:filepath filepath :monitor monitor :verbose verbose :save_best_only save_best_only :save_weights_only save_weights_only :mode mode :period period }))

(defn on-batch-begin 
  ""
  [self  & {:keys [batch logs]} ]
    (py/call-attr-kw callbacks "on_batch_begin" [self] {:batch batch :logs logs }))

(defn on-batch-end 
  ""
  [self  & {:keys [batch logs]} ]
    (py/call-attr-kw callbacks "on_batch_end" [self] {:batch batch :logs logs }))

(defn on-epoch-begin 
  ""
  [self  & {:keys [epoch logs]} ]
    (py/call-attr-kw callbacks "on_epoch_begin" [self] {:epoch epoch :logs logs }))

(defn on-epoch-end 
  ""
  [self  & {:keys [epoch logs]} ]
    (py/call-attr-kw callbacks "on_epoch_end" [self] {:epoch epoch :logs logs }))

(defn on-train-begin 
  ""
  [self  & {:keys [logs]} ]
    (py/call-attr-kw callbacks "on_train_begin" [self] {:logs logs }))

(defn on-train-end 
  ""
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
