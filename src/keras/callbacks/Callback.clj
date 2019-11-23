(ns keras.callbacks.Callback
  "Abstract base class used to build new callbacks.

    # Properties
        params: dict. Training parameters
            (eg. verbosity, batch size, number of epochs...).
        model: instance of `keras.models.Model`.
            Reference of the model being trained.

    The `logs` dictionary that callback methods
    take as argument will contain keys for quantities relevant to
    the current batch or epoch.

    Currently, the `.fit()` method of the `Sequential` model class
    will include the following quantities in the `logs` that
    it passes to its callbacks:

        on_epoch_end: logs include `acc` and `loss`, and
            optionally include `val_loss`
            (if validation is enabled in `fit`), and `val_acc`
            (if validation and accuracy monitoring are enabled).
        on_batch_begin: logs include `size`,
            the number of samples in the current batch.
        on_batch_end: logs include `loss`, and optionally `acc`
            (if accuracy monitoring is enabled).
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

(defn Callback 
  "Abstract base class used to build new callbacks.

    # Properties
        params: dict. Training parameters
            (eg. verbosity, batch size, number of epochs...).
        model: instance of `keras.models.Model`.
            Reference of the model being trained.

    The `logs` dictionary that callback methods
    take as argument will contain keys for quantities relevant to
    the current batch or epoch.

    Currently, the `.fit()` method of the `Sequential` model class
    will include the following quantities in the `logs` that
    it passes to its callbacks:

        on_epoch_end: logs include `acc` and `loss`, and
            optionally include `val_loss`
            (if validation is enabled in `fit`), and `val_acc`
            (if validation and accuracy monitoring are enabled).
        on_batch_begin: logs include `size`,
            the number of samples in the current batch.
        on_batch_end: logs include `loss`, and optionally `acc`
            (if accuracy monitoring is enabled).
    "
  [  ]
  (py/call-attr callbacks "Callback"   ))

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
