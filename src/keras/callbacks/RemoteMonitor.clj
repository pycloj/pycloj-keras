(ns keras.callbacks.RemoteMonitor
  "Callback used to stream events to a server.

    Requires the `requests` library.
    Events are sent to `root + '/publish/epoch/end/'` by default. Calls are
    HTTP POST, with a `data` argument which is a
    JSON-encoded dictionary of event data.
    If send_as_json is set to True, the content type of the request will be
    application/json. Otherwise the serialized JSON will be send within a form

    # Arguments
        root: String; root url of the target server.
        path: String; path relative to `root` to which the events will be sent.
        field: String; JSON field under which the data will be stored.
            The field is used only if the payload is sent within a form
            (i.e. send_as_json is set to False).
        headers: Dictionary; optional custom HTTP headers.
        send_as_json: Boolean; whether the request should be send as
            application/json.
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

(defn RemoteMonitor 
  "Callback used to stream events to a server.

    Requires the `requests` library.
    Events are sent to `root + '/publish/epoch/end/'` by default. Calls are
    HTTP POST, with a `data` argument which is a
    JSON-encoded dictionary of event data.
    If send_as_json is set to True, the content type of the request will be
    application/json. Otherwise the serialized JSON will be send within a form

    # Arguments
        root: String; root url of the target server.
        path: String; path relative to `root` to which the events will be sent.
        field: String; JSON field under which the data will be stored.
            The field is used only if the payload is sent within a form
            (i.e. send_as_json is set to False).
        headers: Dictionary; optional custom HTTP headers.
        send_as_json: Boolean; whether the request should be send as
            application/json.
    "
  [ & {:keys [root path field headers send_as_json]
       :or {root "http://localhost:9000" path "/publish/epoch/end/" field "data" send_as_json false}} ]
  
   (py/call-attr-kw callbacks "RemoteMonitor" [] {:root root :path path :field field :headers headers :send_as_json send_as_json }))

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
