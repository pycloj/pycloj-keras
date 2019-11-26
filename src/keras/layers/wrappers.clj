(ns keras.layers.wrappers
  "Layers that augment the functionality of a base layer.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce wrappers (import-module "keras.layers.wrappers"))

(defn disable-tracking 
  ""
  [ func ]
  (py/call-attr wrappers "disable_tracking"  func ))

(defn has-arg 
  "Checks if a callable accepts a given keyword argument.

    For Python 2, checks if there is an argument with the given name.

    For Python 3, checks if there is an argument with the given name, and
    also whether this argument can be called with a keyword (i.e. if it is
    not a positional-only argument).

    # Arguments
        fn: Callable to inspect.
        name: Check if `fn` can be called with `name` as a keyword argument.
        accept_all: What to return if there is no parameter called `name`
                    but the function accepts a `**kwargs` argument.

    # Returns
        bool, whether `fn` accepts a `name` keyword argument.
    "
  [fn name & {:keys [accept_all]
                       :or {accept_all false}} ]
    (py/call-attr-kw wrappers "has_arg" [fn name] {:accept_all accept_all }))

(defn object-list-uid 
  ""
  [ object_list ]
  (py/call-attr wrappers "object_list_uid"  object_list ))
