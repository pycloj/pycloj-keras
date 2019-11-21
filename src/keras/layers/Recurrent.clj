
(ns keras.layers.recurrent
  "Recurrent layers and their base classes.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw
                     att-type-map
                     ->py-dict
                     ->py-list
                     ]
             :as py]
            [clojure.pprint :as pp]))

(py/initialize!)
(defonce recurrent (import-module "keras.layers.recurrent"))

(defn -generate-dropout-mask 
  ""
  [ & {:keys [ones rate training count]
       :or {count 1}} ]
  
   (py/call-attr-kw recurrent "_generate_dropout_mask" [] {:ones ones :rate rate :training training :count count }))

(defn -standardize-args [ & {:keys [inputs initial_state constants num_constants]} ]
  "Standardize `__call__` to a single list of tensor inputs.

    When running a model loaded from file, the input tensors
    `initial_state` and `constants` can be passed to `RNN.__call__` as part
    of `inputs` instead of by the dedicated keyword arguments. This method
    makes sure the arguments are separated and that `initial_state` and
    `constants` are lists of tensors (or None).

    # Arguments
        inputs: tensor or list/tuple of tensors
        initial_state: tensor or list of tensors or None
        constants: tensor or list of tensors or None

    # Returns
        inputs: tensor
        initial_state: list of tensors or None
        constants: list of tensors or None
    "
   (py/call-attr-kw recurrent "_standardize_args" [] {:inputs inputs :initial_state initial_state :constants constants :num_constants num_constants }))

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
  [ & {:keys [fn name accept_all]
       :or {accept_all false}} ]
  
   (py/call-attr-kw recurrent "has_arg" [] {:fn fn :name name :accept_all accept_all }))

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
  
   (py/call-attr-kw recurrent "to_list" [] {:x x :allow_tuple allow_tuple }))
