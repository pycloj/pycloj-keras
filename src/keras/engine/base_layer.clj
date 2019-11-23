(ns keras.engine.base-layer
  "Contains the base Layer class, from which all layers inherit.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce base-layer (import-module "keras.engine.base_layer"))

(defn count-params 
  "Count the total number of scalars composing the weights.

    # Arguments
        weights: An iterable containing the weights on which to compute params

    # Returns
        The total number of scalars composing the weights
    "
  [ & {:keys [weights]} ]
   (py/call-attr-kw base-layer "count_params" [] {:weights weights }))

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
  
   (py/call-attr-kw base-layer "has_arg" [] {:fn fn :name name :accept_all accept_all }))

(defn is-all-none 
  ""
  [ & {:keys [iterable_or_element]} ]
   (py/call-attr-kw base-layer "is_all_none" [] {:iterable_or_element iterable_or_element }))

(defn object-list-uid 
  ""
  [ & {:keys [object_list]} ]
   (py/call-attr-kw base-layer "object_list_uid" [] {:object_list object_list }))

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
  
   (py/call-attr-kw base-layer "to_list" [] {:x x :allow_tuple allow_tuple }))

(defn unpack-singleton 
  "Gets the first element if the iterable has only one value.

    Otherwise return the iterable.

    # Argument:
        x: A list or tuple.

    # Returns:
        The same iterable or the first element.
    "
  [ & {:keys [x]} ]
   (py/call-attr-kw base-layer "unpack_singleton" [] {:x x }))
