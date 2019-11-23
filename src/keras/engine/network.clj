(ns keras.engine.network
  "A `Network` is way to compose layers: the topological form of a `Model`.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce network (import-module "keras.engine.network"))

(defn ask-to-proceed-with-overwrite 
  "Produces a prompt asking about overwriting a file.

    # Arguments
        filepath: the path to the file to be overwritten.

    # Returns
        True if we can proceed with overwrite, False otherwise.
    "
  [ & {:keys [filepath]} ]
   (py/call-attr-kw network "ask_to_proceed_with_overwrite" [] {:filepath filepath }))

(defn get-source-inputs 
  "Returns the list of input tensors necessary to compute `tensor`.

    Output will always be a list of tensors
    (potentially with 1 element).

    # Arguments
        tensor: The tensor to start from.
        layer: Origin layer of the tensor. Will be
            determined via tensor._keras_history if not provided.
        node_index: Origin node index of the tensor.

    # Returns
        List of input tensors.
    "
  [ & {:keys [tensor layer node_index]} ]
   (py/call-attr-kw network "get_source_inputs" [] {:tensor tensor :layer layer :node_index node_index }))

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
  
   (py/call-attr-kw network "has_arg" [] {:fn fn :name name :accept_all accept_all }))

(defn object-list-uid 
  ""
  [ & {:keys [object_list]} ]
   (py/call-attr-kw network "object_list_uid" [] {:object_list object_list }))

(defn print-layer-summary 
  "Prints a summary of a model.

    # Arguments
        model: Keras model instance.
        line_length: Total length of printed lines
            (e.g. set this to adapt the display to different
            terminal window sizes).
        positions: Relative or absolute positions of log elements in each line.
            If not provided, defaults to `[.33, .55, .67, 1.]`.
        print_fn: Print function to use.
            It will be called on each line of the summary.
            You can set it to a custom function
            in order to capture the string summary.
            It defaults to `print` (prints to stdout).
    "
  [ & {:keys [model line_length positions print_fn]} ]
   (py/call-attr-kw network "print_layer_summary" [] {:model model :line_length line_length :positions positions :print_fn print_fn }))

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
  
   (py/call-attr-kw network "to_list" [] {:x x :allow_tuple allow_tuple }))

(defn unpack-singleton 
  "Gets the first element if the iterable has only one value.

    Otherwise return the iterable.

    # Argument:
        x: A list or tuple.

    # Returns:
        The same iterable or the first element.
    "
  [ & {:keys [x]} ]
   (py/call-attr-kw network "unpack_singleton" [] {:x x }))
