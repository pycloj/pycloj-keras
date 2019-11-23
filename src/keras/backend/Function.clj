(ns keras.backend.Function
  "Runs a computation graph.

    It's possible to pass arguments to `tf.Session.run()` via `session_kwargs`.
    In particular additional operations via `fetches` argument and additional
    tensor substitutions via `feed_dict` arguments. Note that given
    substitutions are merged with substitutions from `inputs`. Even though
    `feed_dict` is passed once in the constructor (called in `model.compile()`)
    we can modify the values in the dictionary. Through this feed_dict we can
    provide additional substitutions besides Keras inputs.

    # Arguments
        inputs: Feed placeholders to the computation graph.
        outputs: Output tensors to fetch.
        updates: Additional update ops to be run at function call.
        name: a name to help users identify what this function does.
        session_kwargs: arguments to `tf.Session.run()`:
            `fetches`, `feed_dict`,
            `options`, `run_metadata`
    "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce backend (import-module "keras.backend"))

(defn Function 
  "Runs a computation graph.

    It's possible to pass arguments to `tf.Session.run()` via `session_kwargs`.
    In particular additional operations via `fetches` argument and additional
    tensor substitutions via `feed_dict` arguments. Note that given
    substitutions are merged with substitutions from `inputs`. Even though
    `feed_dict` is passed once in the constructor (called in `model.compile()`)
    we can modify the values in the dictionary. Through this feed_dict we can
    provide additional substitutions besides Keras inputs.

    # Arguments
        inputs: Feed placeholders to the computation graph.
        outputs: Output tensors to fetch.
        updates: Additional update ops to be run at function call.
        name: a name to help users identify what this function does.
        session_kwargs: arguments to `tf.Session.run()`:
            `fetches`, `feed_dict`,
            `options`, `run_metadata`
    "
  [ & {:keys [inputs outputs updates name]} ]
   (py/call-attr-kw backend "Function" [] {:inputs inputs :outputs outputs :updates updates :name name }))
