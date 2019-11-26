(ns keras.utils.data-utils.ThreadPool
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce data-utils (import-module "keras.utils.data_utils"))

(defn ThreadPool 
  ""
  [ & {:keys [processes initializer initargs]
       :or {initargs ()}} ]
  
   (py/call-attr-kw data-utils "ThreadPool" [] {:processes processes :initializer initializer :initargs initargs }))

(defn Process 
  ""
  [ self  ]
  (py/call-attr self "Process"  self  ))

(defn apply 
  "
        Equivalent of `func(*args, **kwds)`.
        "
  [self func & {:keys [args kwds]
                       :or {args () kwds {}}} ]
    (py/call-attr-kw self "apply" [func] {:args args :kwds kwds }))

(defn apply-async 
  "
        Asynchronous version of `apply()` method.
        "
  [self func & {:keys [args kwds callback error_callback]
                       :or {args () kwds {}}} ]
    (py/call-attr-kw self "apply_async" [func] {:args args :kwds kwds :callback callback :error_callback error_callback }))

(defn close 
  ""
  [ self  ]
  (py/call-attr self "close"  self  ))

(defn imap 
  "
        Equivalent of `map()` -- can be MUCH slower than `Pool.map()`.
        "
  [self func iterable & {:keys [chunksize]
                       :or {chunksize 1}} ]
    (py/call-attr-kw self "imap" [func iterable] {:chunksize chunksize }))

(defn imap-unordered 
  "
        Like `imap()` method but ordering of results is arbitrary.
        "
  [self func iterable & {:keys [chunksize]
                       :or {chunksize 1}} ]
    (py/call-attr-kw self "imap_unordered" [func iterable] {:chunksize chunksize }))

(defn join 
  ""
  [ self  ]
  (py/call-attr self "join"  self  ))
(defn map 
  "
        Apply `func` to each element in `iterable`, collecting the results
        in a list that is returned.
        "
  [self func iterable  & {:keys [chunksize]} ]
    (py/call-attr-kw self "map" [func iterable] {:chunksize chunksize }))
(defn map-async 
  "
        Asynchronous version of `map()` method.
        "
  [self func iterable  & {:keys [chunksize callback error_callback]} ]
    (py/call-attr-kw self "map_async" [func iterable] {:chunksize chunksize :callback callback :error_callback error_callback }))
(defn starmap 
  "
        Like `map()` method but the elements of the `iterable` are expected to
        be iterables as well and will be unpacked as arguments. Hence
        `func` and (a, b) becomes func(a, b).
        "
  [self func iterable  & {:keys [chunksize]} ]
    (py/call-attr-kw self "starmap" [func iterable] {:chunksize chunksize }))
(defn starmap-async 
  "
        Asynchronous version of `starmap()` method.
        "
  [self func iterable  & {:keys [chunksize callback error_callback]} ]
    (py/call-attr-kw self "starmap_async" [func iterable] {:chunksize chunksize :callback callback :error_callback error_callback }))

(defn terminate 
  ""
  [ self  ]
  (py/call-attr self "terminate"  self  ))
