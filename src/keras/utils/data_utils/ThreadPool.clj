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
  [  ]
  (py/call-attr data-utils "Process"   ))

(defn apply 
  "
        Equivalent of `func(*args, **kwds)`.
        Pool must be running.
        "
  [self & {:keys [func args kwds]
                       :or {args () kwds {}}} ]
    (py/call-attr-kw data-utils "apply" [] {:func func :args args :kwds kwds }))

(defn apply-async 
  "
        Asynchronous version of `apply()` method.
        "
  [self & {:keys [func args kwds callback error_callback]
                       :or {args () kwds {}}} ]
    (py/call-attr-kw data-utils "apply_async" [] {:func func :args args :kwds kwds :callback callback :error_callback error_callback }))

(defn close 
  ""
  [ self ]
  (py/call-attr data-utils "close"  self ))

(defn imap 
  "
        Equivalent of `map()` -- can be MUCH slower than `Pool.map()`.
        "
  [self & {:keys [func iterable chunksize]
                       :or {chunksize 1}} ]
    (py/call-attr-kw data-utils "imap" [] {:func func :iterable iterable :chunksize chunksize }))

(defn imap-unordered 
  "
        Like `imap()` method but ordering of results is arbitrary.
        "
  [self & {:keys [func iterable chunksize]
                       :or {chunksize 1}} ]
    (py/call-attr-kw data-utils "imap_unordered" [] {:func func :iterable iterable :chunksize chunksize }))

(defn join 
  ""
  [ self ]
  (py/call-attr data-utils "join"  self ))

(defn map 
  "
        Apply `func` to each element in `iterable`, collecting the results
        in a list that is returned.
        "
  [self  & {:keys [func iterable chunksize]} ]
    (py/call-attr-kw data-utils "map" [self] {:func func :iterable iterable :chunksize chunksize }))

(defn map-async 
  "
        Asynchronous version of `map()` method.
        "
  [self  & {:keys [func iterable chunksize callback error_callback]} ]
    (py/call-attr-kw data-utils "map_async" [self] {:func func :iterable iterable :chunksize chunksize :callback callback :error_callback error_callback }))

(defn starmap 
  "
        Like `map()` method but the elements of the `iterable` are expected to
        be iterables as well and will be unpacked as arguments. Hence
        `func` and (a, b) becomes func(a, b).
        "
  [self  & {:keys [func iterable chunksize]} ]
    (py/call-attr-kw data-utils "starmap" [self] {:func func :iterable iterable :chunksize chunksize }))

(defn starmap-async 
  "
        Asynchronous version of `starmap()` method.
        "
  [self  & {:keys [func iterable chunksize callback error_callback]} ]
    (py/call-attr-kw data-utils "starmap_async" [self] {:func func :iterable iterable :chunksize chunksize :callback callback :error_callback error_callback }))

(defn terminate 
  ""
  [ self ]
  (py/call-attr data-utils "terminate"  self ))
