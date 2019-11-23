(ns keras.callbacks.deque
  "deque([iterable[, maxlen]]) --> deque object

A list-like sequence optimized for data accesses near its endpoints."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce callbacks (import-module "keras.callbacks"))
