(ns keras.backend.tensorflow-backend.name-scope
  "A context manager for use when defining a Python op.

  This context manager pushes a name scope, which will make the name of all
  operations added within it have a prefix.

  For example, to define a new Python op called `my_op`:

  ```python
  def my_op(a, b, c, name=None):
    with tf.name_scope(\"MyOp\") as scope:
      a = tf.convert_to_tensor(a, name=\"a\")
      b = tf.convert_to_tensor(b, name=\"b\")
      c = tf.convert_to_tensor(c, name=\"c\")
      # Define some computation that uses `a`, `b`, and `c`.
      return foo_op(..., name=scope)
  ```

  When executed, the Tensors `a`, `b`, `c`, will have names `MyOp/a`, `MyOp/b`,
  and `MyOp/c`.

  If the scope name already exists, the name will be made unique by appending
  `_n`. For example, calling `my_op` the second time will generate `MyOp_1/a`,
  etc.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tensorflow-backend (import-module "keras.backend.tensorflow_backend"))

(defn name-scope 
  "A context manager for use when defining a Python op.

  This context manager pushes a name scope, which will make the name of all
  operations added within it have a prefix.

  For example, to define a new Python op called `my_op`:

  ```python
  def my_op(a, b, c, name=None):
    with tf.name_scope(\"MyOp\") as scope:
      a = tf.convert_to_tensor(a, name=\"a\")
      b = tf.convert_to_tensor(b, name=\"b\")
      c = tf.convert_to_tensor(c, name=\"c\")
      # Define some computation that uses `a`, `b`, and `c`.
      return foo_op(..., name=scope)
  ```

  When executed, the Tensors `a`, `b`, `c`, will have names `MyOp/a`, `MyOp/b`,
  and `MyOp/c`.

  If the scope name already exists, the name will be made unique by appending
  `_n`. For example, calling `my_op` the second time will generate `MyOp_1/a`,
  etc.
  "
  [ name ]
  (py/call-attr tensorflow-backend "name_scope"  name ))

(defn name 
  ""
  [ self ]
    (py/call-attr self "name"))
