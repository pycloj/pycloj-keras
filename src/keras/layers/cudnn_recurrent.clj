(ns keras.layers.cudnn-recurrent
  "Recurrent layers backed by cuDNN.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce cudnn-recurrent (import-module "keras.layers.cudnn_recurrent"))

(defn namedtuple 
  "Returns a new subclass of tuple with named fields.

    >>> Point = namedtuple('Point', ['x', 'y'])
    >>> Point.__doc__                   # docstring for the new class
    'Point(x, y)'
    >>> p = Point(11, y=22)             # instantiate with positional args or keywords
    >>> p[0] + p[1]                     # indexable like a plain tuple
    33
    >>> x, y = p                        # unpack like a regular tuple
    >>> x, y
    (11, 22)
    >>> p.x + p.y                       # fields also accessible by name
    33
    >>> d = p._asdict()                 # convert to a dictionary
    >>> d['x']
    11
    >>> Point(**d)                      # convert from a dictionary
    Point(x=11, y=22)
    >>> p._replace(x=100)               # _replace() is like str.replace() but targets named fields
    Point(x=100, y=22)

    "
  [ & {:keys [typename field_names rename defaults module]
       :or {rename false}} ]
  
   (py/call-attr-kw cudnn-recurrent "namedtuple" [] {:typename typename :field_names field_names :rename rename :defaults defaults :module module }))
