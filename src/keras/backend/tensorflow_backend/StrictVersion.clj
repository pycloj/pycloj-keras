(ns keras.backend.tensorflow-backend.StrictVersion
  "Version numbering for anal retentives and software idealists.
    Implements the standard interface for version number classes as
    described above.  A version number consists of two or three
    dot-separated numeric components, with an optional \"pre-release\" tag
    on the end.  The pre-release tag consists of the letter 'a' or 'b'
    followed by a number.  If the numeric components of two version
    numbers are equal, then one with a pre-release tag will always
    be deemed earlier (lesser) than one without.

    The following are valid version numbers (shown in the order that
    would be obtained by sorting according to the supplied cmp function):

        0.4       0.4.0  (these two are equivalent)
        0.4.1
        0.5a1
        0.5b3
        0.5
        0.9.6
        1.0
        1.0.4a3
        1.0.4b1
        1.0.4

    The following are examples of invalid version numbers:

        1
        2.7.2.2
        1.3.a4
        1.3pl1
        1.3c4

    The rationale for this version numbering system will be explained
    in the distutils documentation.
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

(defn StrictVersion 
  "Version numbering for anal retentives and software idealists.
    Implements the standard interface for version number classes as
    described above.  A version number consists of two or three
    dot-separated numeric components, with an optional \"pre-release\" tag
    on the end.  The pre-release tag consists of the letter 'a' or 'b'
    followed by a number.  If the numeric components of two version
    numbers are equal, then one with a pre-release tag will always
    be deemed earlier (lesser) than one without.

    The following are valid version numbers (shown in the order that
    would be obtained by sorting according to the supplied cmp function):

        0.4       0.4.0  (these two are equivalent)
        0.4.1
        0.5a1
        0.5b3
        0.5
        0.9.6
        1.0
        1.0.4a3
        1.0.4b1
        1.0.4

    The following are examples of invalid version numbers:

        1
        2.7.2.2
        1.3.a4
        1.3pl1
        1.3c4

    The rationale for this version numbering system will be explained
    in the distutils documentation.
    "
  [ & {:keys [vstring]} ]
   (py/call-attr-kw tensorflow-backend "StrictVersion" [] {:vstring vstring }))

(defn parse 
  ""
  [ self vstring ]
  (py/call-attr self "parse"  self vstring ))
