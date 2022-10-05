# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
Provides json emitting and parsing functionality that can handle graphs.
A majority of the code is based on monty.json.
Credits: Materials Virtual Job
"""

import datetime
import json
import os
import pathlib
import types
from collections import OrderedDict, defaultdict
from enum import Enum
from importlib import import_module
from inspect import getfullargspec
from uuid import UUID
import numpy as np


try:
    import jax
    import jaxlib.xla_extension
except ImportError:
    jax = None

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

try:
    import bson
except ImportError:
    bson = None  # type: ignore

try:
    from ruamel.yaml import YAML
except ImportError:
    YAML = None  # type: ignore

try:
    import orjson
except ImportError:
    orjson = None  # type: ignore

__version__ = "3.0.0"


def _load_redirect(redirect_file):
    try:
        with open(redirect_file) as f:
            yaml = YAML()
            d = yaml.load(f)
    except OSError:
        # If we can't find the file
        # Just use an empty redirect dict
        return {}

    # Convert the full paths to module/class
    redirect_dict = defaultdict(dict)
    for old_path, new_path in d.items():
        old_class = old_path.split(".")[-1]
        old_module = ".".join(old_path.split(".")[:-1])

        new_class = new_path.split(".")[-1]
        new_module = ".".join(new_path.split(".")[:-1])

        redirect_dict[old_module][old_class] = {
            "@module": new_module,
            "@class": new_class,
        }

    return dict(redirect_dict)


class GSONable:
    """
    This is a mix-in base class specifying an API for GSONable objects. GSON
    is Graph JSON. This class aims to overcome the limitation of JSON in
    serializing/deserializing of objects whose compositional pattern resembles
    a direct acyclic graph (DAG). Essentially, GSONable objects must implement an as_dict
    method, which must return a json serializable dict and also a unique path
    within the json document plus a unique identifier,
    and a from_dict class method that regenerates the object from the dict
    generated by the as_dict method. The as_dict method should contain the
    "@module", "@class", and "@name" keys which will allow the GSONEncoder to
    dynamically deserialize the class. E.g.::
        d["@module"] = self.__class__.__module__
        d["@class"] = self.__class__.__name__
        d["@name"] = self.name
    The "@name" should provide a unique id to represent the instance.
    A default implementation is provided in GSONable, which automatically
    determines if the class already contains self.argname or self._argname
    attributes for every arg. If so, these will be used for serialization in
    the dict format. Similarly, the default from_dict will deserialization
    classes of such form. An example is given below::
        class GSONClass(GSONable):
        def __init__(self, a, b, c, d=1, **kwargs):
            self.a = a
            self.b = b
            self._c = c
            self._d = d
            self.kwargs = kwargs
    For such classes, you merely need to inherit from GSONable and you do not
    need to implement your own as_dict or from_dict protocol.
    GSONable objects need to have a `name` attribute that is unique.
    Classes can be redirected to moved implementations by putting in the old
    fully qualified path and new fully qualified path into .simsopt.yaml in the
    home folder
    Example:
    old_module.old_class: new_module.new_class
    """

    REDIRECT = _load_redirect(
        os.path.join(os.path.expanduser("~"), ".simsopt.yaml"))

    def as_dict(self, serial_objs_dict):
        """
        A JSON serializable dict representation of an object.
        """
        name = getattr(self, "name", str(id(self)))
        d = {"@module": self.__class__.__module__,
             "@class": self.__class__.__name__,
             "@name": name}

        try:
            parent_module = \
                self.__class__.__module__.split(".", maxsplit=1)[0]
            module_version = import_module(
                parent_module).__version__  # type: ignore
            d["@version"] = str(module_version)
        except (AttributeError, ImportError):
            d["@version"] = None  # type: ignore

        spec = getfullargspec(self.__class__.__init__)
        args = spec.args

        def recursive_as_dict(obj):
            if isinstance(obj, (list, tuple)):
                return [recursive_as_dict(it) for it in obj]
            if isinstance(obj, dict):
                return {kk: recursive_as_dict(vv) for kk, vv in obj.items()}
            if callable(obj) and not isinstance(obj, GSONable):
                return _serialize_callable(obj, serial_objs_dict=serial_objs_dict)
            if hasattr(obj, "as_dict"):
                name = getattr(obj, "name", str(id(obj)))
                if name not in serial_objs_dict:  # Add the path
                    serial_obj = obj.as_dict(serial_objs_dict)  # serial_objs is modified in place
                    serial_objs_dict[name] = serial_obj
                return {"$type": "ref", "value": name}
            return obj

        for c in args:
            if c != "self":
                try:
                    a = getattr(self, c)
                except AttributeError:
                    try:
                        a = getattr(self, "_" + c)
                    except AttributeError:
                        print(f"Missing attribute is {c}")
                        raise NotImplementedError(
                            "Unable to automatically determine as_dict "
                            "format from class. GSONAble requires all "
                            "args to be present as either self.argname or "
                            "self._argname, and kwargs to be present under"
                            "a self.kwargs variable to automatically "
                            "determine the dict format. Alternatively, "
                            "you can implement both as_dict and from_dict."
                        )
                d[c] = recursive_as_dict(a)
        if hasattr(self, "kwargs"):
            # type: ignore
            d.update(**getattr(self, "kwargs"))  # pylint: disable=E1101
        if spec.varargs is not None and getattr(self, spec.varargs,
                                                None) is not None:
            d.update({spec.varargs: getattr(self, spec.varargs)})
        if hasattr(self, "_kwargs"):
            d.update(**getattr(self, "_kwargs"))  # pylint: disable=E1101
        if isinstance(self, Enum):
            d.update({"value": self.value})  # pylint: disable=E1101
        return d  # , serial_objs_dict

    @classmethod
    def from_dict(cls, d, serial_objs_dict, recon_objs):
        """
        :param d: Dict representation.
        :return: GSONable class.
        """
        decoded = {k: GSONDecoder().process_decoded(v, serial_objs_dict, recon_objs) for k, v in
                   d.items() if not k.startswith("@")}
        return cls(**decoded)

    def to_json(self) -> str:
        """
        Returns a json string representation of the GSONable object.
        """
        return json.dumps(SIMSON(self), cls=GSONEncoder)

    @classmethod
    def __get_validators__(cls):
        """Return validators for use in pydantic"""
        yield cls.validate_gson

    @classmethod
    def validate_gson(cls, v):
        """
        pydantic Validator for GSONable pattern
        """
        if isinstance(v, cls):
            return v
        if isinstance(v, dict):
            new_obj = GSONDecoder().process_decoded(v)
            if isinstance(new_obj, cls):
                return new_obj

            new_obj = cls(**v)
            return new_obj

        raise ValueError(
            f"Must provide {cls.__name__}, the as_dict form, or the proper")

    @classmethod
    def __modify_schema__(cls, field_schema):
        """JSON schema for GSONable pattern"""
        field_schema.update(
            {
                "type": "object",
                "properties": {
                    "@class": {"enum": [cls.__name__], "type": "string"},
                    "@module": {"enum": [cls.__module__], "type": "string"},
                    "@version": {"type": "string"},
                },
                "required": ["@class", "@module"],
            }
        )


class SIMSON:
    """
    Wrapper class providing a scaffolding for serializing the graph
    framework implemented in simsopt. This class aims to overcome the
    limitation of JSON in serializing/deserializing of objects whose
    compositional pattern resembles a direct acyclic graph (DAG). This
    class is used to wrap the simsopt graph just before passing the simsopt
    graph to serializing functions. Essentially, SIMSON generates an extra
    dictionary that contains only serialized GSONable objects and the
    conventionally generated JSON doc contains only references (by the way of
    keys in the extra dictionary) for serialized GSONable objects.
    Only one instance should be used to enclose the entire graph.
    This class implements an as_dict method, which returns a json serializable
    dict with two subdicts with keys: "graph" and "simsopt_objs". The "graph"
    subdict consists of the json doc typically produced where all simsopt objs
    replaced by their references in the "simsopt_objs" subdict. "simsopt_objs"
    is a dict whose keys are the unique identifiers for simsopt objects and
    the values are the serialized simsopt objects.
    For deserialization a from_dict class method is implemented that
    regenerates the object from the dicts
    generated by the as_dict method. The as_dict method should contain the
    "@module" and "@class" keys which will allow the GSONEncoder to
    dynamically deserialize the class. E.g.::
        d["@module"] = self.__class__.__module__
        d["@class"] = self.__class__.__name__
    """

    def __init__(self, simsopt_objs):
        self.simsopt_objs = simsopt_objs

    def as_dict(self, serial_objs_dict=None):
        """
        A JSON serializable dict representation of an object.
        """
        d = {"@module": self.__class__.__module__,
             "@class": self.__class__.__name__}

        try:
            parent_module = \
                self.__class__.__module__.split(".", maxsplit=1)[0]
            module_version = import_module(
                parent_module).__version__  # type: ignore
            d["@version"] = str(module_version)
        except (AttributeError, ImportError):
            d["@version"] = None  # type: ignore

        serial_objs_dict = {}

        def recursive_as_dict(obj):
            if isinstance(obj, (list, tuple)):
                return [recursive_as_dict(it) for it in obj]
            if isinstance(obj, dict):
                return {kk: recursive_as_dict(vv) for kk, vv in obj.items()}
            if callable(obj) and not isinstance(obj, GSONable):
                return _serialize_callable(obj, serial_objs_dict=serial_objs_dict)
            if hasattr(obj, "as_dict"):
                name = getattr(obj, "name", str(id(obj)))
                if name not in serial_objs_dict:  # Add the path
                    serial_obj = obj.as_dict(
                        serial_objs_dict=serial_objs_dict)  # serial_objs is modified in place
                    serial_objs_dict[name] = serial_obj
                return {"$type": "ref", "value": name}
            return obj

        d["graph"] = recursive_as_dict(self.simsopt_objs)
        d["simsopt_objs"] = serial_objs_dict
        return d

    @classmethod
    def from_dict(cls, d, serial_objs_dict=None, recon_objs=None):
        graph_subdict = d["graph"]
        serial_objs_dict = d["simsopt_objs"]
        gson_decoder = GSONDecoder()
        recon_objs = {}
        return gson_decoder.process_decoded(
            graph_subdict, serial_objs_dict, recon_objs)


class GSONEncoder(json.JSONEncoder):
    """
    A Json Encoder which supports the GSONable API, plus adds support for
    numpy arrays, datetime objects, bson ObjectIds (requires bson).
    Usage::
        # Add it as a *cls* keyword when using json.dump
        json.dumps(object, cls=GSONEncoder)
    """

    def default(self, o) -> dict:  # pylint: disable=E0202
        """
        Overriding default method for JSON encoding. This method does two
        things: (a) If an object has a to_dict property, return the to_dict
        output. (b) If the @module and @class keys are not in the to_dict,
        add them to the output automatically. If the object has no to_dict
        property, the default Python json encoder default method is called.
        Args:
            o: Python object.
        Return:
            Python dict representation.
        """
        if isinstance(o, datetime.datetime):
            return {"@module": "datetime", "@class": "datetime",
                    "string": str(o)}
        if isinstance(o, UUID):
            return {"@module": "uuid", "@class": "UUID", "string": str(o)}

        if jax is not None and np is not None:
            if isinstance(o, jaxlib.xla_extension.DeviceArray):
                o = np.asarray(o)

        if np is not None:
            if isinstance(o, np.ndarray):
                if str(o.dtype).startswith("complex"):
                    return {
                        "@module": "numpy",
                        "@class": "array",
                        "dtype": str(o.dtype),
                        "data": [o.real.tolist(), o.imag.tolist()],
                    }
                return {
                    "@module": "numpy",
                    "@class": "array",
                    "dtype": str(o.dtype),
                    "data": o.tolist(),
                }
            if isinstance(o, np.generic):
                return o.item()

        if pd is not None:
            if isinstance(o, pd.DataFrame):
                return {
                    "@module": "pandas",
                    "@class": "DataFrame",
                    "data": o.to_json(
                        default_handler=GSONEncoder().encode),
                }
            if isinstance(o, pd.Series):
                return {
                    "@module": "pandas",
                    "@class": "Series",
                    "data": o.to_json(
                        default_handler=GSONEncoder().encode),
                }

        if bson is not None:
            if isinstance(o, bson.objectid.ObjectId):
                return {"@module": "bson.objectid", "@class": "ObjectId",
                        "oid": str(o)}

        if callable(o) and not isinstance(o, GSONable):
            return _serialize_callable(o)

        try:
            d = o.as_dict()
            if hasattr(o, "name") and "@name" not in d:
                d["@name"] = o.name

            if "@module" not in d:
                d["@module"] = str(o.__class__.__module__)
            if "@class" not in d:
                d["@class"] = str(o.__class__.__name__)
            if "@version" not in d:
                try:
                    parent_module = o.__class__.__module__.split(".")[0]
                    module_version = import_module(
                        parent_module).__version__  # type: ignore
                    d["@version"] = str(module_version)
                except (AttributeError, ImportError):
                    d["@version"] = None
            return d
        except AttributeError:
            return json.JSONEncoder.default(self, o)


class GSONDecoder(json.JSONDecoder):
    """
    A Json Decoder which supports the GSONable API. By default, the
    decoder attempts to find a module and name associated with a dict. If
    found, the decoder will generate a SIMSOPT object as a priority.  If that fails,
    the original decoded dictionary from the string is returned. Note that
    nested lists and dicts containing pymatgen object will be decoded correctly
    as well.
    Usage:
        # Add it as a *cls* keyword when using json.load
        json.loads(json_string, cls=GSONDecoder)
    """

    def process_decoded(self, d, serial_objs_dict=None, recon_objs=None):
        """
        Recursive method to support decoding dicts and lists containing
        GSONable objects.
        """
        if isinstance(d, dict):
            if "$type" in d.keys():
                if d["$type"] == "ref":
                    if d["value"] not in recon_objs:
                        sub_dict = serial_objs_dict[d["value"]]
                        recon_obj = self.process_decoded(sub_dict,
                                                         serial_objs_dict, recon_objs)
                        recon_objs[d["value"]] = recon_obj
                    return recon_objs[d["value"]]
            if "@module" in d and "@class" in d:
                modname = d["@module"]
                classname = d["@class"]
                if classname in GSONable.REDIRECT.get(modname, {}):
                    modname = GSONable.REDIRECT[modname][classname][
                        "@module"]
                    classname = GSONable.REDIRECT[modname][classname][
                        "@class"]
            elif "@module" in d and "@callable" in d:
                modname = d["@module"]
                objname = d["@callable"]
                classname = None
                if d.get("@bound", None) is not None:
                    # if the function is bound to an instance or class, first
                    # deserialize the bound object and then remove the object name
                    # from the function name.
                    obj = self.process_decoded(d["@bound"], serial_objs_dict=serial_objs_dict,
                                               recon_objs=recon_objs)
                    objname = objname.split(".")[1:]
                else:
                    # if the function is not bound to an object, import the
                    # function from the module name
                    obj = __import__(modname, globals(), locals(),
                                     [objname], 0)
                    objname = objname.split(".")
                try:
                    # the function could be nested. e.g., MyClass.NestedClass.function
                    # so iteratively access the nesting
                    for attr in objname:
                        obj = getattr(obj, attr)

                    return obj

                except AttributeError:
                    pass
            else:
                modname = None
                classname = None

            if classname:

                if modname and modname not in ["bson.objectid", "numpy",
                                               "pandas"]:
                    if modname == "datetime" and classname == "datetime":
                        try:
                            dt = datetime.datetime.strptime(d["string"],
                                                            "%Y-%m-%d %H:%M:%S.%f")
                        except ValueError:
                            dt = datetime.datetime.strptime(d["string"],
                                                            "%Y-%m-%d %H:%M:%S")
                        return dt

                    if modname == "uuid" and classname == "UUID":
                        return UUID(d["string"])

                    mod = __import__(modname, globals(), locals(),
                                     [classname], 0)
                    if hasattr(mod, classname):
                        cls_ = getattr(mod, classname)
                        data = {k: v for k, v in d.items() if
                                not k.startswith("@")}
                        if hasattr(cls_, "from_dict"):
                            obj = cls_.from_dict(data, serial_objs_dict, recon_objs)
                            if "@name" in d:
                                recon_objs[d["@name"]] = obj
                            return obj
                elif np is not None and modname == "numpy" and classname == "array":
                    if d["dtype"].startswith("complex"):
                        return np.array(
                            [np.array(r) + np.array(i) * 1j for r, i in
                             zip(*d["data"])],
                            dtype=d["dtype"],
                        )
                    return np.array(d["data"], dtype=d["dtype"])
                elif pd is not None and modname == "pandas":
                    if classname == "DataFrame":
                        decoded_data = GSONDecoder().decode(d["data"])
                        return pd.DataFrame(decoded_data)
                    if classname == "Series":
                        decoded_data = GSONDecoder().decode(d["data"])
                        return pd.Series(decoded_data)
                elif (
                        bson is not None) and modname == "bson.objectid" and classname == "ObjectId":
                    return bson.objectid.ObjectId(d["oid"])

            return {self.process_decoded(k, serial_objs_dict, recon_objs): self.process_decoded(v, serial_objs_dict, recon_objs) for
                    k, v in d.items()}

        if isinstance(d, list):
            return [self.process_decoded(x, serial_objs_dict, recon_objs) for x in d]

        return d

    def decode(self, s):
        """
        Overrides decode from JSONDecoder.
        :param s: string
        :return: Object.
        """
        if orjson is not None:
            try:
                d = orjson.loads(s)  # pylint: disable=E1101
            except orjson.JSONDecodeError:  # pylint: disable=E1101
                d = json.loads(s)
        else:
            d = json.loads(s)
        return self.process_decoded(d)


class GSONError(Exception):
    """
    Exception class for serialization errors.
    """


def jsanitize(obj, strict=False, allow_bson=False, enum_values=False,
              recursive_gsonable=False, serial_objs_dict=None):
    """
    This method cleans an input json-like object, either a list or a dict or
    some sequence, nested or otherwise, by converting all non-string
    dictionary keys (such as int and float) to strings, and also recursively
    encodes all objects using GSON's as_dict() protocol.
    Args:
        obj: input json-like object.
        strict (bool): This parameters sets the behavior when jsanitize
            encounters an object it does not understand. If strict is True,
            jsanitize will try to get the as_dict() attribute of the object. If
            no such attribute is found, an attribute error will be thrown. If
            strict is False, jsanitize will simply call str(object) to convert
            the object to a string representation.
        allow_bson (bool): This parameters sets the behavior when jsanitize
            encounters a bson supported type such as objectid and datetime. If
            True, such bson types will be ignored, allowing for proper
            insertion into MongoDB databases.
        enum_values (bool): Convert Enums to their values.
        recursive_gsonable (bool): If True, uses .as_dict() for GSONables regardless
            of the value of strict.
        serial_objs_dict: Dictionary of serialized objects produced during jsonification
    Returns:
        Sanitized dict that can be json serialized.
    """
    if serial_objs_dict is None:
        serial_objs_dict = {}
    if isinstance(obj, Enum) and enum_values:
        return obj.value

    if allow_bson and (
            isinstance(obj, (datetime.datetime, bytes)) or (
            bson is not None and isinstance(obj, bson.objectid.ObjectId))
    ):
        return obj
    if isinstance(obj, (list, tuple)):
        return [jsanitize(i, strict=strict, allow_bson=allow_bson,
                          enum_values=enum_values) for i in obj]
    if np is not None and isinstance(obj, np.ndarray):
        return [jsanitize(i, strict=strict, allow_bson=allow_bson,
                          enum_values=enum_values) for i in obj.tolist()]
    if np is not None and isinstance(obj, np.generic):
        return obj.item()
    if pd is not None and isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {
            str(k): jsanitize(
                v,
                strict=strict,
                allow_bson=allow_bson,
                enum_values=enum_values,
                recursive_gsonable=recursive_gsonable,
            )
            for k, v in obj.items()
        }
    if isinstance(obj, (int, float)):
        return obj
    if obj is None:
        return None
    if isinstance(obj, pathlib.Path):
        return str(obj)

    if callable(obj) and not isinstance(obj, GSONable):
        try:
            return _serialize_callable(obj)
        except TypeError:
            pass

    if recursive_gsonable and isinstance(obj, GSONable):
        return obj.as_dict(serial_objs_dict=serial_objs_dict)

    if not strict:
        return str(obj)

    if isinstance(obj, str):
        return obj

    return jsanitize(
        obj.as_dict(serial_objs_dict=serial_objs_dict),
        strict=strict,
        allow_bson=allow_bson,
        enum_values=enum_values,
        recursive_gsonable=recursive_gsonable,
    )


def _serialize_callable(o, serial_objs_dict={}):
    if isinstance(o, types.BuiltinFunctionType):
        # don't care about what builtin functions (sum, open, etc) are bound to
        bound = None
    else:
        # bound methods (i.e., instance methods) have a __self__ attribute
        # that points to the class/module/instance
        bound = getattr(o, "__self__", None)

    # we are only able to serialize bound methods if the object the method is
    # bound to is itself serializable
    if bound is not None:
        if isinstance(bound, GSONable):
            bound = bound.as_dict(serial_objs_dict=serial_objs_dict)
        else:
            try:
                bound = GSONEncoder().default(bound)
            except TypeError:
                raise TypeError(
                    "Only bound methods of classes or GSONable instances are supported.")

    return {
        "@module": o.__module__,
        "@callable": getattr(o, "__qualname__", o.__name__),
        "@bound": bound,
    }
