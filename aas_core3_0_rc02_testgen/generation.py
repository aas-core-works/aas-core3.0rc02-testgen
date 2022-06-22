"""Generate the intermediate representation of the test data."""
import collections
import contextlib
import hashlib
import json
import pathlib
import re
from typing import (
    OrderedDict,
    Union,
    List,
    Sequence,
    Any,
    MutableMapping,
    get_args,
    Optional,
    Callable,
    Tuple,
    Iterable,
    Iterator,
    Set,
)

import aas_core_codegen.common
import aas_core_meta.v3rc2
from aas_core_codegen import intermediate, infer_for_schema
from aas_core_codegen.common import Identifier
from icontract import require, ensure, DBC

from aas_core3_0_rc02_testgen import ontology
from aas_core3_0_rc02_testgen.frozen_examples import (
    pattern as frozen_examples_pattern,
    xs_value as frozen_examples_xs_value,
)

PrimitiveValueUnion = Union[bool, int, float, str, bytearray]

PrimitiveValueTuple = (bool, int, float, str, bytearray)
assert PrimitiveValueTuple == get_args(PrimitiveValueUnion)

ValueUnion = Union[PrimitiveValueUnion, "Instance", "ListOfInstances"]


class Instance:
    """Represent an instance of a class."""

    def __init__(
        self, properties: OrderedDict[str, ValueUnion], model_type: Identifier
    ) -> None:
        """
        Initialize with the given values.

        The ``model_type`` needs to be always indicated. Whether it is represented in
        the final serialization depends on the context of the serialization.

        The ``model_type`` corresponds to the class name in the meta-model, not to the
        class name in the respective serialization.
        """
        self.properties = properties
        self.model_type = model_type


class ListOfInstances:
    """Represent a list of instances."""

    def __init__(self, values: List[Instance]) -> None:
        """Initialize with the given values."""
        self.values = values


def _to_jsonable(value: ValueUnion) -> Any:
    """
    Represent the ``value`` as a JSON-able object.

    This is meant for debugging, not for the end-user serialization.
    """
    if isinstance(value, PrimitiveValueTuple):
        if isinstance(value, bytearray):
            return repr(value)
        else:
            return value
    elif isinstance(value, Instance):
        obj = collections.OrderedDict()  # type: MutableMapping[str, Any]
        obj["model_type"] = value.model_type

        properties_dict = collections.OrderedDict()  # type: MutableMapping[str, Any]
        for prop_name, prop_value in value.properties.items():
            properties_dict[prop_name] = _to_jsonable(prop_value)

        obj["properties"] = properties_dict

        return obj
    elif isinstance(value, ListOfInstances):
        return [_to_jsonable(item) for item in value.values]
    else:
        aas_core_codegen.common.assert_never(value)


def dump(value: ValueUnion) -> str:
    """
    Represent the ``value`` as a string.

    This is meant for debugging, not for the end-user serialization.
    """
    return json.dumps(_to_jsonable(value), indent=2)


@require(lambda environment: environment.model_type == "Environment")
def _dereference(
    environment: Instance, path_segments: Sequence[Union[int, str]]
) -> Instance:
    """Dereference the path to an instance starting from an environment."""
    cursor = environment  # type: Any
    for i, segment in enumerate(path_segments):
        if isinstance(segment, str):
            if not isinstance(cursor, Instance):
                raise AssertionError(
                    f"Expected the path {_posix_path(path_segments)} "
                    f"in the environment: {dump(environment)}; "
                    f"however, the cursor at the segment {i} "
                    f"does not point to an instance, but to: {dump(cursor)}"
                )

            if segment not in cursor.properties:
                raise AssertionError(
                    f"Expected the path {_posix_path(path_segments)} "
                    f"in the environment: {dump(environment)}; "
                    f"however, the segment {i + 1},{segment}, "
                    f"does not exist as a property "
                    f"in the instance: {dump(cursor)}"
                )

            cursor = cursor.properties[segment]

        elif isinstance(segment, int):
            if not isinstance(cursor, ListOfInstances):
                raise AssertionError(
                    f"Expected the path {_posix_path(path_segments)} "
                    f"in the environment: {dump(environment)}; "
                    f"however, the cursor at the segment {i} "
                    f"does not point to a list of instances, but to: {dump(cursor)}"
                )

            if segment >= len(cursor.values):
                raise AssertionError(
                    f"Expected the path {_posix_path(path_segments)} "
                    f"in the environment: {dump(environment)}; "
                    f"however, the segment {i + 1}, {segment}, "
                    f"does not exist as an item "
                    f"in the list of instances: {dump(cursor)}"
                )

            cursor = cursor.values[segment]
        else:
            aas_core_codegen.common.assert_never(segment)

    if not isinstance(cursor, Instance):
        raise AssertionError(
            f"Expected the path {_posix_path(path_segments)} "
            f"in the environment: {json.dumps(environment, indent=2)} "
            f"to _dereference an instance, but got: {dump(cursor)}"
        )

    return cursor


def _deep_copy(value: ValueUnion) -> ValueUnion:
    """Make a deep copy of the given value."""
    if isinstance(value, PrimitiveValueTuple):
        return value
    elif isinstance(value, Instance):
        props = collections.OrderedDict()  # type: OrderedDict[str, ValueUnion]
        for prop_name, prop_value in value.properties.items():
            props[prop_name] = _deep_copy(prop_value)

        return Instance(properties=props, model_type=value.model_type)

    elif isinstance(value, ListOfInstances):
        values = []  # type: List[Instance]
        for item in value.values:
            a_copy = _deep_copy(item)
            assert isinstance(a_copy, Instance)
            values.append(a_copy)

        return ListOfInstances(values=values)
    else:
        aas_core_codegen.common.assert_never(value)


# noinspection RegExpSimplifiable
_HEX_RE = re.compile(r"[a-fA-F0-9]+")


@ensure(lambda result: _HEX_RE.fullmatch(result))
def _hash_path(path_segments: Sequence[Union[str, int]]) -> str:
    """Hash the given path to a value in the model."""
    hsh = hashlib.md5()
    hsh.update(("".join(repr(segment) for segment in path_segments)).encode("utf-8"))
    return hsh.hexdigest()[:8]


def _posix_path(path_segments: Sequence[Union[str, int]]) -> pathlib.PurePosixPath:
    """Make a POSIX path out of the path segments."""
    pth = pathlib.PurePosixPath("/")
    for segment in path_segments:
        pth = pth / str(segment)

    return pth


@require(lambda length: length > 0)
@ensure(lambda result, length: len(result) == length)
def _generate_long_string(
    length: int,
    path_segments: List[Union[int, str]],
) -> str:
    """
    Generate a string longer than the ``length``.

    >>> _generate_long_string(2, ['some', 3, 'path'])
    'x9'

    >>> _generate_long_string(9, ['some', 3, 'path'])
    'x99ef1573'

    >>> _generate_long_string(10, ['some', 3, 'path'])
    'x99ef15730'

    >>> _generate_long_string(15, ['some', 3, 'path'])
    'x99ef1573012345'

    >>> _generate_long_string(20, ['some', 3, 'path'])
    'x99ef157301234567890'

    >>> _generate_long_string(21, ['some', 3, 'path'])
    'x99ef1573012345678901'
    """
    prefix = f"x{_hash_path(path_segments=path_segments)}"
    if len(prefix) > length:
        return prefix[:length]

    ruler = "1234567890"

    if length <= 10:
        return prefix + ruler[len(prefix) : length]

    tens = length // 10
    remainder = length % 10
    return "".join(
        [prefix, ruler[len(prefix) : 10], ruler * (tens - 1), ruler[:remainder]]
    )


def _generate_time_of_day(path_segments: List[Union[int, str]]) -> str:
    """Generate a random time of the day based on the path to the value."""
    hsh = _hash_path(path_segments=path_segments)

    hsh_as_int = int(hsh, base=16)

    remainder = hsh_as_int
    hours = (remainder // 3600) % 24
    remainder = remainder % 3600
    minutes = (remainder // 60) % 60
    seconds = remainder % 60

    fraction = hsh_as_int % 1000000

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{fraction}"


@require(
    lambda type_annotation: (
        type_anno := intermediate.beneath_optional(type_annotation),
        isinstance(type_anno, intermediate.PrimitiveTypeAnnotation)
        or (
            isinstance(type_anno, intermediate.OurTypeAnnotation)
            and isinstance(type_anno.symbol, intermediate.ConstrainedPrimitive)
        ),
    )[1]
)
def _generate_primitive_value(
    type_annotation: intermediate.TypeAnnotationExceptOptional,
    path_segments: List[Union[str, int]],
    len_constraint: Optional[infer_for_schema.LenConstraint],
    pattern_constraints: Optional[Sequence[infer_for_schema.PatternConstraint]],
) -> PrimitiveValueUnion:
    """Generate the primitive value based on the ``path_segments``."""
    # noinspection PyUnusedLocal
    primitive_type = None  # type: Optional[intermediate.PrimitiveType]

    if isinstance(type_annotation, intermediate.PrimitiveTypeAnnotation):
        primitive_type = type_annotation.a_type
    elif isinstance(type_annotation, intermediate.OurTypeAnnotation) and isinstance(
        type_annotation.symbol, intermediate.ConstrainedPrimitive
    ):
        primitive_type = type_annotation.symbol.constrainee
    else:
        raise AssertionError(f"Unexpected {type(type_annotation)}: {type_annotation=}")

    assert primitive_type is not None

    def implementation() -> Union[bool, int, float, str, bytearray]:
        """Wrap the body so that we can ensure the len constraints."""
        hsh = _hash_path(path_segments=path_segments)

        # region Handle the special case of a single pattern constraint first

        if pattern_constraints is not None:
            if len(pattern_constraints) > 1:
                patterns = [
                    pattern_constraint.pattern
                    for pattern_constraint in pattern_constraints
                ]
                raise NotImplementedError(
                    "We did not implement the generation of a value based on two or "
                    "more pattern constraints, which is the case "
                    f"for the value {_posix_path(path_segments)}: {patterns}"
                )

            if primitive_type is not intermediate.PrimitiveType.STR:
                raise NotImplementedError(
                    "We did not implement the generation of a non-string value with "
                    "the pattern constraint, which is the case "
                    f"for the value {_posix_path(path_segments)}"
                )

            assert primitive_type is intermediate.PrimitiveType.STR

            assert len(pattern_constraints) > 0, "Unexpected empty pattern constraints"

            pattern = pattern_constraints[0].pattern
            pattern_examples = frozen_examples_pattern.BY_PATTERN.get(pattern, None)
            if pattern_examples is None:
                raise NotImplementedError(
                    f"The entry is missing "
                    f"in the {frozen_examples_pattern.__name__!r} "
                    f"for the pattern {pattern!r} "
                    f"when generating the value at {_posix_path(path_segments)}"
                )

            if len(pattern_examples.positives) == 0:
                raise NotImplementedError(
                    f"Unexpected an empty list of positive examples "
                    f"in the {frozen_examples_pattern.__name__!r} "
                    f"for the pattern {pattern!r} "
                    f"when generating the value at {_posix_path(path_segments)}"
                )

            for value in pattern_examples.positives.values():
                return value

            raise AssertionError("Expected to check for at least one positive example")

        # endregion

        hsh_as_int = int(hsh, base=16)

        assert primitive_type is not None
        if primitive_type is intermediate.PrimitiveType.BOOL:
            return hsh_as_int % 2 == 0

        elif primitive_type is intermediate.PrimitiveType.INT:
            return hsh_as_int

        elif primitive_type is intermediate.PrimitiveType.FLOAT:
            return float(hsh_as_int) / 100

        elif primitive_type is intermediate.PrimitiveType.STR:
            return f"something_random_{hsh}"

        elif primitive_type is intermediate.PrimitiveType.BYTEARRAY:
            return bytearray.fromhex(hsh)
        else:
            aas_core_codegen.common.assert_never(primitive_type)

    # NOTE (mristin, 2022-05-11):
    # We ensure here that the constraint on ``len(.)`` of the result is satisfied.
    # This covers some potential errors, but mind that still does not check
    # the constraints. Hence, you have to manually inspect the generated data and
    # decide yourself whether you need to write a generator manually.

    result = implementation()

    if len_constraint is not None:
        if primitive_type in (
            intermediate.PrimitiveType.BOOL,
            intermediate.PrimitiveType.INT,
            intermediate.PrimitiveType.FLOAT,
        ):
            raise ValueError(
                f"We do not know how to apply the length constraint "
                f"on the primitive type: {primitive_type.value}; path: {path_segments}"
            )

        assert isinstance(result, (str, bytearray))

        if (
            len_constraint.min_value is not None
            and len(result) < len_constraint.min_value
        ) or (
            len_constraint.max_value is not None
            and len(result) > len_constraint.max_value
        ):
            raise ValueError(
                f"Expected the value {_posix_path(path_segments)} "
                f"to satisfy the length constraint "
                f"[{len_constraint.min_value!r}, {len_constraint.max_value!r}], "
                f"but got the length {len(result)}. You have to write the generator "
                f"for this value instance yourself"
            )

    return result


@contextlib.contextmanager
def _extend_in_place(
    path_segments: List[Union[str, int]], extension: Iterable[Union[str, int]]
) -> Iterator[Any]:
    """Extend the ``path_segments`` with the ``extension`` and revert it on exit."""
    path_segments.extend(extension)
    try:
        yield
    finally:
        for _ in extension:
            path_segments.pop()


def _generate_model_reference(
    expected_type: aas_core_meta.v3rc2.Key_types,
    path_segments: List[Union[str, int]],
) -> Instance:
    """Generate a model Reference pointing to an instance of ``expected_type``."""
    props = collections.OrderedDict()  # type: OrderedDict[str, Any]
    props["type"] = aas_core_meta.v3rc2.Reference_types.Model_reference.value

    if expected_type in (
        aas_core_meta.v3rc2.Key_types.Asset_administration_shell,
        aas_core_meta.v3rc2.Key_types.Concept_description,
        aas_core_meta.v3rc2.Key_types.Submodel,
    ):
        with _extend_in_place(path_segments, ["keys", 0, "value"]):
            props["keys"] = ListOfInstances(
                values=[
                    Instance(
                        properties=collections.OrderedDict(
                            [
                                ("type", expected_type.value),
                                ("value", _hash_path(path_segments)),
                            ]
                        ),
                        model_type=Identifier("Key"),
                    )
                ]
            )

    elif expected_type is aas_core_meta.v3rc2.Key_types.Referable:
        with _extend_in_place(path_segments, ["keys", 0, "value"]):
            key0 = Instance(
                properties=collections.OrderedDict(
                    [
                        ("type", aas_core_meta.v3rc2.Key_types.Submodel.value),
                        ("value", f"something_random_{_hash_path(path_segments)}"),
                    ]
                ),
                model_type=Identifier("Key"),
            )

        with _extend_in_place(path_segments, ["keys", 1, "value"]):
            key1 = Instance(
                properties=collections.OrderedDict(
                    [
                        ("type", aas_core_meta.v3rc2.Key_types.Referable.value),
                        ("value", f"something_random_{_hash_path(path_segments)}"),
                    ]
                ),
                model_type=Identifier("Key"),
            )

        props["keys"] = ListOfInstances(values=[key0, key1])
    else:
        raise NotImplementedError(
            f"Unhandled {expected_type=}; when we developed this script there were "
            f"no other key types expected in the meta-model as a reference, "
            f"but this has obvious changed. Please contact the developers."
        )

    return Instance(properties=props, model_type=Identifier("Reference"))


def _generate_global_reference(
    path_segments: List[Union[str, int]],
) -> Instance:
    """Generate an instance of a global Reference."""

    props = collections.OrderedDict()  # type: OrderedDict[str, ValueUnion]
    props["type"] = aas_core_meta.v3rc2.Reference_types.Global_reference.value

    with _extend_in_place(path_segments, ["keys", 0, "value"]):
        key = Instance(
            properties=collections.OrderedDict(
                [
                    ("type", "GlobalReference"),
                    ("value", f"something_random_{_hash_path(path_segments)}"),
                ]
            ),
            model_type=Identifier("Key"),
        )
        props["keys"] = ListOfInstances(values=[key])

    return Instance(properties=props, model_type=Identifier("Reference"))


def _generate_property_value(
    type_annotation: intermediate.TypeAnnotationExceptOptional,
    path_segments: List[Union[str, int]],
    len_constraint: Optional[infer_for_schema.LenConstraint],
    pattern_constraints: Optional[Sequence[infer_for_schema.PatternConstraint]],
    generate_instance: Callable[
        [intermediate.ClassUnion, List[Union[str, int]]], Instance
    ],
) -> ValueUnion:
    """
    Generate the value for the given property.

    Since ``path_segments`` are extended in-place, this function is not thread-safe.

    The callable ``generate_instance`` instructs how to generate the instances
    recursively.
    """
    if isinstance(type_annotation, intermediate.PrimitiveTypeAnnotation) or (
        isinstance(type_annotation, intermediate.OurTypeAnnotation)
        and isinstance(type_annotation.symbol, intermediate.ConstrainedPrimitive)
    ):
        return _generate_primitive_value(
            type_annotation=type_annotation,
            path_segments=path_segments,
            len_constraint=len_constraint,
            pattern_constraints=pattern_constraints,
        )
    elif isinstance(type_annotation, intermediate.OurTypeAnnotation):
        if pattern_constraints is not None:
            raise ValueError(
                f"Unexpected pattern constraints for a value "
                f"of type {type_annotation} at {_posix_path(path_segments)}"
            )

        if len_constraint is not None:
            raise ValueError(
                f"Unexpected len constraint for a value "
                f"of type {type_annotation} at {_posix_path(path_segments)}"
            )

        if isinstance(type_annotation.symbol, intermediate.Enumeration):
            hsh_as_int = int(_hash_path(path_segments=path_segments), base=16)

            text = type_annotation.symbol.literals[
                hsh_as_int % len(type_annotation.symbol.literals)
            ].value

            return text

        elif isinstance(type_annotation.symbol, intermediate.ConstrainedPrimitive):
            raise AssertionError(
                f"Should have been handled before: {type_annotation.symbol}"
            )

        elif isinstance(
            type_annotation.symbol,
            (intermediate.AbstractClass, intermediate.ConcreteClass),
        ):
            return generate_instance(type_annotation.symbol, path_segments)
        else:
            aas_core_codegen.common.assert_never(type_annotation.symbol)

    elif isinstance(type_annotation, intermediate.ListTypeAnnotation):
        if pattern_constraints is not None:
            raise ValueError(
                f"Unexpected pattern constraints for a value "
                f"of type {type_annotation} at {_posix_path(path_segments)}"
            )

        if not isinstance(
            type_annotation.items, intermediate.OurTypeAnnotation
        ) or not isinstance(
            type_annotation.items.symbol,
            (intermediate.AbstractClass, intermediate.ConcreteClass),
        ):
            raise NotImplementedError(
                f"Implemented only handling lists of classes, "
                f"but got: {type_annotation}; please contact the developers"
            )

        with _extend_in_place(path_segments, [0]):
            instance = generate_instance(type_annotation.items.symbol, path_segments)

        result = ListOfInstances(values=[instance])

        if len_constraint is not None:
            if (
                len_constraint.min_value is not None
                and len(result.values) < len_constraint.min_value
            ) or (
                len_constraint.max_value is not None
                and len(result.values) > len_constraint.max_value
            ):
                raise ValueError(
                    f"Expected the value {_posix_path(path_segments)} "
                    f"to satisfy the len constraint "
                    f"[{len_constraint.min_value!r}, {len_constraint.max_value!r}], "
                    f"but got the list of length {len(result.values)}. "
                    f"You have to write the generator for this value yourself."
                )

        return result
    else:
        aas_core_codegen.common.assert_never(type_annotation)


def _generate_concrete_minimal_instance(
    cls: intermediate.ConcreteClass,
    path_segments: List[Union[str, int]],
    constraints_by_class: MutableMapping[
        intermediate.ClassUnion, infer_for_schema.ConstraintsByProperty
    ],
    symbol_table: intermediate.SymbolTable,
) -> Instance:
    """
    Generate an instance with only required properties of exactly type ``cls``.

    The ``path_segments`` refer to the path leading to the instance of the ``cls``.

    We recursively generate minimal instances for all the nested classes.
    We will re-use the ``path_segments`` in the subsequent recursive calls to avoid
    the quadratic time complexity, so beware that this function is *NOT* thread-safe.

    The generation is deterministic, *i.e.*, re-generating with the same input
    should yield the same output.
    """
    reference_cls = symbol_table.must_find(Identifier("Reference"))
    if cls is reference_cls:
        # NOTE (mristin, 2022-06-19):
        # We generate a global reference by default, since this makes for much better
        # examples with less confusion for the reader. If you need something else, fix
        # it afterwards.
        return _generate_global_reference(path_segments=path_segments)

    constraints_by_prop = constraints_by_class[cls]

    props = collections.OrderedDict()  # type: OrderedDict[str, ValueUnion]

    def generate_instance(
        a_cls: intermediate.ClassUnion, a_path_segments: List[Union[str, int]]
    ) -> Instance:
        """Generate an instance passing over the parameters from the closure."""
        return _generate_minimal_instance(
            cls=a_cls,
            path_segments=a_path_segments,
            constraints_by_class=constraints_by_class,
            symbol_table=symbol_table,
        )

    for prop in cls.properties:
        if isinstance(prop.type_annotation, intermediate.OptionalTypeAnnotation):
            continue

        with _extend_in_place(path_segments, [prop.name]):
            props[prop.name] = _generate_property_value(
                type_annotation=prop.type_annotation,
                path_segments=path_segments,
                len_constraint=constraints_by_prop.len_constraints_by_property.get(
                    prop, None
                ),
                pattern_constraints=constraints_by_prop.patterns_by_property.get(
                    prop, None
                ),
                generate_instance=generate_instance,
            )

    return Instance(properties=props, model_type=cls.name)

    # endregion


def _generate_minimal_instance(
    cls: intermediate.ClassUnion,
    path_segments: List[Union[str, int]],
    constraints_by_class: MutableMapping[
        intermediate.ClassUnion, infer_for_schema.ConstraintsByProperty
    ],
    symbol_table: intermediate.SymbolTable,
) -> Instance:
    """
    Generate an instance with only required properties of type ``cls``.

    The ``path_segments`` refer to the path leading to the instance of the ``cls``.

    If the ``cls`` is abstract or has concrete descendants, we arbitrarily pick one
    of the concrete descending classes or the ``cls`` itself, if it is concrete.

    We recursively generate minimal instances for all the nested classes.
    We will re-use the ``path_segments`` in the subsequent recursive calls to avoid
    the quadratic time complexity, so beware that this function is *NOT* thread-safe.

    The generation is deterministic, *i.e.*, re-generating with the same input
    should yield the same output.
    """
    if cls.interface is not None:
        hsh_as_int = int(_hash_path(path_segments=path_segments), base=16)

        concrete_classes = cls.interface.implementers
        concrete_cls = concrete_classes[hsh_as_int % len(concrete_classes)]
    else:
        assert isinstance(cls, intermediate.ConcreteClass)
        concrete_cls = cls

    return _generate_concrete_minimal_instance(
        cls=concrete_cls,
        path_segments=path_segments,
        constraints_by_class=constraints_by_class,
        symbol_table=symbol_table,
    )


class _Handyman:
    """
    Fix the instances recursively in-place so that the constraints are preserved.

    We assume that it is easier to fix the instances after the generation than to
    generate them correctly in the first pass.
    """

    def __init__(self, symbol_table: intermediate.SymbolTable) -> None:
        """Initialize with the given values."""
        self.symbol_table = symbol_table

        self._dispatch_concrete = {
            "Asset_administration_shell": _Handyman._fix_asset_administration_shell,
            "Basic_event_element": _Handyman._fix_basic_event_element,
            "Concept_description": _Handyman._fix_concept_description,
            "Entity": _Handyman._fix_entity,
            "Extension": _Handyman._fix_extension,
            "Property": _Handyman._fix_property,
            "Qualifier": _Handyman._fix_qualifier,
            "Range": _Handyman._fix_range,
            "Submodel": _Handyman._fix_submodel,
            "Submodel_element_collection": _Handyman._fix_submodel_element_collection,
            "Submodel_element_list": _Handyman._fix_submodel_element_list,
        }

        # region Ensure that all the dispatches has been properly defined
        inverted_dispatch = set(
            method.__name__ for method in self._dispatch_concrete.values()
        )

        for attr_name in dir(_Handyman):
            if attr_name.startswith("_fix_") and attr_name not in inverted_dispatch:
                raise AssertionError(
                    f"The method {attr_name} is missing in the dispatch set."
                )
        # endregion

        # region Ensure that the dispatch map is correct
        for cls_name in self._dispatch_concrete:
            cls = self.symbol_table.must_find(name=Identifier(cls_name))
            assert isinstance(cls, intermediate.ConcreteClass)
        # endregion

        # region Ensure that the dispatch methods are appropriate

        for cls_name, method in self._dispatch_concrete.items():
            method_stem = re.sub(r"^_fix_", "", method.__name__)
            assert (
                cls_name.lower() == method_stem.lower()
            ), f"{cls_name=}, {method_stem=}, {method.__name__=}"

        # endregion

    def fix_instance(
        self, instance: Instance, path_segments: List[Union[str, int]]
    ) -> None:
        """Fix the ``instance`` recursively in-place."""
        cls = self.symbol_table.must_find(name=instance.model_type)
        assert isinstance(cls, intermediate.ConcreteClass)

        # region Fix for the ancestor classes

        data_element_cls = self.symbol_table.must_find(Identifier("Data_element"))
        assert isinstance(data_element_cls, intermediate.AbstractClass)

        if cls.is_subclass_of(data_element_cls):
            category_value = instance.properties.get("category", None)
            if category_value is not None and category_value not in (
                "CONSTANT",
                "PARAMETER",
                "VARIABLE",
            ):
                instance.properties["category"] = "CONSTANT"

        # endregion

        # region Fix for the concrete class

        dispatch = self._dispatch_concrete.get(instance.model_type, None)
        if dispatch is not None:
            # noinspection PyArgumentList
            dispatch(self, instance, path_segments)
        else:
            self._recurse_into_properties(
                instance=instance, path_segments=path_segments
            )

    def fix_list_of_instances(
        self, list_of_instances: ListOfInstances, path_segments: List[Union[str, int]]
    ) -> None:
        """Fix the instances recursively in-place."""
        for i, instance in enumerate(list_of_instances.values):
            with _extend_in_place(path_segments, [i]):
                self.fix_instance(instance=instance, path_segments=path_segments)

    def _recurse_into_properties(
        self, instance: Instance, path_segments: List[Union[str, int]]
    ) -> None:
        """Fix the properties of the ``instance`` recursively in-place."""
        for prop_name, prop_value in instance.properties.items():
            if isinstance(prop_value, PrimitiveValueTuple):
                # NOTE (mristin, 2022-06-20):
                # There is nothing to recurse into primitive properties.
                pass

            elif isinstance(prop_value, Instance):
                with _extend_in_place(path_segments, [prop_name]):
                    self.fix_instance(prop_value, path_segments)

            elif isinstance(prop_value, ListOfInstances):
                with _extend_in_place(path_segments, [prop_name]):
                    self.fix_list_of_instances(prop_value, path_segments)

            else:
                aas_core_codegen.common.assert_never(prop_value)

    def _fix_basic_event_element(
        self, instance: Instance, path_segments: List[Union[str, int]]
    ) -> None:
        # Fix that the observed is a proper model reference
        if "observed" in instance.properties:
            with _extend_in_place(path_segments, ["observed"]):
                instance.properties["observed"] = _generate_model_reference(
                    expected_type=aas_core_meta.v3rc2.Key_types.Referable,
                    path_segments=path_segments,
                )

        # Override that the direction is output so that we can always set
        # the max interval
        if "direction" in instance.properties:
            direction_enum = self.symbol_table.must_find(name=Identifier("Direction"))
            assert isinstance(direction_enum, intermediate.Enumeration)

            instance.properties["direction"] = direction_enum.literals_by_name[
                "Output"
            ].value

        # Fix that the message broker is a proper model reference
        if "message_broker" in instance.properties:
            with _extend_in_place(path_segments, ["message_broker"]):
                instance.properties["message_broker"] = _generate_model_reference(
                    expected_type=aas_core_meta.v3rc2.Key_types.Referable,
                    path_segments=path_segments,
                )

        self._recurse_into_properties(instance=instance, path_segments=path_segments)

    def _fix_asset_administration_shell(
        self, instance: Instance, path_segments: List[Union[str, int]]
    ) -> None:
        # Fix the invariant that the derivedFrom is a reference to a shell
        if "derived_from" in instance.properties:
            with _extend_in_place(path_segments, ["derived_from"]):
                instance.properties["derived_from"] = _generate_model_reference(
                    expected_type=(
                        aas_core_meta.v3rc2.Key_types.Asset_administration_shell
                    ),
                    path_segments=path_segments,
                )

        # Fix the submodels to be proper model references
        if "submodels" in instance.properties:
            with _extend_in_place(path_segments, ["submodels", 0]):
                instance.properties["submodels"] = ListOfInstances(
                    values=[
                        _generate_model_reference(
                            expected_type=aas_core_meta.v3rc2.Key_types.Submodel,
                            path_segments=path_segments,
                        )
                    ]
                )

        self._recurse_into_properties(instance=instance, path_segments=path_segments)

    def _fix_concept_description(
        self, instance: Instance, path_segments: List[Union[str, int]]
    ) -> None:
        if "category" in instance.properties:
            instance.properties["category"] = "VALUE"

        self._recurse_into_properties(instance=instance, path_segments=path_segments)

    def _fix_entity(
        self, instance: Instance, path_segments: List[Union[str, int]]
    ) -> None:
        entity_type = instance.properties.get("entity_type", None)
        if entity_type is not None:
            entity_type_enum = self.symbol_table.must_find(Identifier("Entity_type"))
            assert isinstance(entity_type_enum, intermediate.Enumeration)

            self_managed_entity_literal = entity_type_enum.literals_by_name[
                "Self_managed_entity"
            ]

            if entity_type == self_managed_entity_literal.value:
                instance.properties.pop("specific_asset_id", None)
            else:
                instance.properties.pop("specific_asset_id", None)
                instance.properties.pop("global_asset_id", None)

        self._recurse_into_properties(instance=instance, path_segments=path_segments)

    def _fix_extension(
        self, instance: Instance, path_segments: List[Union[str, int]]
    ) -> None:
        extension_cls = self.symbol_table.must_find(Identifier("Extension"))
        assert isinstance(extension_cls, intermediate.ConcreteClass)

        # NOTE (mristin, 2022-06-20):
        # We need to assert this as we are automatically setting the ``value_type``.
        assert not isinstance(
            extension_cls.properties_by_name[Identifier("value_type")],
            intermediate.OptionalTypeAnnotation,
        )

        instance.properties["value_type"] = "xs:boolean"
        if "value" in instance.properties:
            instance.properties["value"] = "true"

        self._recurse_into_properties(instance=instance, path_segments=path_segments)

    def _fix_property(
        self, instance: Instance, path_segments: List[Union[str, int]]
    ) -> None:
        property_cls = self.symbol_table.must_find(Identifier("Property"))
        assert isinstance(property_cls, intermediate.ConcreteClass)

        # NOTE (mristin, 2022-06-20):
        # We need to assert this as we are automatically setting the ``value_type``.
        assert not isinstance(
            property_cls.properties_by_name[Identifier("value_type")],
            intermediate.OptionalTypeAnnotation,
        )

        instance.properties["value_type"] = "xs:boolean"
        if "value" in instance.properties:
            instance.properties["value"] = "true"

        self._recurse_into_properties(instance=instance, path_segments=path_segments)

    def _fix_qualifier(
        self, instance: Instance, path_segments: List[Union[str, int]]
    ) -> None:
        qualifier_cls = self.symbol_table.must_find(Identifier("Qualifier"))
        assert isinstance(qualifier_cls, intermediate.ConcreteClass)

        # NOTE (mristin, 2022-06-20):
        # We need to assert this as we are automatically setting the ``value_type``.
        assert not isinstance(
            qualifier_cls.properties_by_name[Identifier("value_type")],
            intermediate.OptionalTypeAnnotation,
        )

        instance.properties["value_type"] = "xs:boolean"
        if "value" in instance.properties:
            instance.properties["value"] = "true"

        self._recurse_into_properties(instance=instance, path_segments=path_segments)

    def _fix_range(
        self, instance: Instance, path_segments: List[Union[str, int]]
    ) -> None:
        range_cls = self.symbol_table.must_find(Identifier("Range"))
        assert isinstance(range_cls, intermediate.ConcreteClass)

        # NOTE (mristin, 2022-06-20):
        # We need to assert this as we are automatically setting the ``value_type``.
        assert not isinstance(
            range_cls.properties_by_name[Identifier("value_type")],
            intermediate.OptionalTypeAnnotation,
        )

        instance.properties["value_type"] = "xs:int"
        if "min" in instance.properties:
            instance.properties["min"] = "1234"

        if "max" in instance.properties:
            instance.properties["max"] = "4321"

        self._recurse_into_properties(instance=instance, path_segments=path_segments)

    def _fix_submodel(
        self, instance: Instance, path_segments: List[Union[str, int]]
    ) -> None:
        submodel_elements = instance.properties.get("submodel_elements", None)
        if submodel_elements is not None:
            assert isinstance(submodel_elements, ListOfInstances)

            for i, submodel_element in enumerate(submodel_elements.values):
                # NOTE (mristin, 2022-06-20):
                # ID-shorts are mandatory, so we always override them, regardless if
                # they existed or not.
                with _extend_in_place(path_segments, ["submodel_elements", i]):
                    submodel_element.properties[
                        "id_short"
                    ] = f"some_id_short_{_hash_path(path_segments)}"

        # region Fix qualifiers for the constraint AASd-119

        qualifier_kind_enum = self.symbol_table.must_find(Identifier("Qualifier_kind"))
        assert isinstance(qualifier_kind_enum, intermediate.Enumeration)

        qualifier_kind_template_qualifier = qualifier_kind_enum.literals_by_name[
            "Template_qualifier"
        ].value

        qualifiers = instance.properties.get("qualifiers", None)
        if qualifiers is not None:
            must_be_modeling_kind_template = False

            assert isinstance(qualifiers, ListOfInstances)
            for qualifier in qualifiers.values:
                if (
                    qualifier.properties.get("kind", None)
                    == qualifier_kind_template_qualifier
                ):
                    must_be_modeling_kind_template = True
                    break

            if must_be_modeling_kind_template:
                modeling_kind_enum = self.symbol_table.must_find(
                    Identifier("Modeling_kind")
                )

                assert isinstance(modeling_kind_enum, intermediate.Enumeration)

                modeling_kind_template = modeling_kind_enum.literals_by_name[
                    "Template"
                ].value

                instance.properties["kind"] = modeling_kind_template

        # endregion

        self._recurse_into_properties(instance=instance, path_segments=path_segments)

    def _fix_submodel_element_collection(
        self, instance: Instance, path_segments: List[Union[str, int]]
    ) -> None:
        # Fix that ID-shorts are always defined for the items of a submodel element
        # collection
        value = instance.properties.get("value", None)
        if value is not None:
            assert isinstance(value, ListOfInstances)

            for item in value.values:
                if "id_short" not in item.properties:
                    with _extend_in_place(path_segments, ["id_short"]):
                        hsh = _hash_path(path_segments=path_segments)
                        item.properties["id_short"] = f"something_random_{hsh}"

        self._recurse_into_properties(instance=instance, path_segments=path_segments)

    def _fix_submodel_element_list(
        self, instance: Instance, path_segments: List[Union[str, int]]
    ) -> None:
        # Fix that ID-shorts are not defined for the items of a submodel element list
        value = instance.properties.get("value", None)
        if value is not None:
            assert isinstance(value, ListOfInstances)

            for item in value.values:
                if "id_short" in item.properties:
                    del item.properties["id_short"]

        self._recurse_into_properties(instance=instance, path_segments=path_segments)


@ensure(lambda result: result[0].model_type == "Environment")
def _generate_minimal_instance_in_minimal_environment(
    cls: intermediate.ConcreteClass,
    class_graph: ontology.ClassGraph,
    constraints_by_class: MutableMapping[
        intermediate.ClassUnion, infer_for_schema.ConstraintsByProperty
    ],
    symbol_table: intermediate.SymbolTable,
) -> Tuple[Instance, List[Union[str, int]]]:
    """
    Generate the minimal instance of ``cls`` in a minimal environment instance.

    The environment needs to be fixed after the generation. Use :class:`~Handyman`.

    Return the environment and the path to the instance.
    """
    shortest_path_in_class_graph_from_environment = class_graph.shortest_paths[cls.name]

    environment_instance: Optional[Instance] = None

    path_segments: List[Union[str, int]] = []
    source_instance: Optional[Instance] = None

    instance_path = None  # type: Optional[List[Union[int, str]]]

    for i, edge in enumerate(shortest_path_in_class_graph_from_environment):
        if source_instance is None:
            assert edge.source.name == "Environment", (
                "Expected the generation to start from an instance "
                "of the class 'Environment'"
            )
            source_instance = _generate_minimal_instance(
                cls=edge.source,
                path_segments=[],
                constraints_by_class=constraints_by_class,
                symbol_table=symbol_table,
            )
            environment_instance = source_instance

        target_instance: Optional[Instance] = None

        if isinstance(edge.relationship, ontology.PropertyRelationship):
            prop_name = edge.relationship.property_name

            path_segments.append(prop_name)

            target_instance = _generate_minimal_instance(
                cls=edge.target,
                path_segments=path_segments,
                constraints_by_class=constraints_by_class,
                symbol_table=symbol_table,
            )

            source_instance.properties[prop_name] = target_instance

        elif isinstance(edge.relationship, ontology.ListPropertyRelationship):
            prop_name = edge.relationship.property_name
            path_segments.append(prop_name)
            path_segments.append(0)

            target_instance = _generate_minimal_instance(
                cls=edge.target,
                path_segments=path_segments,
                constraints_by_class=constraints_by_class,
                symbol_table=symbol_table,
            )

            source_instance.properties[prop_name] = ListOfInstances(
                values=[target_instance]
            )

        else:
            aas_core_codegen.common.assert_never(edge.relationship)

        if i == len(shortest_path_in_class_graph_from_environment) - 1:
            instance_path = list(path_segments)

        assert target_instance is not None
        source_instance = target_instance

    # NOTE (mristin, 2022-05-12):
    # The name ``source_instance`` is a bit of a misnomer here. We actually refer to
    # the last generated instance which should be our desired final instance.
    assert source_instance is not None

    assert environment_instance is not None
    assert instance_path is not None

    return environment_instance, instance_path


def _make_minimal_instance_complete(
    instance: Instance,
    path_segments: List[Union[int, str]],
    cls: intermediate.ConcreteClass,
    constraints_by_class: MutableMapping[
        intermediate.ClassUnion, infer_for_schema.ConstraintsByProperty
    ],
    symbol_table: intermediate.SymbolTable,
) -> None:
    """
    Set all the optional properties in the ``instance`` in-place.

    The containing environment needs to be fixed afterwards. Use :class:`~Handyman`.
    """
    constraints_by_prop = constraints_by_class[cls]

    for prop in cls.properties:
        if isinstance(prop.type_annotation, intermediate.OptionalTypeAnnotation):
            type_anno = intermediate.beneath_optional(prop.type_annotation)

            with _extend_in_place(path_segments, [prop.name]):
                instance.properties[prop.name] = _generate_property_value(
                    type_annotation=type_anno,
                    path_segments=path_segments,
                    len_constraint=constraints_by_prop.len_constraints_by_property.get(
                        prop, None
                    ),
                    pattern_constraints=constraints_by_prop.patterns_by_property.get(
                        prop, None
                    ),
                    generate_instance=(
                        lambda a_cls, a_path_segments: _generate_minimal_instance(
                            cls=a_cls,
                            path_segments=a_path_segments,
                            constraints_by_class=constraints_by_class,
                            symbol_table=symbol_table,
                        )
                    ),
                )


class _EnvironmentInstanceReplicator:
    """Make a deep copy of the environment and _dereference the instance in the copy."""

    def __init__(
        self,
        environment: Instance,
        path_to_instance_from_environment: Sequence[Union[str, int]],
    ) -> None:
        """Initialize with the given values."""
        # NOTE (mristin, 2022-06-20):
        # Make a copy so that modifications do not mess it up
        self.environment = _deep_copy(environment)
        self.path_to_instance_from_environment = list(path_to_instance_from_environment)

    def replicate(self) -> Tuple[Instance, Instance]:
        """Replicate the environment and _dereference the instance in the copy."""
        new_environment = _deep_copy(self.environment)
        assert isinstance(new_environment, Instance)

        return (
            new_environment,
            _dereference(
                environment=new_environment,
                path_segments=self.path_to_instance_from_environment,
            ),
        )


class Case(DBC):
    """Represent an abstract test case."""

    def __init__(self, environment: Instance, expected: bool) -> None:
        """Initialize with the given values."""
        self.environment = environment
        self.expected = expected


class CaseMinimal(Case):
    """Represent a minimal test case."""

    def __init__(self, environment: Instance, cls: intermediate.ConcreteClass) -> None:
        """Initialize with the given values."""
        Case.__init__(self, environment=environment, expected=True)
        self.cls = cls


class CaseComplete(Case):
    """Represent a complete test case."""

    def __init__(self, environment: Instance, cls: intermediate.ConcreteClass) -> None:
        """Initialize with the given values."""
        Case.__init__(self, environment=environment, expected=True)
        self.cls = cls


class CaseTypeViolation(Case):
    """Represent a test case where a property has invalid type."""

    def __init__(
        self,
        environment: Instance,
        cls: intermediate.ConcreteClass,
        property_name: Identifier,
    ) -> None:
        """Initialize with the given values."""
        Case.__init__(self, environment=environment, expected=False)
        self.cls = cls
        self.property_name = property_name


class CasePositivePatternExample(Case):
    """Represent a test case with a property set to a pattern example."""

    def __init__(
        self,
        environment: Instance,
        cls: intermediate.ConcreteClass,
        property_name: Identifier,
        example_name: str,
    ) -> None:
        """Initialize with the given values."""
        Case.__init__(self, environment=environment, expected=True)
        self.cls = cls
        self.property_name = property_name
        self.example_name = example_name


class CaseNegativePatternExample(Case):
    """Represent a test case with a property set to a pattern example."""

    def __init__(
        self,
        environment: Instance,
        cls: intermediate.ConcreteClass,
        property_name: Identifier,
        example_name: str,
    ) -> None:
        """Initialize with the given values."""
        Case.__init__(self, environment=environment, expected=False)
        self.cls = cls
        self.property_name = property_name
        self.example_name = example_name


class CaseRequiredViolation(Case):
    """Represent a test case where a required property is missing."""

    def __init__(
        self,
        environment: Instance,
        cls: intermediate.ConcreteClass,
        property_name: Identifier,
    ) -> None:
        """Initialize with the given values."""
        Case.__init__(self, environment=environment, expected=False)
        self.cls = cls
        self.property_name = property_name


class CaseMinLengthViolation(Case):
    """Represent a test case where a min. len constraint is violated."""

    def __init__(
        self,
        environment: Instance,
        cls: intermediate.ConcreteClass,
        property_name: Identifier,
    ) -> None:
        """Initialize with the given values."""
        Case.__init__(self, environment=environment, expected=False)
        self.cls = cls
        self.property_name = property_name


class CaseMaxLengthViolation(Case):
    """Represent a test case where a max. len constraint is violated."""

    def __init__(
        self,
        environment: Instance,
        cls: intermediate.ConcreteClass,
        property_name: Identifier,
    ) -> None:
        """Initialize with the given values."""
        Case.__init__(self, environment=environment, expected=False)
        self.cls = cls
        self.property_name = property_name


class CaseDateTimeStampUtcViolationOnFebruary29th(Case):
    """Represent a test case where we supply an invalid UTC date time stamp."""

    def __init__(
        self,
        environment: Instance,
        cls: intermediate.ConcreteClass,
        property_name: Identifier,
    ) -> None:
        """Initialize with the given values."""
        Case.__init__(self, environment=environment, expected=False)
        self.cls = cls
        self.property_name = property_name


class CasePositiveValueExample(Case):
    """Represent a test case with a XSD value set to a positive example."""

    def __init__(
        self,
        environment: Instance,
        cls: intermediate.ConcreteClass,
        data_type_def_literal: intermediate.EnumerationLiteral,
        example_name: str,
    ) -> None:
        """Initialize with the given values."""
        Case.__init__(self, environment=environment, expected=True)
        self.cls = cls
        self.data_type_def_literal = data_type_def_literal
        self.example_name = example_name


class CaseNegativeValueExample(Case):
    """Represent a test case with a XSD value set to a negative example."""

    def __init__(
        self,
        environment: Instance,
        cls: intermediate.ConcreteClass,
        data_type_def_literal: intermediate.EnumerationLiteral,
        example_name: str,
    ) -> None:
        """Initialize with the given values."""
        Case.__init__(self, environment=environment, expected=False)
        self.cls = cls
        self.data_type_def_literal = data_type_def_literal
        self.example_name = example_name


class CasePositiveMinMaxExample(Case):
    """Represent a test case with a min/max XSD values set to a positive example."""

    def __init__(
        self,
        environment: Instance,
        cls: intermediate.ConcreteClass,
        data_type_def_literal: intermediate.EnumerationLiteral,
        example_name: str,
    ) -> None:
        """Initialize with the given values."""
        Case.__init__(self, environment=environment, expected=True)
        self.cls = cls
        self.data_type_def_literal = data_type_def_literal
        self.example_name = example_name


class CaseNegativeMinMaxExample(Case):
    """Represent a test case with a min/max XSD values set to a negative example."""

    def __init__(
        self,
        environment: Instance,
        cls: intermediate.ConcreteClass,
        data_type_def_literal: intermediate.EnumerationLiteral,
        example_name: str,
    ) -> None:
        """Initialize with the given values."""
        Case.__init__(self, environment=environment, expected=False)
        self.cls = cls
        self.data_type_def_literal = data_type_def_literal
        self.example_name = example_name


class CaseEnumViolation(Case):
    """Represent a test case with a min/max XSD values set to a negative example."""

    # fmt: on
    @require(
        lambda enum, prop: (
            type_anno := intermediate.beneath_optional(prop.type_annotation),
            isinstance(type_anno, intermediate.OurTypeAnnotation)
            and type_anno.symbol == enum,
        )[1],
        "Enum corresponds to the property",
    )
    @require(
        lambda cls, prop: id(prop) in cls.property_id_set,
        "Property belongs to the class",
    )
    # fmt: off
    def __init__(
            self,
            environment: Instance,
            enum: intermediate.Enumeration,
            cls: intermediate.ConcreteClass,
            prop: intermediate.Property
    ) -> None:
        """Initialize with the given values."""
        Case.__init__(self, environment=environment, expected=False)
        self.enum = enum
        self.cls = cls
        self.prop = prop


CaseUnion = Union[
    CaseMinimal,
    CaseComplete,
    CaseTypeViolation,
    CasePositivePatternExample,
    CaseNegativePatternExample,
    CaseRequiredViolation,
    CaseMinLengthViolation,
    CaseMaxLengthViolation,
    CaseDateTimeStampUtcViolationOnFebruary29th,
    CasePositiveValueExample,
    CaseNegativeValueExample,
    CasePositiveMinMaxExample,
    CaseNegativeMinMaxExample,
    CaseEnumViolation,
]

aas_core_codegen.common.assert_union_of_descendants_exhaustive(
    union=CaseUnion, base_class=Case
)


# fmt: off
# noinspection PyUnusedLocal
@require(
    lambda len_constraint:
    len_constraint.min_value is not None
    and len_constraint.min_value > 0
)
# fmt: on
def _make_instance_violate_min_len_constraint(
    instance: Instance,
    prop: intermediate.Property,
    len_constraint: infer_for_schema.LenConstraint,
) -> None:
    """Modify the ``instance`` in-place so that it violates the ``len_constraint``."""
    # NOTE (mristin, 2022-05-15):
    # We handle only a subset of cases here automatically since
    # otherwise it would be too difficult to implement. The
    # remainder of the cases needs to be implemented manually.

    type_anno = intermediate.beneath_optional(prop.type_annotation)

    # NOTE (mristin, 2022-06-20):
    # Please consider that the ``min_value`` > 0 in the pre-conditions.

    if isinstance(type_anno, intermediate.PrimitiveTypeAnnotation):
        if type_anno.a_type is intermediate.PrimitiveType.STR:
            instance.properties[prop.name] = ""

    elif (
        isinstance(type_anno, intermediate.OurTypeAnnotation)
        and isinstance(type_anno.symbol, intermediate.ConstrainedPrimitive)
        and (type_anno.symbol.constrainee is intermediate.PrimitiveType.STR)
    ):
        instance.properties[prop.name] = ""

    elif isinstance(type_anno, intermediate.ListTypeAnnotation):
        instance.properties[prop.name] = ListOfInstances(values=[])

    else:
        raise NotImplementedError(
            f"We did not implement the violation of len constraint "
            f"on property {prop.name!r} of type {prop.type_annotation}. "
            f"Please contact the developers."
        )


# fmt: off
# noinspection PyUnusedLocal
@require(
    lambda len_constraint:
    len_constraint.max_value is not None
)
# fmt: on
def _make_instance_violate_max_len_constraint(
    instance: Instance,
    prop: intermediate.Property,
    path_segments: List[Union[str, int]],
    len_constraint: infer_for_schema.LenConstraint,
) -> None:
    """
    Modify the ``instance`` in-place so that it violates the ``len_constraint``.

    ``path_segments`` refer to the instance, not property.
    """
    # NOTE (mristin, 2022-05-15):
    # We handle only a subset of cases here automatically since
    # otherwise it would be too difficult to implement. The
    # remainder of the cases needs to be implemented manually.
    #
    # We also optimistically assume we do not break any patterns,
    # invariants *etc.* If that is the case, you have to write
    # manual generation code.

    type_anno = intermediate.beneath_optional(prop.type_annotation)

    assert len_constraint.max_value is not None  # for mypy

    with _extend_in_place(path_segments, [prop.name]):
        too_long_text = _generate_long_string(
            length=len_constraint.max_value + 1, path_segments=path_segments
        )

    handled = False

    if isinstance(type_anno, intermediate.PrimitiveTypeAnnotation):
        if type_anno.a_type is intermediate.PrimitiveType.STR:
            instance.properties[prop.name] = too_long_text
            handled = True

        else:
            handled = False

    elif (
        isinstance(type_anno, intermediate.OurTypeAnnotation)
        and isinstance(type_anno.symbol, intermediate.ConstrainedPrimitive)
        and (type_anno.symbol.constrainee is intermediate.PrimitiveType.STR)
    ):
        instance.properties[prop.name] = too_long_text
        handled = True

    else:
        handled = False

    if not handled:
        raise NotImplementedError(
            "We could not generate the data to violate the length constraint for "
            f"the property {prop.name!r} at {_posix_path(path_segments)}. "
            f"You have to either generate the data manually, or contact the developers "
            f"to implement this missing feature."
        )


def generate(
    symbol_table: intermediate.SymbolTable,
    constraints_by_class: MutableMapping[
        intermediate.ClassUnion, infer_for_schema.ConstraintsByProperty
    ],
) -> Iterator[CaseUnion]:
    """Generate the test cases."""
    relationship_map = ontology.compute_relationship_map(symbol_table=symbol_table)

    class_graph = ontology.ClassGraph(
        relationship_map=relationship_map,
        shortest_paths=ontology.compute_shortest_paths_from_environment(
            symbol_table=symbol_table,
            relationship_map=relationship_map,
        ),
    )

    handyman = _Handyman(symbol_table=symbol_table)

    # region Generate the minimal and complete example for the Environment

    environment_cls = symbol_table.must_find(Identifier("Environment"))
    assert isinstance(environment_cls, intermediate.ConcreteClass)

    # NOTE (mristin, 2022-06-21):
    # This is a special case as we do not reach the environment from an environment.
    instance = _generate_minimal_instance(
        cls=environment_cls,
        path_segments=[],
        constraints_by_class=constraints_by_class,
        symbol_table=symbol_table,
    )

    yield CaseMinimal(environment=instance, cls=environment_cls)

    _make_minimal_instance_complete(
        instance=instance,
        path_segments=[],
        cls=environment_cls,
        constraints_by_class=constraints_by_class,
        symbol_table=symbol_table,
    )

    yield CaseComplete(environment=instance, cls=environment_cls)

    # endregion

    for symbol in symbol_table.symbols:
        if not isinstance(symbol, intermediate.ConcreteClass):
            continue

        if symbol.name not in class_graph.shortest_paths:
            # NOTE (mristin, 2022-05-12):
            # Skip the unreachable classes from the environment
            continue

        # region Minimal example

        minimal_env, path_segments = _generate_minimal_instance_in_minimal_environment(
            cls=symbol,
            class_graph=class_graph,
            constraints_by_class=constraints_by_class,
            symbol_table=symbol_table,
        )

        handyman.fix_instance(minimal_env, path_segments=[])

        yield CaseMinimal(environment=minimal_env, cls=symbol)

        replicator_minimal = _EnvironmentInstanceReplicator(
            environment=minimal_env, path_to_instance_from_environment=path_segments
        )

        # endregion

        # BEFORE-RELEASE (mristin, 2022-06-19):
        # Remove this ``if`` and implement a proper function once we tested the
        # SDK with XML.
        if symbol.name != "Submodel_element_list":
            # region Complete example

            env, instance = replicator_minimal.replicate()

            _make_minimal_instance_complete(
                instance=instance,
                path_segments=path_segments,
                cls=symbol,
                constraints_by_class=constraints_by_class,
                symbol_table=symbol_table,
            )

            handyman.fix_instance(instance=env, path_segments=[])

            yield CaseComplete(environment=env, cls=symbol)

            replicator_complete = _EnvironmentInstanceReplicator(
                environment=env, path_to_instance_from_environment=path_segments
            )

            # endregion

            # region Type violation

            for prop in symbol.properties:
                env, instance = replicator_complete.replicate()

                type_anno = intermediate.beneath_optional(prop.type_annotation)

                # NOTE (mristin, 2022-06-20):
                # If it is a primitive, supply a global reference.
                # If it is not a primitive, supply a string.

                if isinstance(type_anno, intermediate.PrimitiveTypeAnnotation) or (
                    isinstance(type_anno, intermediate.OurTypeAnnotation)
                    and isinstance(type_anno.symbol, intermediate.ConstrainedPrimitive)
                ):
                    with _extend_in_place(path_segments, [prop.name]):
                        instance.properties[prop.name] = _generate_global_reference(
                            path_segments=path_segments
                        )

                else:
                    with _extend_in_place(path_segments, [prop.name]):
                        instance.properties[prop.name] = "Unexpected string value"

                yield CaseTypeViolation(
                    environment=env, cls=symbol, property_name=prop.name
                )

            # endregion

        # region Positive and negative pattern examples

        constraints_by_prop = constraints_by_class[symbol]

        for prop in symbol.properties:
            pattern_constraints = constraints_by_prop.patterns_by_property.get(
                prop, None
            )

            if pattern_constraints is None:
                continue

            if len(pattern_constraints) > 1:
                # NOTE (mristin, 2022-06-20):
                # We currently do not know how to handle multiple patterns,
                # so we skip these properties.
                continue

            pattern_examples = frozen_examples_pattern.BY_PATTERN[
                pattern_constraints[0].pattern
            ]

            for example_name, example_text in pattern_examples.positives.items():
                env, instance = replicator_minimal.replicate()

                instance.properties[prop.name] = example_text

                yield CasePositivePatternExample(
                    environment=env,
                    cls=symbol,
                    property_name=prop.name,
                    example_name=example_name,
                )

            for example_name, example_text in pattern_examples.negatives.items():
                env, instance = replicator_minimal.replicate()

                instance.properties[prop.name] = example_text

                yield CaseNegativePatternExample(
                    environment=env,
                    cls=symbol,
                    property_name=prop.name,
                    example_name=example_name,
                )

        # endregion

        # region Required violation

        for prop in symbol.properties:
            if isinstance(prop.type_annotation, intermediate.OptionalTypeAnnotation):
                continue

            env, instance = replicator_minimal.replicate()

            del instance.properties[prop.name]

            yield CaseRequiredViolation(
                environment=env, cls=symbol, property_name=prop.name
            )

        # endregion

        # region Length violation

        for prop in symbol.properties:
            len_constraint = constraints_by_prop.len_constraints_by_property.get(
                prop, None
            )

            if len_constraint is None:
                continue

            if len_constraint.min_value is not None and len_constraint.min_value > 0:
                env, instance = replicator_minimal.replicate()

                _make_instance_violate_min_len_constraint(
                    instance=instance, prop=prop, len_constraint=len_constraint
                )

                yield CaseMinLengthViolation(
                    environment=env, cls=symbol, property_name=prop.name
                )

            if len_constraint.max_value is not None:
                env, instance = replicator_minimal.replicate()

                _make_instance_violate_max_len_constraint(
                    instance=instance,
                    path_segments=path_segments,
                    prop=prop,
                    len_constraint=len_constraint,
                )

                yield CaseMaxLengthViolation(
                    environment=env, cls=symbol, property_name=prop.name
                )

        # endregion

        # region Break date-time with UTC with February 29th

        date_time_stamp_utc_symbol = symbol_table.must_find(
            Identifier("Date_time_stamp_UTC")
        )
        assert isinstance(date_time_stamp_utc_symbol, intermediate.ConstrainedPrimitive)

        for prop in symbol.properties:
            type_anno = intermediate.beneath_optional(prop.type_annotation)
            if (
                isinstance(type_anno, intermediate.OurTypeAnnotation)
                and type_anno.symbol is date_time_stamp_utc_symbol
            ):
                env, instance = replicator_minimal.replicate()

                with _extend_in_place(path_segments, [prop.name]):
                    time_of_day = _generate_time_of_day(path_segments=path_segments)

                    instance.properties[prop.name] = f"2022-02-29T{time_of_day}Z"

                    yield CaseDateTimeStampUtcViolationOnFebruary29th(
                        environment=env, cls=symbol, property_name=prop.name
                    )

        # endregion

    # region Generate positive and negative examples for Property and Range

    property_cls = symbol_table.must_find(Identifier("Property"))
    range_cls = symbol_table.must_find(Identifier("Range"))
    extension_cls = symbol_table.must_find(Identifier("Extension"))
    qualifier_cls = symbol_table.must_find(Identifier("Qualifier"))

    data_type_def_xsd_symbol = symbol_table.must_find(Identifier("Data_type_def_XSD"))
    assert isinstance(data_type_def_xsd_symbol, intermediate.Enumeration)

    for cls in (property_cls, range_cls, extension_cls, qualifier_cls):
        assert isinstance(cls, intermediate.ConcreteClass)

        minimal_env, path_segments = _generate_minimal_instance_in_minimal_environment(
            cls=cls,
            class_graph=class_graph,
            constraints_by_class=constraints_by_class,
            symbol_table=symbol_table,
        )

        handyman.fix_instance(instance=minimal_env, path_segments=[])

        replicator_minimal = _EnvironmentInstanceReplicator(
            environment=minimal_env, path_to_instance_from_environment=path_segments
        )

        for literal in data_type_def_xsd_symbol.literals:
            examples = frozen_examples_xs_value.BY_VALUE_TYPE.get(literal.value, None)

            if examples is None:
                raise NotImplementedError(
                    f"The entry is missing "
                    f"in the {frozen_examples_xs_value.__name__!r} "
                    f"for the value type {literal.value!r}"
                )

            if cls in (property_cls, extension_cls, qualifier_cls):
                for example_name, example_value in examples.positives.items():
                    env, instance = replicator_minimal.replicate()

                    instance.properties["value"] = example_value
                    instance.properties["value_type"] = literal.value

                    yield CasePositiveValueExample(
                        environment=env,
                        cls=cls,
                        data_type_def_literal=literal,
                        example_name=example_name,
                    )

                for example_name, example_value in examples.negatives.items():
                    env, instance = replicator_minimal.replicate()

                    instance.properties["value"] = example_value
                    instance.properties["value_type"] = literal.value

                    yield CaseNegativeValueExample(
                        environment=env,
                        cls=cls,
                        data_type_def_literal=literal,
                        example_name=example_name,
                    )

            elif cls is range_cls:
                for example_name, example_value in examples.positives.items():
                    env, instance = replicator_minimal.replicate()

                    instance.properties["min"] = example_value
                    instance.properties["max"] = example_value
                    instance.properties["value_type"] = literal.value

                    yield CasePositiveMinMaxExample(
                        environment=env,
                        cls=cls,
                        data_type_def_literal=literal,
                        example_name=example_name,
                    )

                for example_name, example_value in examples.negatives.items():
                    env, instance = replicator_minimal.replicate()

                    instance.properties["min"] = example_value
                    instance.properties["max"] = example_value
                    instance.properties["value_type"] = literal.value

                    yield CaseNegativeMinMaxExample(
                        environment=env,
                        cls=cls,
                        data_type_def_literal=literal,
                        example_name=example_name,
                    )
            else:
                raise AssertionError(f"Unexpected {cls=}")

    # endregion

    # region Generate enum violations

    # fmt: off
    enums_props_classes: List[
        Tuple[
            intermediate.Enumeration,
            intermediate.Property,
            intermediate.ConcreteClass
        ]
    ] = []
    # fmt: on

    observed_enums = set()  # type: Set[Identifier]

    for symbol in symbol_table.symbols:
        if not isinstance(symbol, intermediate.ConcreteClass):
            continue

        for prop in symbol.properties:
            type_anno = intermediate.beneath_optional(prop.type_annotation)

            if not (
                isinstance(type_anno, intermediate.OurTypeAnnotation)
                and isinstance(type_anno.symbol, intermediate.Enumeration)
                and type_anno.symbol.name not in observed_enums
            ):
                continue

            enums_props_classes.append((type_anno.symbol, prop, symbol))

    for enum, prop, cls in enums_props_classes:
        minimal_env, path_segments = _generate_minimal_instance_in_minimal_environment(
            cls=cls,
            class_graph=class_graph,
            constraints_by_class=constraints_by_class,
            symbol_table=symbol_table,
        )

        literal_value_set = {literal.value for literal in enum.literals}

        instance = _dereference(environment=minimal_env, path_segments=path_segments)
        with _extend_in_place(path_segments, [prop.name]):
            literal_value = "invalid-literal"
            while literal_value in literal_value_set:
                literal_value = f"really-{literal_value}"

            instance.properties[prop.name] = literal_value

        yield CaseEnumViolation(environment=minimal_env, enum=enum, cls=cls, prop=prop)

    # endregion

    # BEFORE-RELEASE (mristin, 2022-06-19):
    # Manually write Unexpected/ConstraintViolation/{class name}/
    # {describe how we break it somehow}.json
