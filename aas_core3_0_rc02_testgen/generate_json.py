"""Generate test data in JSON for the meta-model V3RC02."""
import base64
import collections
import collections.abc
import copy
import itertools
import json
import os
import pathlib
from typing import (
    Union,
    Final,
    Mapping,
    Tuple,
    OrderedDict,
    Sequence,
    List,
    MutableMapping,
    Any,
    Optional,
    Callable,
)

import aas_core_codegen.common
import aas_core_meta.v3rc2
import networkx
from aas_core_codegen import intermediate, infer_for_schema, naming
from aas_core_codegen.common import Identifier

from aas_core3_0_rc02_testgen import common
from aas_core3_0_rc02_testgen.frozen_examples import (
    pattern as frozen_examples_pattern,
    xs_value as frozen_examples_xs_value,
)


class PropertyRelationship:
    """
    Model a relationship between two classes as a property.

    Namely, instances of the target class appear as a property of the source class.
    """

    def __init__(self, property_name: Identifier) -> None:
        """Initialize with the given values."""
        self.property_name = property_name


class ListPropertyRelationship:
    """
    Model a relationship between two classes as an item of a list property.

    Namely, instances of the target class appear as items in a list property of
    the source class.
    """

    def __init__(self, property_name: Identifier) -> None:
        """Initialize with the given values."""
        self.property_name = property_name


RelationshipUnion = Union[PropertyRelationship, ListPropertyRelationship]

RelationshipMap = Mapping[Tuple[Identifier, Identifier], RelationshipUnion]


class Segment:
    """Represent a segment from the class ``Environment`` to a concrete class."""

    def __init__(
        self,
        source: intermediate.ConcreteClass,
        target: intermediate.ConcreteClass,
        relationship: RelationshipUnion,
    ) -> None:
        """Initialize with the given values."""
        self.source = source
        self.target = target
        self.relationship = relationship


ShortestPathMap = Mapping[Identifier, Sequence[Segment]]


class ClassGraph:
    """Model how classes of the meta-model are related to each other."""

    relationship_map: Final[RelationshipMap]
    shortest_paths: Final[ShortestPathMap]

    def __init__(
        self, relationship_map: RelationshipMap, shortest_paths: ShortestPathMap
    ) -> None:
        """Initialize with the given values."""
        self.relationship_map = relationship_map
        self.shortest_paths = shortest_paths


def compute_relationship_map(symbol_table: intermediate.SymbolTable) -> RelationshipMap:
    """Compute the relationships between the classes as edges in the class graph."""
    rel_map: OrderedDict[
        Tuple[Identifier, Identifier], RelationshipUnion
    ] = collections.OrderedDict()

    for symbol in symbol_table.symbols:
        if isinstance(symbol, intermediate.ConcreteClass):
            for prop in symbol.properties:
                type_anno = intermediate.beneath_optional(prop.type_annotation)

                if isinstance(type_anno, intermediate.ListTypeAnnotation):
                    assert isinstance(
                        type_anno.items, intermediate.OurTypeAnnotation
                    ), (
                        "Expected only lists of enums or classes in the meta-model, "
                        f"but got: {type_anno}"
                    )

                    if isinstance(type_anno.items.symbol, intermediate.AbstractClass):
                        for concrete_cls in type_anno.items.symbol.concrete_descendants:
                            source_target = (symbol.name, concrete_cls.name)

                            rel = rel_map.get(source_target, None)
                            if rel is None:
                                rel_map[source_target] = ListPropertyRelationship(
                                    property_name=prop.name
                                )
                    elif isinstance(type_anno.items.symbol, intermediate.ConcreteClass):
                        for concrete_cls in itertools.chain(
                            [type_anno.items.symbol],
                            type_anno.items.symbol.concrete_descendants,
                        ):
                            source_target = (symbol.name, concrete_cls.name)

                            rel = rel_map.get(source_target, None)
                            if rel is None:
                                rel_map[source_target] = ListPropertyRelationship(
                                    property_name=prop.name
                                )
                    else:
                        pass

                elif isinstance(type_anno, intermediate.OurTypeAnnotation):
                    if isinstance(type_anno.symbol, intermediate.AbstractClass):
                        for concrete_cls in type_anno.symbol.concrete_descendants:
                            source_target = (symbol.name, concrete_cls.name)

                            rel = rel_map.get(source_target, None)

                            # NOTE (mristin, 2022-05-07):
                            # Property edges have smaller distance than list-property
                            # edges. Hence, we keep only the shortest edges.
                            #
                            # See: https://groups.google.com/g/networkx-discuss/c/87uC9F0ug8Y
                            if rel is None or isinstance(rel, ListPropertyRelationship):
                                rel_map[source_target] = PropertyRelationship(
                                    property_name=prop.name
                                )

                    elif isinstance(type_anno.symbol, intermediate.ConcreteClass):
                        for concrete_cls in itertools.chain(
                            [type_anno.symbol], type_anno.symbol.concrete_descendants
                        ):
                            source_target = (symbol.name, concrete_cls.name)

                            rel = rel_map.get(source_target, None)

                            # NOTE (mristin, 2022-05-07):
                            # See the note above re property edge *versus* list-property
                            # edge.
                            if rel is None or isinstance(rel, ListPropertyRelationship):
                                rel_map[source_target] = PropertyRelationship(
                                    property_name=prop.name
                                )
                    else:
                        pass

    return rel_map


def compute_shortest_paths_from_environment(
    symbol_table: intermediate.SymbolTable, relationship_map: RelationshipMap
) -> ShortestPathMap:
    """Compute the shortest path from the environment to the concrete classes."""
    graph = networkx.DiGraph()
    for symbol in symbol_table.symbols:
        if isinstance(symbol, intermediate.ConcreteClass):
            graph.add_node(symbol.name)

    for (source, target), relationship in relationship_map.items():
        if isinstance(relationship, PropertyRelationship):
            graph.add_edge(source, target, weight=1)
        elif isinstance(relationship, ListPropertyRelationship):
            # NOTE (mristin, 2022-05-07):
            # Creating a list and adding an item is more work than creating an instance.
            # Thus, we pay two coins for the list-property creation.
            graph.add_edge(source, target, weight=2)
        else:
            aas_core_codegen.common.assert_never(relationship)

    _, raw_path_map = networkx.single_source_dijkstra(G=graph, source="Environment")

    path_map: OrderedDict[Identifier, Sequence[Segment]] = collections.OrderedDict()

    for symbol in symbol_table.symbols:
        if symbol.name == "Environment":
            continue

        raw_path = raw_path_map.get(symbol.name, None)
        if raw_path is None:
            continue

        assert len(raw_path) >= 2
        cursor = iter(raw_path)
        current = next(cursor, None)

        assert current is None or isinstance(current, str)

        path: List[Segment] = []

        while True:
            prev = current
            current = next(cursor, None)
            assert current is None or isinstance(current, str)

            if current is None:
                break

            assert prev is not None
            source_symbol = symbol_table.must_find(Identifier(prev))
            assert isinstance(
                source_symbol, intermediate.ConcreteClass
            ), "Only edges between concrete classes expected in the graph"

            assert current is not None
            target_symbol = symbol_table.must_find(Identifier(current))
            assert isinstance(
                target_symbol, intermediate.ConcreteClass
            ), "Only edges between concrete classes expected in the graph"

            relationship = relationship_map[(source_symbol.name, target_symbol.name)]
            path.append(
                Segment(
                    source=source_symbol,
                    target=target_symbol,
                    relationship=relationship,
                )
            )

        path_map[symbol.name] = path

    return path_map


def generate_value(
    type_annotation: intermediate.TypeAnnotationUnion,
    path_segments: List[Union[str, int]],
    len_constraint: Optional[infer_for_schema.LenConstraint],
    pattern_constraints: Optional[Sequence[infer_for_schema.PatternConstraint]],
    generate_instance: Callable[
        [intermediate.ConcreteClass, List[Union[str, int]]], OrderedDict[str, Any]
    ],
) -> Any:
    """
    Generate the value without side effects based on the ``path_segments``.

    The callable ``generate_instance`` instructs how to generate the instances
    recursively.
    """

    def implementation() -> Any:
        """Wrap the body in the separate function so that we can ensure the result."""
        type_anno = intermediate.beneath_optional(type_annotation)

        # noinspection PyUnusedLocal
        primitive_type = None  # type: Optional[intermediate.PrimitiveType]
        if isinstance(type_anno, intermediate.PrimitiveTypeAnnotation):
            primitive_type = type_anno.a_type
        elif isinstance(type_anno, intermediate.OurTypeAnnotation) and isinstance(
            type_anno.symbol, intermediate.ConstrainedPrimitive
        ):
            primitive_type = type_anno.symbol.constrainee
        else:
            # It is not a primitive type.
            primitive_type = None

        hsh = common.hash_path(path_segments=path_segments)

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
                    f"for the value {common.posix_path(path_segments)}: {patterns}"
                )

            if (
                primitive_type is None
                or primitive_type is not intermediate.PrimitiveType.STR
            ):
                raise NotImplementedError(
                    "We did not implement the generation of a non-string value with "
                    "the pattern constraint, which is the case "
                    f"for the value {common.posix_path(path_segments)}"
                )
            else:
                assert primitive_type is intermediate.PrimitiveType.STR

            assert len(pattern_constraints) > 0, "Unexpected empty pattern constraints"

            pattern = pattern_constraints[0].pattern
            pattern_examples = frozen_examples_pattern.BY_PATTERN.get(pattern, None)
            if pattern_examples is None:
                raise NotImplementedError(
                    f"The entry is missing "
                    f"in the {frozen_examples_pattern.__name__!r} "
                    f"for the pattern {pattern!r} "
                    f"when generating the value at {common.posix_path(path_segments)}"
                )

            if len(pattern_examples.positives) == 0:
                raise NotImplementedError(
                    f"Unexpected an empty list of positive examples "
                    f"in the {frozen_examples_pattern.__name__!r} "
                    f"for the pattern {pattern!r} "
                    f"when generating the value at {common.posix_path(path_segments)}"
                )

            for value in pattern_examples.positives.values():
                return value

            raise AssertionError("Expected to check for at least one positive example")

        # endregion

        if isinstance(type_anno, intermediate.PrimitiveTypeAnnotation) or (
            isinstance(type_anno, intermediate.OurTypeAnnotation)
            and isinstance(type_anno.symbol, intermediate.ConstrainedPrimitive)
        ):
            assert primitive_type is not None

            hsh_as_int = int(hsh, base=16)
            if primitive_type is intermediate.PrimitiveType.BOOL:
                return hsh_as_int % 2 == 0

            elif primitive_type is intermediate.PrimitiveType.INT:
                return hsh_as_int

            elif primitive_type is intermediate.PrimitiveType.FLOAT:
                return float(hsh_as_int) / 100

            elif primitive_type is intermediate.PrimitiveType.STR:
                return f"something_random_{hsh}"

            elif primitive_type is intermediate.PrimitiveType.BYTEARRAY:
                return base64.b64encode(bytearray.fromhex(hsh)).decode(encoding="ascii")

            else:
                aas_core_codegen.common.assert_never(primitive_type)

        elif isinstance(type_anno, intermediate.OurTypeAnnotation):
            if isinstance(type_anno.symbol, intermediate.Enumeration):
                hsh_as_int = int(hsh, base=16)
                return type_anno.symbol.literals[
                    hsh_as_int % len(type_anno.symbol.literals)
                ].value

            elif isinstance(type_anno.symbol, intermediate.ConstrainedPrimitive):
                raise AssertionError(
                    f"Should have been handled before: {type_anno.symbol}"
                )

            elif isinstance(
                type_anno.symbol,
                (intermediate.AbstractClass, intermediate.ConcreteClass),
            ):
                if type_anno.symbol.interface is not None:
                    concrete_classes = type_anno.symbol.interface.implementers
                    hsh_as_int = int(hsh, base=16)
                    concrete_cls = concrete_classes[hsh_as_int % len(concrete_classes)]

                    return generate_instance(concrete_cls, path_segments)
                else:
                    assert isinstance(type_anno.symbol, intermediate.ConcreteClass)
                    return generate_instance(type_anno.symbol, path_segments)
            else:
                aas_core_codegen.common.assert_never(type_anno.symbol)

        elif isinstance(type_anno, intermediate.ListTypeAnnotation):
            assert isinstance(
                type_anno.items, intermediate.OurTypeAnnotation
            ) and isinstance(
                type_anno.items.symbol,
                (intermediate.AbstractClass, intermediate.ConcreteClass),
            ), f"Expected all lists to be lists of classes, but got: {type_anno}"

            path_segments.append(1)
            try:
                if type_anno.items.symbol.interface is not None:
                    concrete_classes = type_anno.items.symbol.interface.implementers
                    hsh_as_int = int(hsh, base=16)
                    concrete_cls = concrete_classes[hsh_as_int % len(concrete_classes)]

                    return [generate_instance(concrete_cls, path_segments)]
                else:
                    assert isinstance(
                        type_anno.items.symbol, intermediate.ConcreteClass
                    )
                    return [generate_instance(type_anno.items.symbol, path_segments)]
            finally:
                path_segments.pop()
        else:
            aas_core_codegen.common.assert_never(type_anno)

    # NOTE (mristin, 2022-05-11):
    # We ensure here that the constraint on ``len(.)`` of the result is satisfied.
    # This covers some potential errors, but mind that still does not check
    # the constraints. Hence, you have to manually inspect the generated data and
    # decide yourself whether you need to write a generator manually.

    result = implementation()

    if len_constraint is not None and isinstance(result, (str, list)):
        if (
            len_constraint.min_value is not None
            and len(result) < len_constraint.min_value
        ) or (
            len_constraint.max_value is not None
            and len(result) > len_constraint.max_value
        ):
            raise ValueError(
                f"Expected the value {common.posix_path(path_segments)} "
                f"to satisfy the length constraint "
                f"[{len_constraint.min_value!r}, {len_constraint.max_value!r}], "
                f"but got the length {len(result)}. You have to write the generator "
                f"for this property or instance yourself"
            )

    return result


def generate_property(
    prop: intermediate.Property,
    constraints_by_prop: infer_for_schema.ConstraintsByProperty,
    path_segments: List[Union[str, int]],
    generate_instance: Callable[
        [intermediate.ConcreteClass, List[Union[str, int]]], OrderedDict[str, Any]
    ],
) -> Any:
    """
    Generate the property ``prop`` of an instance of ``cls``.

    The ``path_segments`` points to the property.

    The ``generate_instance`` callable is used to recursively generate further
    instances (not necessarily of the class ``cls``, but of any class in
    the meta-model).
    """
    return generate_value(
        type_annotation=prop.type_annotation,
        path_segments=path_segments,
        len_constraint=constraints_by_prop.len_constraints_by_property.get(prop, None),
        pattern_constraints=constraints_by_prop.patterns_by_property.get(prop, None),
        generate_instance=generate_instance,
    )


def generate_minimal_instance(
    cls: intermediate.ConcreteClass,
    path_segments: List[Union[str, int]],
    constraints_by_class: MutableMapping[
        intermediate.ClassUnion, infer_for_schema.ConstraintsByProperty
    ],
    symbol_table: intermediate.SymbolTable,
) -> OrderedDict[str, Any]:
    """
    Generate an instance with only required properties.

    The ``path_segments`` refer to the JSON path leading to the instance of the ``cls``.

    We recursively generate minimal instances for all the nested classes.
    We will re-use the ``path`` in the subsequent recursive calls to avoid
    the quadratic time complexity, so beware that this function is *NOT* thread-safe.

    The generation should not be random, *i.e.*, re-generating with the same input
    should yield the same output.
    """
    reference_cls = symbol_table.must_find(Identifier("Reference"))
    if cls is reference_cls:
        # NOTE (mristin, 2022-06-19):
        # We generate a global reference by default, since this makes for much better
        # examples with less confusion for the reader. If you need something else, fix
        # it afterwards.
        return generate_global_reference(path_segments=path_segments)

    constraints_by_prop = constraints_by_class[cls]

    instance: OrderedDict[str, Any] = collections.OrderedDict()

    for prop in cls.properties:
        if isinstance(prop.type_annotation, intermediate.OptionalTypeAnnotation):
            continue

        path_segments.append(prop.name)
        try:
            # fmt: off
            instance[naming.json_property(prop.name)] = generate_property(
                prop=prop,
                path_segments=path_segments,
                constraints_by_prop=constraints_by_prop,
                generate_instance=(
                    lambda a_cls, a_path_segments:
                    generate_minimal_instance(
                        cls=a_cls, path_segments=a_path_segments,
                        constraints_by_class=constraints_by_class,
                        symbol_table=symbol_table
                    )
                )
            )
            # fmt: on
        finally:
            path_segments.pop()

    # region Fix for specific class

    basic_event_element_cls = symbol_table.must_find(Identifier("Basic_event_element"))
    if cls == basic_event_element_cls:
        # Fix that the observed is a proper model reference
        instance["observed"] = generate_model_reference(
            expected_type=aas_core_meta.v3rc2.Key_types.Referable,
            path_segments=path_segments + ["observed"],
        )

        # Override that the direction is output so that we can always set
        # the max interval
        instance["direction"] = "OUTPUT"

    # endregion

    # region Set modelType

    if cls.serialization is not None and cls.serialization.with_model_type:
        instance["modelType"] = naming.json_model_type(cls.name)

    # endregion

    return instance


class InstanceWithPath:
    """Represent a JSON-able instance with the path to it from the environment."""

    def __init__(
        self,
        instance: MutableMapping[str, Any],
        path_segments: Sequence[Union[str, int]],
    ) -> None:
        self.instance = instance
        self.path_segments = path_segments


def generate_minimal_instance_in_minimal_environment(
    cls: intermediate.ConcreteClass,
    class_graph: ClassGraph,
    constraints_by_class: MutableMapping[
        intermediate.ClassUnion, infer_for_schema.ConstraintsByProperty
    ],
    symbol_table: intermediate.SymbolTable,
) -> Tuple[OrderedDict[str, Any], List[Union[str, int]]]:
    """
    Generate the minimal instance of ``cls`` in a minimal environment instance.

    Return the environment and the path to the instance.
    """
    shortest_path_in_class_graph_from_environment = class_graph.shortest_paths[cls.name]

    environment_instance: Optional[OrderedDict[str, Any]] = None

    path_segments: List[Union[str, int]] = []
    source_instance: Optional[OrderedDict[str, Any]] = None

    # NOTE (mristin, 2022-05-13):
    # We have to keep track of submodels so that we can set the idShorts on their
    # elements.
    submodels = []  # type: List[InstanceWithPath]
    submodel_cls = symbol_table.must_find(Identifier("Submodel"))
    assert isinstance(submodel_cls, intermediate.ConcreteClass)

    # NOTE (mristin, 2022-05-14):
    # We need to track asset administration shells so that we set
    # the references derivedFrom and submodels correctly.
    asset_administration_shells = []  # type: List[InstanceWithPath]
    asset_administration_shell_cls = symbol_table.must_find(
        Identifier(Identifier("Asset_administration_shell"))
    )
    assert isinstance(asset_administration_shell_cls, intermediate.ConcreteClass)

    instance_path = None  # type: Optional[List[Union[int, str]]]

    for i, segment in enumerate(shortest_path_in_class_graph_from_environment):
        if source_instance is None:
            assert segment.source.name == "Environment", (
                "Expected the generation to start from an instance "
                "of the class 'Environment'"
            )
            source_instance = generate_minimal_instance(
                cls=segment.source,
                path_segments=[],
                constraints_by_class=constraints_by_class,
                symbol_table=symbol_table,
            )
            environment_instance = source_instance

        target_instance: Optional[OrderedDict[str, Any]] = None

        if isinstance(segment.relationship, PropertyRelationship):
            prop_name = segment.relationship.property_name
            path_segments.append(naming.json_property(prop_name))
            target_instance = generate_minimal_instance(
                cls=segment.target,
                path_segments=path_segments,
                constraints_by_class=constraints_by_class,
                symbol_table=symbol_table,
            )

            source_instance[naming.json_property(prop_name)] = target_instance

        elif isinstance(segment.relationship, ListPropertyRelationship):
            prop_name = segment.relationship.property_name
            path_segments.append(naming.json_property(prop_name))
            path_segments.append(0)
            target_instance = generate_minimal_instance(
                cls=segment.target,
                path_segments=path_segments,
                constraints_by_class=constraints_by_class,
                symbol_table=symbol_table,
            )

            source_instance[naming.json_property(prop_name)] = [target_instance]

        else:
            aas_core_codegen.common.assert_never(segment.relationship)

        if i == len(shortest_path_in_class_graph_from_environment) - 1:
            instance_path = list(path_segments)

        if segment.target.is_subclass_of(submodel_cls):
            submodels.append(
                InstanceWithPath(
                    instance=target_instance, path_segments=list(path_segments)
                )
            )

        elif segment.target.is_subclass_of(asset_administration_shell_cls):
            asset_administration_shells.append(
                InstanceWithPath(
                    instance=target_instance, path_segments=list(path_segments)
                )
            )
        else:
            pass

        assert target_instance is not None
        source_instance = target_instance

    # NOTE (mristin, 2022-05-12):
    # The name ``source_instance`` is a bit of a misnomer here. We actually refer to
    # the last generated instance which should be our desired final instance.
    assert source_instance is not None

    assert environment_instance is not None

    # region Fix the invariant that all the submodel elements have the IdShort set

    for submodel in submodels:
        submodel_elements = submodel.instance.get("submodelElements", None)
        if submodel_elements is not None:
            for submodel_element in submodel.instance["submodelElements"]:
                submodel_element[
                    "idShort"
                ] = f"some_id_short_{common.hash_path(submodel.path_segments)}"

    # endregion

    # region Fix the invariant that the derivedFrom is of correct type

    for asset_administration_shell in asset_administration_shells:
        derived_from = asset_administration_shell.instance.get("derivedFrom", None)
        if derived_from is not None:
            # len(reference.keys) != 0 or reference.keys[-1].type == expected_type
            assert "keys" in derived_from and len(derived_from["keys"]) > 0, (
                f"Unexpected derivedFrom with empty keys: "
                f"{json.dumps(asset_administration_shell.instance)}"
            )

            # Fix
            derived_from["keys"][-1]["type"] = "AssetAdministrationShell"

    # endregion

    assert instance_path is not None

    return environment_instance, instance_path


def dereference(
    environment: MutableMapping[str, Any], path_segments: Sequence[Union[int, str]]
) -> MutableMapping[str, Any]:
    """Dereference the path to an object starting from an environment."""
    result = environment  # type: Any
    for i, segment in enumerate(path_segments):
        if (isinstance(segment, str) and segment not in result) or (
            isinstance(segment, int) and segment >= len(result)
        ):
            raise AssertionError(
                f"Expected the path {path_segments} in the environment: "
                f"{json.dumps(environment, indent=2)}; "
                f"the segment {i + 1}, {segment!r}, was not there"
            )

        result = result[segment]

    if not isinstance(result, collections.abc.MutableMapping):
        raise AssertionError(
            f"Unexpected non-mapping after " f"dereferencing {path_segments=}: {result}"
        )

    for key in result:
        if not isinstance(key, str):
            raise AssertionError(
                f"Unexpected non-string key {key} after "
                f"dereferencing {path_segments=}: {result}"
            )

    return result


def make_minimal_instance_complete(
    instance: MutableMapping[str, Any],
    path_segments: List[Union[int, str]],
    cls: intermediate.ConcreteClass,
    constraints_by_class: MutableMapping[
        intermediate.ClassUnion, infer_for_schema.ConstraintsByProperty
    ],
    symbol_table: intermediate.SymbolTable,
) -> None:
    """Set all the optional properties in the ``instance``."""
    constraints_by_prop = constraints_by_class[cls]

    data_element_cls = symbol_table.must_find(Identifier("Data_element"))
    assert isinstance(data_element_cls, intermediate.AbstractClass)

    asset_administration_shell_cls = symbol_table.must_find(
        Identifier("Asset_administration_shell")
    )
    assert isinstance(asset_administration_shell_cls, intermediate.ConcreteClass)

    concept_description_cls = symbol_table.must_find(Identifier("Concept_description"))
    assert isinstance(concept_description_cls, intermediate.ConcreteClass)

    entity_cls = symbol_table.must_find(Identifier("Entity"))
    assert isinstance(entity_cls, intermediate.ConcreteClass)

    property_cls = symbol_table.must_find(Identifier("Property"))
    assert isinstance(property_cls, intermediate.ConcreteClass)

    qualifier_cls = symbol_table.must_find(Identifier("Qualifier"))
    assert isinstance(qualifier_cls, intermediate.ConcreteClass)

    range_cls = symbol_table.must_find(Identifier("Range"))
    assert isinstance(range_cls, intermediate.ConcreteClass)

    submodel_cls = symbol_table.must_find(Identifier("Submodel"))
    assert isinstance(submodel_cls, intermediate.ConcreteClass)

    submodel_element_collection_cls = symbol_table.must_find(
        Identifier("Submodel_element_collection")
    )
    assert isinstance(submodel_element_collection_cls, intermediate.ConcreteClass)

    basic_event_element_cls = symbol_table.must_find(Identifier("Basic_event_element"))
    assert isinstance(basic_event_element_cls, intermediate.ConcreteClass)

    for prop in cls.properties:
        if isinstance(prop.type_annotation, intermediate.OptionalTypeAnnotation):
            path_segments.append(prop.name)
            try:
                # fmt: off
                instance[naming.json_property(prop.name)] = generate_property(
                    prop=prop,
                    path_segments=path_segments,
                    constraints_by_prop=constraints_by_prop,
                    generate_instance=(
                        lambda a_cls, a_path_segments:
                        generate_minimal_instance(
                            cls=a_cls, path_segments=a_path_segments,
                            constraints_by_class=constraints_by_class,
                            symbol_table=symbol_table
                        )
                    )
                )
                # fmt: on

            finally:
                path_segments.pop()

    # region Fix for the ancestor classes

    if cls.is_subclass_of(data_element_cls):
        if instance["category"] not in ["CONSTANT", "PARAMETER", "VARIABLE"]:
            instance["category"] = "CONSTANT"

    # endregion

    # region Fix for the concrete class

    if cls == asset_administration_shell_cls:
        # Fix the derivedFrom to be a proper model reference
        instance["derivedFrom"] = generate_model_reference(
            expected_type=aas_core_meta.v3rc2.Key_types.Asset_administration_shell,
            path_segments=path_segments + ["derivedFrom"],
        )

        # Fix the submodels to be proper model references
        instance["submodels"] = [
            generate_model_reference(
                expected_type=aas_core_meta.v3rc2.Key_types.Submodel,
                path_segments=path_segments + ["submodels"],
            )
        ]

    elif cls == concept_description_cls:
        instance["category"] = "VALUE"

    elif cls == entity_cls:
        if instance["entityType"] == "SelfManagedEntity":
            del instance["specificAssetId"]
        else:
            del instance["specificAssetId"]
            del instance["globalAssetId"]

    elif cls == property_cls:
        # NOTE (mristin, 2022-05-15):
        # We hard-code the type and the value since this would be otherwise
        # unmaintainable.
        instance["value"] = "true"
        instance["valueType"] = "xs:boolean"

    elif cls == qualifier_cls:
        # NOTE (mristin, 2022-05-15):
        # We hard-code the type and the value since this would be otherwise
        # unmaintainable.
        instance["value"] = "1234"
        instance["valueType"] = "xs:int"

    elif cls == range_cls:
        # NOTE (mristin, 2022-05-15):
        # We hard-code the type and the value since this would be otherwise
        # unmaintainable.
        instance["min"] = "1234"
        instance["max"] = "4321"
        instance["valueType"] = "xs:int"

    elif cls == submodel_cls:
        try:
            path_segments.append("submodelElements")

            for i, submodel_element in enumerate(instance["submodelElements"]):
                path_segments.append(i)
                try:
                    submodel_element[
                        "idShort"
                    ] = f"some_id_short_{common.hash_path(path_segments)}"
                finally:
                    path_segments.pop()
        finally:
            path_segments.pop()

        # region Fix qualifiers

        if (
            instance.get("kind", None)
            == aas_core_meta.v3rc2.Modeling_kind.Template.value
        ):
            qualifiers = instance.get("qualifiers", None)
            if qualifiers is not None:
                for qualifier in qualifiers:
                    qualifier[
                        "kind"
                    ] = aas_core_meta.v3rc2.Qualifier_kind.Template_qualifier.value

        # endregion

    elif cls == submodel_element_collection_cls:
        for i, value in enumerate(instance["value"]):
            path_segments.append(i)
            try:
                value["idShort"] = f"some_id_short_{common.hash_path(path_segments)}"
            finally:
                path_segments.pop()

    elif cls == basic_event_element_cls:
        # Fix that the message_broker is a proper model reference
        instance["messageBroker"] = generate_model_reference(
            expected_type=aas_core_meta.v3rc2.Key_types.Referable,
            path_segments=path_segments + ["messageBroker"],
        )

    else:
        # No fix is necessary.
        pass

    # endregion


def generate_model_reference(
    expected_type: aas_core_meta.v3rc2.Key_types,
    path_segments: List[Union[str, int]],
) -> OrderedDict[str, Any]:
    """Generate a model Reference pointing to an instance of ``expected_type``."""
    instance = collections.OrderedDict()  # type: OrderedDict[str, Any]
    instance["type"] = aas_core_meta.v3rc2.Reference_types.Model_reference.value

    if expected_type in (
        aas_core_meta.v3rc2.Key_types.Asset_administration_shell,
        aas_core_meta.v3rc2.Key_types.Concept_description,
        aas_core_meta.v3rc2.Key_types.Submodel,
    ):
        instance["keys"] = [
            collections.OrderedDict(
                [
                    ("type", expected_type.value),
                    ("value", common.hash_path(path_segments + ["keys", 0, "value"])),
                ]
            )
        ]
    elif expected_type is aas_core_meta.v3rc2.Key_types.Referable:
        instance["keys"] = [
            collections.OrderedDict(
                [
                    ("type", aas_core_meta.v3rc2.Key_types.Submodel.value),
                    (
                        "value",
                        "something_random_"
                        + common.hash_path(path_segments + ["keys", 0, "value"]),
                    ),
                ]
            ),
            collections.OrderedDict(
                [
                    ("type", aas_core_meta.v3rc2.Key_types.Referable.value),
                    (
                        "value",
                        "something_random_"
                        + common.hash_path(path_segments + ["keys", 1, "value"]),
                    ),
                ]
            ),
        ]
    else:
        raise NotImplementedError(
            f"Unhandled {expected_type=}; when we developed this script there were "
            f"no other key types expected in the meta-model as a reference, "
            f"but this has obvious changed. Please contact the developers."
        )

    return instance


def generate_global_reference(
    path_segments: List[Union[str, int]],
) -> OrderedDict[str, Any]:
    """Generate an instance of a global Reference."""
    instance = collections.OrderedDict()  # type: OrderedDict[str, Any]
    instance["type"] = aas_core_meta.v3rc2.Reference_types.Global_reference.value

    instance["keys"] = [
        collections.OrderedDict(
            [
                ("type", "GlobalReference"),
                (
                    "value",
                    "something_random_"
                    + common.hash_path(path_segments + ["keys", 0, "value"]),
                ),
            ]
        )
    ]

    return instance


def _copy_minimal_environment_and_instance(
    minimal_environment: MutableMapping[str, Any],
    path_to_instance_from_environment: Sequence[Union[str, int]],
) -> Tuple[MutableMapping[str, Any], MutableMapping[str, Any]]:
    """
    Make a deep copy of the minimal environment.

    Return the copied environment as well as the instance.
    """
    environment_copy = copy.deepcopy(minimal_environment)
    instance_copy = dereference(
        environment=environment_copy,
        path_segments=path_to_instance_from_environment,
    )

    return environment_copy, instance_copy


def _copy_complete_environment_and_instance(
    complete_environment: MutableMapping[str, Any],
    path_to_instance_from_environment: Sequence[Union[int, str]],
) -> Tuple[MutableMapping[str, Any], MutableMapping[str, Any]]:
    """
    Make a deep copy of the environment containing a complete instance.

    Return the copied environment as well as the instance.
    """
    environment_copy = copy.deepcopy(complete_environment)
    instance_copy = dereference(
        environment=environment_copy,
        path_segments=path_to_instance_from_environment,
    )

    return environment_copy, instance_copy


def generate(test_data_dir: pathlib.Path) -> None:
    """Generate the JSON files."""
    (
        symbol_table,
        constraints_by_class,
    ) = common.load_symbol_table_and_infer_constraints_for_schema()

    rel_map = compute_relationship_map(symbol_table=symbol_table)

    class_graph = ClassGraph(
        relationship_map=rel_map,
        shortest_paths=compute_shortest_paths_from_environment(
            symbol_table=symbol_table,
            relationship_map=rel_map,
        ),
    )

    # noinspection PyUnusedLocal
    environment = None  # type: Optional[MutableMapping[str, Any]]

    for symbol in symbol_table.symbols:
        if isinstance(symbol, intermediate.ConcreteClass):
            if symbol.name not in class_graph.shortest_paths:
                # NOTE (mristin, 2022-05-12):
                # Skip the unreachable classes from the environment
                continue

            # region Minimal example

            (
                minimal_environment,
                path_to_instance_from_environment,
            ) = generate_minimal_instance_in_minimal_environment(
                cls=symbol,
                class_graph=class_graph,
                constraints_by_class=constraints_by_class,
                symbol_table=symbol_table,
            )

            pth = (
                test_data_dir
                / "Json"
                / "Expected"
                / naming.json_model_type(symbol.name)
                / "minimal.json"
            )
            pth.parent.mkdir(exist_ok=True, parents=True)

            with pth.open("wt", encoding="utf-8") as fid:
                json.dump(minimal_environment, fid, indent=2, sort_keys=True)

            # endregion

            # BEFORE-RELEASE (mristin, 2022-06-19):
            # Remove this ``if`` and implement a proper function once we tested the
            # SDK with XML.
            if symbol.name != "Submodel_element_list":
                # region Complete example

                environment, instance = _copy_minimal_environment_and_instance(
                    minimal_environment=minimal_environment,
                    path_to_instance_from_environment=path_to_instance_from_environment,
                )

                make_minimal_instance_complete(
                    instance=instance,
                    path_segments=path_to_instance_from_environment,
                    cls=symbol,
                    constraints_by_class=constraints_by_class,
                    symbol_table=symbol_table,
                )

                pth = (
                    test_data_dir
                    / "Json"
                    / "Expected"
                    / naming.json_model_type(symbol.name)
                    / "complete.json"
                )
                with pth.open("wt", encoding="utf-8") as fid:
                    json.dump(environment, fid, indent=2, sort_keys=True)

                complete_environment = environment

                # endregion

                # region Type violation

                for prop in symbol.properties:
                    # fmt: off
                    environment, instance = _copy_complete_environment_and_instance(
                        complete_environment=complete_environment,
                        path_to_instance_from_environment=
                        path_to_instance_from_environment
                    )
                    # fmt: on

                    type_anno = intermediate.beneath_optional(prop.type_annotation)

                    # noinspection PyUnusedLocal
                    unexpected_type = None  # type: Optional[str]
                    if isinstance(type_anno, intermediate.ListTypeAnnotation):
                        instance[
                            naming.json_property(prop.name)
                        ] = "Expected an array, but we put a string here"
                        unexpected_type = "String"
                    else:
                        instance[naming.json_property(prop.name)] = [
                            "Unexpected array here"
                        ]
                        unexpected_type = "Array"

                    assert unexpected_type is not None

                    pth = (
                        test_data_dir
                        / "Json"
                        / "Unexpected"
                        / "TypeViolation"
                        / naming.json_model_type(symbol.name)
                        / f"{naming.json_property(prop.name)}As{unexpected_type}.json"
                    )
                    pth.parent.mkdir(exist_ok=True, parents=True)
                    with pth.open("wt", encoding="utf-8") as fid:
                        json.dump(environment, fid, indent=2, sort_keys=True)

                # endregion

            # region Positive and negative pattern frozen_examples

            constraints_by_prop = constraints_by_class[symbol]
            for prop in symbol.properties:
                pattern_constraints = constraints_by_prop.patterns_by_property.get(
                    prop, None
                )

                if pattern_constraints is not None and len(pattern_constraints) == 1:
                    pattern_examples = frozen_examples_pattern.BY_PATTERN[
                        pattern_constraints[0].pattern
                    ]

                    for name, text in pattern_examples.positives.items():
                        # fmt: off
                        (
                            environment,
                            instance,
                        ) = _copy_minimal_environment_and_instance(
                            minimal_environment=minimal_environment,
                            path_to_instance_from_environment=
                            path_to_instance_from_environment,
                        )
                        # fmt: on

                        instance[naming.json_property(prop.name)] = text

                        pth = (
                            test_data_dir
                            / "Json"
                            / "Expected"
                            / naming.json_model_type(symbol.name)
                            / f"{naming.json_property(prop.name)}OverPatternExamples"
                            / f"{name}.json"
                        )
                        pth.parent.mkdir(exist_ok=True, parents=True)
                        with pth.open("wt", encoding="utf-8") as fid:
                            json.dump(environment, fid, indent=2, sort_keys=True)

                    for name, text in pattern_examples.negatives.items():
                        # fmt: off
                        (
                            environment,
                            instance,
                        ) = _copy_minimal_environment_and_instance(
                            minimal_environment=minimal_environment,
                            path_to_instance_from_environment=
                            path_to_instance_from_environment,
                        )
                        # fmt: on

                        instance[naming.json_property(prop.name)] = text

                        pth = (
                            test_data_dir
                            / "Json"
                            / "Unexpected"
                            / "PatternViolation"
                            / naming.json_model_type(symbol.name)
                            / f"{naming.json_property(prop.name)}"
                            / f"{name}.json"
                        )
                        pth.parent.mkdir(exist_ok=True, parents=True)
                        with pth.open("wt", encoding="utf-8") as fid:
                            json.dump(environment, fid, indent=2, sort_keys=True)
            # endregion

            # region Required violation

            for prop in symbol.properties:
                if isinstance(
                    prop.type_annotation, intermediate.OptionalTypeAnnotation
                ):
                    continue

                environment, instance = _copy_minimal_environment_and_instance(
                    minimal_environment=minimal_environment,
                    path_to_instance_from_environment=path_to_instance_from_environment,
                )

                del instance[naming.json_property(prop.name)]

                pth = (
                    test_data_dir
                    / "Json"
                    / "Unexpected"
                    / "RequiredViolation"
                    / naming.json_model_type(symbol.name)
                    / f"{naming.json_property(prop.name)}.json"
                )
                pth.parent.mkdir(exist_ok=True, parents=True)
                with pth.open("wt", encoding="utf-8") as fid:
                    json.dump(environment, fid, indent=2, sort_keys=True)

            # endregion

            # region Length violation

            for prop in symbol.properties:
                len_constraint = constraints_by_prop.len_constraints_by_property.get(
                    prop, None
                )

                if len_constraint is not None:
                    type_anno = intermediate.beneath_optional(prop.type_annotation)
                    if (
                        len_constraint.min_value is not None
                        and len_constraint.min_value > 0
                    ):
                        environment = None

                        # NOTE (mristin, 2022-05-15):
                        # We handle only a subset of cases here automatically since
                        # otherwise it would be too difficult to implement. The
                        # remainder of the cases needs to be implemented manually.
                        if isinstance(type_anno, intermediate.PrimitiveTypeAnnotation):
                            if type_anno.a_type is intermediate.PrimitiveType.STR:
                                # fmt: off
                                (
                                    environment,
                                    instance,
                                ) = _copy_minimal_environment_and_instance(
                                    minimal_environment=minimal_environment,
                                    path_to_instance_from_environment=
                                    path_to_instance_from_environment,
                                )
                                # fmt: on

                                instance[naming.json_property(prop.name)] = ""
                        elif (
                            isinstance(type_anno, intermediate.OurTypeAnnotation)
                            and isinstance(
                                type_anno.symbol, intermediate.ConstrainedPrimitive
                            )
                            and (
                                type_anno.symbol.constrainee
                                is intermediate.PrimitiveType.STR
                            )
                        ):
                            # fmt: off
                            (
                                environment,
                                instance,
                            ) = _copy_minimal_environment_and_instance(
                                minimal_environment=minimal_environment,
                                path_to_instance_from_environment=
                                path_to_instance_from_environment,
                            )
                            # fmt: on

                            instance[naming.json_property(prop.name)] = ""

                        elif isinstance(type_anno, intermediate.ListTypeAnnotation):
                            # fmt: off
                            (
                                environment,
                                instance,
                            ) = _copy_minimal_environment_and_instance(
                                minimal_environment=minimal_environment,
                                path_to_instance_from_environment=
                                path_to_instance_from_environment,
                            )
                            # fmt: on

                            instance[naming.json_property(prop.name)] = []

                        if environment is not None:
                            pth = (
                                test_data_dir
                                / "Json"
                                / "Unexpected"
                                / "MinLengthViolation"
                                / naming.json_model_type(symbol.name)
                                / f"{naming.json_property(prop.name)}.json"
                            )
                            pth.parent.mkdir(exist_ok=True, parents=True)
                            with pth.open("wt", encoding="utf-8") as fid:
                                json.dump(environment, fid, indent=2, sort_keys=True)

                    if len_constraint.max_value is not None:
                        environment = None

                        # NOTE (mristin, 2022-05-15):
                        # We handle only a subset of cases here automatically since
                        # otherwise it would be too difficult to implement. The
                        # remainder of the cases needs to be implemented manually.
                        #
                        # We also optimistically assume we do not break any patterns,
                        # invariants *etc.* If that is the case, you have to write
                        # manual generation code.

                        prop_name = naming.json_property(prop.name)
                        too_long_text = common.generate_long_string(
                            length=len_constraint.max_value + 1,
                            path_segments=(
                                path_to_instance_from_environment + [prop_name]
                            ),
                        )

                        if isinstance(type_anno, intermediate.PrimitiveTypeAnnotation):
                            if type_anno.a_type is intermediate.PrimitiveType.STR:
                                # fmt: off
                                (
                                    environment,
                                    instance,
                                ) = _copy_minimal_environment_and_instance(
                                    minimal_environment=minimal_environment,
                                    path_to_instance_from_environment=
                                    path_to_instance_from_environment,
                                )
                                # fmt: on

                                instance[prop_name] = too_long_text

                        elif (
                            isinstance(type_anno, intermediate.OurTypeAnnotation)
                            and isinstance(
                                type_anno.symbol, intermediate.ConstrainedPrimitive
                            )
                            and (
                                type_anno.symbol.constrainee
                                is intermediate.PrimitiveType.STR
                            )
                        ):
                            # fmt: off
                            (
                                environment,
                                instance,
                            ) = _copy_minimal_environment_and_instance(
                                minimal_environment=minimal_environment,
                                path_to_instance_from_environment=
                                path_to_instance_from_environment,
                            )
                            # fmt: on

                            instance[prop_name] = too_long_text

                        if environment is not None:
                            pth = (
                                test_data_dir
                                / "Json"
                                / "Unexpected"
                                / "MaxLengthViolation"
                                / naming.json_model_type(symbol.name)
                                / f"{naming.json_property(prop.name)}.json"
                            )
                            pth.parent.mkdir(exist_ok=True, parents=True)
                            with pth.open("wt", encoding="utf-8") as fid:
                                json.dump(environment, fid, indent=2, sort_keys=True)

            # endregion

            # region Break date-time with UTC with February 29th

            date_time_stamp_utc_symbol = symbol_table.must_find(
                Identifier("Date_time_stamp_UTC")
            )
            assert isinstance(
                date_time_stamp_utc_symbol, intermediate.ConstrainedPrimitive
            )

            for prop in symbol.properties:
                type_anno = intermediate.beneath_optional(prop.type_annotation)
                if (
                    isinstance(type_anno, intermediate.OurTypeAnnotation)
                    and type_anno.symbol is date_time_stamp_utc_symbol
                ):
                    # fmt: off
                    (
                        environment,
                        instance,
                    ) = _copy_minimal_environment_and_instance(
                        minimal_environment=minimal_environment,
                        path_to_instance_from_environment=
                        path_to_instance_from_environment,
                    )
                    # fmt: on

                    time_of_day = common.generate_time_of_day(
                        path_segments=(path_to_instance_from_environment + [prop.name])
                    )

                    instance[
                        naming.json_property(prop.name)
                    ] = f"2022-02-29T{time_of_day}Z"

                    pth = (
                        test_data_dir
                        / "Json"
                        / "Unexpected"
                        / "DateTimeStampUtcViolationOnFebruary29th"
                        / naming.json_model_type(symbol.name)
                        / f"{naming.json_property(prop.name)}.json"
                    )
                    pth.parent.mkdir(exist_ok=True, parents=True)
                    with pth.open("wt", encoding="utf-8") as fid:
                        json.dump(environment, fid, indent=2, sort_keys=True)

            # endregion

            # region Generate positive and negative frozen_examples of Property values

            if symbol.name in ["Property", "Range"]:
                data_type_def_xsd_symbol = symbol_table.must_find(
                    Identifier("Data_type_def_XSD")
                )
                assert isinstance(data_type_def_xsd_symbol, intermediate.Enumeration)

                for literal in data_type_def_xsd_symbol.literals:
                    examples = frozen_examples_xs_value.BY_VALUE_TYPE.get(
                        literal.value, None
                    )

                    if examples is None:
                        raise NotImplementedError(
                            f"The entry is missing "
                            f"in the {frozen_examples_xs_value.__name__!r} "
                            f"for the value type {literal.value!r}"
                        )

                    if symbol.name == "Property":
                        paths_values = [
                            (
                                (
                                    test_data_dir
                                    / "Json/Expected"
                                    / naming.json_model_type(symbol.name)
                                    / "OverValueExamples"
                                    / literal.name
                                    / f"{name}.json"
                                ),
                                value,
                            )
                            for name, value in examples.positives.items()
                        ] + [
                            (
                                (
                                    test_data_dir
                                    / "Json/Unexpected/"
                                    / naming.json_model_type(symbol.name)
                                    / "OverInvalidValueExamples"
                                    / literal.name
                                    / f"{name}.json"
                                ),
                                value,
                            )
                            for name, value in examples.negatives.items()
                        ]

                        for pth, value in paths_values:
                            # fmt: off
                            (
                                environment,
                                instance,
                            ) = _copy_minimal_environment_and_instance(
                                minimal_environment=minimal_environment,
                                path_to_instance_from_environment=
                                path_to_instance_from_environment,
                            )
                            # fmt: on

                            instance[
                                naming.json_property(Identifier("value_type"))
                            ] = literal.value
                            instance[naming.json_property(Identifier("value"))] = value
                            pth.parent.mkdir(exist_ok=True, parents=True)
                            with pth.open("wt", encoding="utf-8") as fid:
                                json.dump(environment, fid, indent=2, sort_keys=True)

                    elif symbol.name == "Range":
                        paths_values = [
                            (
                                (
                                    test_data_dir
                                    / "Json/Expected"
                                    / naming.json_model_type(symbol.name)
                                    / "OverMinMaxExamples"
                                    / literal.name
                                    / f"{name}.json"
                                ),
                                value,
                            )
                            for name, value in examples.positives.items()
                        ] + [
                            (
                                (
                                    test_data_dir
                                    / "Json/Unexpected/"
                                    / naming.json_model_type(symbol.name)
                                    / "OverInvalidMinMaxExamples"
                                    / literal.name
                                    / f"{name}.json"
                                ),
                                value,
                            )
                            for name, value in examples.negatives.items()
                        ]

                        for pth, value in paths_values:
                            # fmt: off
                            (
                                environment,
                                instance,
                            ) = _copy_minimal_environment_and_instance(
                                minimal_environment=minimal_environment,
                                path_to_instance_from_environment=
                                path_to_instance_from_environment,
                            )
                            # fmt: on

                            instance[
                                naming.json_property(Identifier("value_type"))
                            ] = literal.value
                            instance[naming.json_property(Identifier("min"))] = value
                            instance[naming.json_property(Identifier("max"))] = value
                            pth.parent.mkdir(exist_ok=True, parents=True)
                            with pth.open("wt", encoding="utf-8") as fid:
                                json.dump(environment, fid, indent=2, sort_keys=True)

                    else:
                        raise AssertionError(f"Unexpected {symbol.name=}")

            # endregion

    # BEFORE-RELEASE (mristin, 2022-06-19):
    # Manually write Json/Unexpected/ConstraintViolation/{class name}/
    # {describe how we break it somehow}.json


def main() -> None:
    """Execute the main routine."""
    this_path = pathlib.Path(os.path.realpath(__file__))
    test_data_dir = this_path.parent.parent / "test_data"

    generate(test_data_dir=test_data_dir)


if __name__ == "__main__":
    main()
