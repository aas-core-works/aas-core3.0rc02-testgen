"""Generate test data in JSON for the meta-model V3RC02."""
import base64
import collections
import collections.abc
import json
import os
import pathlib
from typing import (
    Union,
    OrderedDict,
    List,
    Any,
    MutableMapping,
    Sequence,
)

import aas_core_codegen.common
import aas_core_codegen.naming
from aas_core_codegen import intermediate, infer_for_schema
from aas_core_codegen.common import Identifier
from icontract import ensure, require

from aas_core3_0_rc02_testgen import common, generation, ontology


@ensure(lambda result: not result.is_absolute())
def _relative_path(test_case: generation.CaseUnion) -> pathlib.Path:
    """Generate the relative path based on the test case."""
    assert test_case.__class__.__name__.startswith("Case")

    base_pth = pathlib.Path("Json")

    if isinstance(test_case, generation.CaseMinimal):
        cls_name = aas_core_codegen.naming.json_model_type(test_case.cls.name)

        return base_pth / "Expected" / cls_name / "minimal.json"

    elif isinstance(test_case, generation.CaseComplete):
        cls_name = aas_core_codegen.naming.json_model_type(test_case.cls.name)

        return base_pth / "Expected" / cls_name / "complete.json"

    elif isinstance(test_case, generation.CaseTypeViolation):
        cls_name = aas_core_codegen.naming.json_model_type(test_case.cls.name)
        prop_name = aas_core_codegen.naming.json_property(test_case.property_name)

        return (
            base_pth / "Unexpected" / "TypeViolation" / cls_name / f"{prop_name}.json"
        )

    elif isinstance(test_case, generation.CasePositivePatternExample):
        cls_name = aas_core_codegen.naming.json_model_type(test_case.cls.name)
        prop_name = aas_core_codegen.naming.json_property(test_case.property_name)

        return (
            base_pth
            / "Expected"
            / cls_name
            / f"{prop_name}OverPatternExamples"
            / f"{test_case.example_name}.json"
        )

    elif isinstance(test_case, generation.CaseNegativePatternExample):
        cls_name = aas_core_codegen.naming.json_model_type(test_case.cls.name)
        prop_name = aas_core_codegen.naming.json_property(test_case.property_name)

        return (
            base_pth
            / "Unexpected"
            / "PatternViolation"
            / cls_name
            / prop_name
            / f"{test_case.example_name}.json"
        )

    elif isinstance(test_case, generation.CaseRequiredViolation):
        cls_name = aas_core_codegen.naming.json_model_type(test_case.cls.name)
        prop_name = aas_core_codegen.naming.json_property(test_case.property_name)

        return (
            base_pth
            / "Unexpected"
            / "RequiredViolation"
            / cls_name
            / f"{prop_name}.json"
        )

    elif isinstance(test_case, generation.CaseMinLengthViolation):
        cls_name = aas_core_codegen.naming.json_model_type(test_case.cls.name)
        prop_name = aas_core_codegen.naming.json_property(test_case.property_name)

        return (
            base_pth
            / "Unexpected"
            / "MinLengthViolation"
            / cls_name
            / f"{prop_name}.json"
        )

    elif isinstance(test_case, generation.CaseMaxLengthViolation):
        cls_name = aas_core_codegen.naming.json_model_type(test_case.cls.name)
        prop_name = aas_core_codegen.naming.json_property(test_case.property_name)

        return (
            base_pth
            / "Unexpected"
            / "MaxLengthViolation"
            / cls_name
            / f"{prop_name}.json"
        )

    elif isinstance(test_case, generation.CaseDateTimeStampUtcViolationOnFebruary29th):
        cls_name = aas_core_codegen.naming.json_model_type(test_case.cls.name)
        prop_name = aas_core_codegen.naming.json_property(test_case.property_name)

        return (
            base_pth
            / "Unexpected"
            / "DateTimeStampUtcViolationOnFebruary29th"
            / cls_name
            / f"{prop_name}.json"
        )

    elif isinstance(test_case, generation.CasePositiveValueExample):
        cls_name = aas_core_codegen.naming.json_model_type(test_case.cls.name)

        return (
            base_pth
            / "Expected"
            / cls_name
            / "OverValueExamples"
            / test_case.data_type_def_literal.name
            / f"{test_case.example_name}.json"
        )

    elif isinstance(test_case, generation.CaseNegativeValueExample):
        cls_name = aas_core_codegen.naming.json_model_type(test_case.cls.name)

        return (
            base_pth
            / "Unexpected"
            / "InvalidValueExamples"
            / cls_name
            / test_case.data_type_def_literal.name
            / f"{test_case.example_name}.json"
        )

    elif isinstance(test_case, generation.CasePositiveMinMaxExample):
        cls_name = aas_core_codegen.naming.json_model_type(test_case.cls.name)

        return (
            base_pth
            / "Expected"
            / cls_name
            / "OverMinMaxExamples"
            / test_case.data_type_def_literal.name
            / f"{test_case.example_name}.json"
        )

    elif isinstance(test_case, generation.CaseNegativeMinMaxExample):
        cls_name = aas_core_codegen.naming.json_model_type(test_case.cls.name)

        return (
            base_pth
            / "Unexpected"
            / "InvalidMinMaxExamples"
            / cls_name
            / test_case.data_type_def_literal.name
            / f"{test_case.example_name}.json"
        )

    elif isinstance(test_case, generation.CaseEnumViolation):
        enum_name = aas_core_codegen.naming.json_model_type(test_case.enum.name)
        cls_name = aas_core_codegen.naming.json_model_type(test_case.cls.name)
        prop_name = aas_core_codegen.naming.json_property(test_case.prop.name)

        return (
            base_pth
            / "Unexpected"
            / "EnumViolation"
            / enum_name
            / f"{cls_name}-{prop_name}.json"
        )

    elif isinstance(test_case, generation.CasePositiveManual):
        cls_name = aas_core_codegen.naming.json_model_type(test_case.cls.name)

        return base_pth / "Expected" / cls_name / f"{test_case.name}.json"

    elif isinstance(test_case, generation.CaseConstraintViolation):
        cls_name = aas_core_codegen.naming.json_model_type(test_case.cls.name)

        return (
            base_pth
            / "Unexpected"
            / "ConstraintViolation"
            / cls_name
            / f"{test_case.name}.json"
        )

    else:
        aas_core_codegen.common.assert_never(test_case)


class _Serializer:
    """Serialize an environment to a JSON object."""

    def __init__(self, symbol_table: intermediate.SymbolTable) -> None:
        """Initialize with the given values."""
        self.symbol_table = symbol_table

    @require(lambda instance: instance.model_type == "Environment")
    def serialize_environment(
        self, instance: generation.Instance
    ) -> OrderedDict[str, Any]:
        """Serialize the ``environment`` to a JSON-able object."""
        return self._serialize_instance(instance=instance)

    def _serialize_value(self, value: generation.ValueUnion) -> Any:
        if isinstance(value, generation.PrimitiveValueTuple):
            return self._serialize_primitive(value)
        elif isinstance(value, generation.Instance):
            return self._serialize_instance(value)
        elif isinstance(value, generation.ListOfInstances):
            return self._serialize_list_of_instances(value)
        else:
            aas_core_codegen.common.assert_never(value)

    def _serialize_primitive(
        self, value: generation.PrimitiveValueUnion
    ) -> Union[bool, int, float, str]:
        if isinstance(value, bytearray):
            return base64.b64encode(value).decode(encoding="ascii")
        else:
            return value

    def _serialize_instance(
        self, instance: generation.Instance
    ) -> OrderedDict[str, Any]:
        jsonable = collections.OrderedDict()  # type: OrderedDict[str, Any]

        for prop_name, prop_value in instance.properties.items():
            jsonable[
                aas_core_codegen.naming.json_property(Identifier(prop_name))
            ] = self._serialize_value(prop_value)

        cls = self.symbol_table.must_find(instance.model_type)
        assert isinstance(cls, (intermediate.AbstractClass, intermediate.ConcreteClass))

        if cls.serialization is not None and cls.serialization.with_model_type:
            jsonable["modelType"] = aas_core_codegen.naming.json_model_type(
                instance.model_type
            )

        return jsonable

    def _serialize_list_of_instances(
        self, list_of_instances: generation.ListOfInstances
    ) -> List[OrderedDict[str, Any]]:
        return [self._serialize_instance(value) for value in list_of_instances.values]


def to_json_path_segments(
    path_segments: Sequence[Union[int, str]]
) -> List[Union[int, Identifier]]:
    """Convert the path segments from the case of the meta-model to JSON casing."""
    result = []  # type: List[Union[int, Identifier]]
    for path_segment in path_segments:
        if isinstance(path_segment, int):
            result.append(path_segment)

        elif isinstance(path_segment, str):
            result.append(
                aas_core_codegen.naming.json_property(Identifier(path_segment))
            )

        else:
            aas_core_codegen.common.assert_never(path_segment)

    return result


def dereference(
    start_object: MutableMapping[str, Any],
    path_segments: Sequence[Union[int, Identifier]],
) -> Any:
    """Get the JSON value following the ``path_segments`` from ``start_object``"""
    cursor = start_object
    for path_segment in path_segments:
        if isinstance(path_segment, str):
            if not isinstance(cursor, collections.abc.MutableMapping):
                raise ValueError(
                    f"Could not access the property {path_segment!r} "
                    f"in a JSON non-object: {cursor}; "
                    f"{start_object=}, {path_segments=}"
                )

            cursor = cursor[path_segment]
        elif isinstance(path_segment, int):
            if not isinstance(cursor, collections.abc.Sequence):
                raise ValueError(
                    f"Could not access the index {path_segment!r} "
                    f"in a JSON non-array: {cursor}; "
                    f"{start_object=}, {path_segments=}"
                )

            cursor = cursor[path_segment]
        else:
            aas_core_codegen.common.assert_never(path_segment)

    return cursor


def _generate_null_violations(
    test_data_dir: pathlib.Path,
    symbol_table: intermediate.SymbolTable,
    constraints_by_class: MutableMapping[
        intermediate.ClassUnion, infer_for_schema.ConstraintsByProperty
    ],
    class_graph: ontology.ClassGraph,
    handyman: generation.Handyman,
    serializer: _Serializer,
) -> None:
    """Generate the files with the properties and items set to ``null``."""
    for symbol in symbol_table.symbols:
        if not isinstance(symbol, intermediate.ConcreteClass):
            continue

        if symbol.name == Identifier("Event_payload"):
            # NOTE (mristin, 2022-06-25):
            # Event payload can not be reached from an environment.
            continue

        (
            minimal_env,
            path_segments,
        ) = generation.generate_minimal_instance_in_minimal_environment(
            cls=symbol,
            class_graph=class_graph,
            constraints_by_class=constraints_by_class,
            symbol_table=symbol_table,
        )

        instance = generation.dereference(
            environment=minimal_env, path_segments=path_segments
        )

        handyman.fix_instance(instance=instance, path_segments=path_segments)

        replicator = generation.EnvironmentInstanceReplicator(
            environment=minimal_env, path_to_instance_from_environment=path_segments
        )

        for prop in symbol.properties:
            type_anno = intermediate.beneath_optional(prop.type_annotation)

            if isinstance(prop.type_annotation, intermediate.OptionalTypeAnnotation):
                # NOTE (mristin, 2022-06-25):
                # We still want to check all the list properties, both optional and
                # required.
                if not isinstance(type_anno, intermediate.ListTypeAnnotation):
                    continue

            env, instance = replicator.replicate()

            jsonable_env = serializer.serialize_environment(instance=env)

            jsonable_instance = dereference(
                start_object=jsonable_env,
                path_segments=to_json_path_segments(path_segments=path_segments),
            )

            prop_name_json = aas_core_codegen.naming.json_property(prop.name)

            if not isinstance(
                    prop.type_annotation, intermediate.OptionalTypeAnnotation):

                jsonable_instance[prop_name_json] = None

                pth = (
                    test_data_dir
                    / "Json/Unexpected/NullViolation"
                    / aas_core_codegen.naming.json_model_type(symbol.name)
                    / f"{prop_name_json}_value.json"
                )

                pth.parent.mkdir(parents=True, exist_ok=True)

                with pth.open("wt") as fid:
                    json.dump(jsonable_env, fid, indent=2, sort_keys=True)

            if isinstance(type_anno, intermediate.ListTypeAnnotation):
                jsonable_instance[prop_name_json] = [None]

                pth = (
                    test_data_dir
                    / "Json/Unexpected/NullViolation"
                    / aas_core_codegen.naming.json_model_type(symbol.name)
                    / f"{prop_name_json}_item.json"
                )

                pth.parent.mkdir(parents=True, exist_ok=True)

                with pth.open("wt") as fid:
                    json.dump(jsonable_env, fid, indent=2, sort_keys=True)


def generate(test_data_dir: pathlib.Path) -> None:
    """Generate the JSON files."""
    (
        symbol_table,
        constraints_by_class,
    ) = common.load_symbol_table_and_infer_constraints_for_schema()

    class_graph = ontology.compute_class_graph(symbol_table=symbol_table)

    serializer = _Serializer(symbol_table=symbol_table)

    for test_case in generation.generate(
        symbol_table=symbol_table,
        constraints_by_class=constraints_by_class,
        class_graph=class_graph,
    ):
        relative_pth = _relative_path(test_case=test_case)
        jsonable = serializer.serialize_environment(test_case.environment)

        pth = test_data_dir / relative_pth

        parent = pth.parent
        if not parent.exists():
            parent.mkdir(parents=True)

        with pth.open("wt") as fid:
            json.dump(jsonable, fid, indent=2, sort_keys=True)

    handyman = generation.Handyman(
        symbol_table=symbol_table, constraints_by_class=constraints_by_class
    )

    _generate_null_violations(
        test_data_dir=test_data_dir,
        symbol_table=symbol_table,
        constraints_by_class=constraints_by_class,
        class_graph=class_graph,
        handyman=handyman,
        serializer=serializer,
    )


def main() -> None:
    """Execute the main routine."""
    this_path = pathlib.Path(os.path.realpath(__file__))
    test_data_dir = this_path.parent.parent / "test_data"

    generate(test_data_dir=test_data_dir)


if __name__ == "__main__":
    main()
