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
)

import aas_core_codegen.common
import aas_core_codegen.naming
from aas_core_codegen import intermediate
from aas_core_codegen.common import Identifier
from icontract import ensure, require

from aas_core3_0_rc02_testgen import common, generation


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
            / cls_name
            / "OverInvalidMinMaxExamples"
            / test_case.data_type_def_literal.name
            / f"{test_case.example_name}.json"
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


def generate(test_data_dir: pathlib.Path) -> None:
    """Generate the JSON files."""
    (
        symbol_table,
        constraints_by_class,
    ) = common.load_symbol_table_and_infer_constraints_for_schema()

    serializer = _Serializer(symbol_table=symbol_table)

    for test_case in generation.generate(
        symbol_table=symbol_table, constraints_by_class=constraints_by_class
    ):
        relative_pth = _relative_path(test_case=test_case)
        jsonable = serializer.serialize_environment(test_case.environment)

        pth = test_data_dir / relative_pth

        parent = pth.parent
        if not parent.exists():
            parent.mkdir(parents=True)

        with pth.open("wt") as fid:
            json.dump(jsonable, fid, indent=2, sort_keys=True)


def main() -> None:
    """Execute the main routine."""
    this_path = pathlib.Path(os.path.realpath(__file__))
    test_data_dir = this_path.parent.parent / "test_data"

    generate(test_data_dir=test_data_dir)


if __name__ == "__main__":
    main()
