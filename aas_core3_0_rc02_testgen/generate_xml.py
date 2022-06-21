"""Generate test data in XML for the meta-model V3RC02."""
import base64
import math
import os
import pathlib
import re
from typing import (
    List,
    Optional,
)
from xml.dom import minidom

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

    base_pth = pathlib.Path("Xml")

    if isinstance(test_case, generation.CaseMinimal):
        cls_name = aas_core_codegen.naming.xml_class_name(test_case.cls.name)

        return base_pth / "Expected" / cls_name / "minimal.xml"

    elif isinstance(test_case, generation.CaseComplete):
        cls_name = aas_core_codegen.naming.xml_class_name(test_case.cls.name)

        return base_pth / "Expected" / cls_name / "complete.xml"

    elif isinstance(test_case, generation.CaseTypeViolation):
        cls_name = aas_core_codegen.naming.xml_class_name(test_case.cls.name)
        prop_name = aas_core_codegen.naming.xml_property(test_case.property_name)

        return base_pth / "Unexpected" / "TypeViolation" / cls_name / f"{prop_name}.xml"

    elif isinstance(test_case, generation.CasePositivePatternExample):
        cls_name = aas_core_codegen.naming.xml_class_name(test_case.cls.name)
        prop_name = aas_core_codegen.naming.xml_property(test_case.property_name)

        return (
                base_pth
                / "Expected"
                / cls_name
                / f"{prop_name}OverPatternExamples"
                / f"{test_case.example_name}.xml"
        )

    elif isinstance(test_case, generation.CaseNegativePatternExample):
        cls_name = aas_core_codegen.naming.xml_class_name(test_case.cls.name)
        prop_name = aas_core_codegen.naming.xml_property(test_case.property_name)

        return (
                base_pth
                / "Unexpected"
                / "PatternViolation"
                / cls_name
                / prop_name
                / f"{test_case.example_name}.xml"
        )

    elif isinstance(test_case, generation.CaseRequiredViolation):
        cls_name = aas_core_codegen.naming.xml_class_name(test_case.cls.name)
        prop_name = aas_core_codegen.naming.xml_property(test_case.property_name)

        return (
                base_pth
                / "Unexpected"
                / "RequiredViolation"
                / cls_name
                / f"{prop_name}.xml"
        )

    elif isinstance(test_case, generation.CaseMinLengthViolation):
        cls_name = aas_core_codegen.naming.xml_class_name(test_case.cls.name)
        prop_name = aas_core_codegen.naming.xml_property(test_case.property_name)

        return (
                base_pth
                / "Unexpected"
                / "MinLengthViolation"
                / cls_name
                / f"{prop_name}.xml"
        )

    elif isinstance(test_case, generation.CaseMaxLengthViolation):
        cls_name = aas_core_codegen.naming.xml_class_name(test_case.cls.name)
        prop_name = aas_core_codegen.naming.xml_property(test_case.property_name)

        return (
                base_pth
                / "Unexpected"
                / "MaxLengthViolation"
                / cls_name
                / f"{prop_name}.xml"
        )

    elif isinstance(test_case, generation.CaseDateTimeStampUtcViolationOnFebruary29th):
        cls_name = aas_core_codegen.naming.xml_class_name(test_case.cls.name)
        prop_name = aas_core_codegen.naming.xml_property(test_case.property_name)

        return (
                base_pth
                / "Unexpected"
                / "DateTimeStampUtcViolationOnFebruary29th"
                / cls_name
                / f"{prop_name}.xml"
        )

    elif isinstance(test_case, generation.CasePositiveValueExample):
        cls_name = aas_core_codegen.naming.xml_class_name(test_case.cls.name)

        return (
                base_pth
                / "Expected"
                / cls_name
                / "OverValueExamples"
                / test_case.data_type_def_literal.name
                / f"{test_case.example_name}.xml"
        )

    elif isinstance(test_case, generation.CaseNegativeValueExample):
        cls_name = aas_core_codegen.naming.xml_class_name(test_case.cls.name)

        return (
                base_pth
                / "Unexpected"
                / "InvalidValueExamples"
                / cls_name
                / test_case.data_type_def_literal.name
                / f"{test_case.example_name}.xml"
        )

    elif isinstance(test_case, generation.CasePositiveMinMaxExample):
        cls_name = aas_core_codegen.naming.xml_class_name(test_case.cls.name)

        return (
                base_pth
                / "Expected"
                / cls_name
                / "OverMinMaxExamples"
                / test_case.data_type_def_literal.name
                / f"{test_case.example_name}.xml"
        )

    elif isinstance(test_case, generation.CaseNegativeMinMaxExample):
        cls_name = aas_core_codegen.naming.xml_class_name(test_case.cls.name)

        return (
                base_pth
                / "Unexpected"
                / "InvalidMinMaxExamples"
                / cls_name
                / test_case.data_type_def_literal.name
                / f"{test_case.example_name}.xml"
        )

    else:
        aas_core_codegen.common.assert_never(test_case)


class _Serializer:
    """Serialize an environment to an XML element."""

    def __init__(self, symbol_table: intermediate.SymbolTable) -> None:
        """Initialize with the given values."""
        self.symbol_table = symbol_table

    @require(lambda instance: instance.model_type == "Environment")
    def serialize_environment(self, instance: generation.Instance) -> minidom.Element:
        """Serialize the ``environment`` to a JSON-able object."""
        impl = minidom.getDOMImplementation()
        assert impl is not None
        doc = impl.createDocument(
            namespaceURI="http://www.admin-shell.io/aas/3/0/RC02",
            qualifiedName="environment",
            doctype=None,
        )

        root = doc.documentElement

        # noinspection SpellCheckingInspection
        root.setAttribute("xmlns", "http://www.admin-shell.io/aas/3/0/RC02")

        sequence = self._serialize_instance(instance=instance, doc=doc)
        for node in sequence:
            root.appendChild(node)

        assert isinstance(root, minidom.Element)
        return root

    # noinspection PyMethodMayBeStatic
    def _serialize_primitive(
            self, value: generation.PrimitiveValueUnion, doc: minidom.Document
    ) -> minidom.Text:
        text = None  # type: Optional[str]

        if isinstance(value, bytearray):
            text = base64.b64encode(value).decode(encoding="ascii")
        elif isinstance(value, bool):
            text = "true" if value else "false"
        elif isinstance(value, int):
            text = str(value)
        elif isinstance(value, float):
            if math.isnan(value):
                text = "NaN"
            else:
                if math.isinf(value):
                    text = "INF" if value >= 0 else "-INF"
                else:
                    # The 17 digits are necessary for the round trip.
                    # See: https://stackoverflow.com/questions/32685380/float-to-string-round-trip-test
                    text = f"{value:.17g}"
        elif isinstance(value, str):
            text = value
        else:
            aas_core_codegen.common.assert_never(value)

        assert text is not None
        text_node = doc.createTextNode(text)  # type: ignore
        assert isinstance(text_node, minidom.Text)
        return text_node

    def _serialize_instance(
            self, instance: generation.Instance, doc: minidom.Document
    ) -> List[minidom.Element]:
        sequence = []  # type: List[minidom.Element]

        # NOTE (mristin, 2022-06-20):
        # We need to re-order the sequence so that it strictly follows the order of the
        # properties in the meta-model. Otherwise, the XML schema will complain.

        cls = self.symbol_table.must_find(instance.model_type)
        assert isinstance(cls, intermediate.ConcreteClass)

        order_map = {prop.name: i for i, prop in enumerate(cls.properties)}

        indices_prop_names = [
            (order_map[Identifier(prop_name)], prop_name)
            if prop_name in order_map
            else (math.inf, prop_name)
            for prop_name in instance.properties
        ]

        indices_prop_names.sort()

        prop_names = [prop_name for _, prop_name in indices_prop_names]

        for prop_name in prop_names:
            prop_value = instance.properties[prop_name]

            prop_element = doc.createElement(
                aas_core_codegen.naming.xml_property(Identifier(prop_name))
            )

            if isinstance(prop_value, generation.PrimitiveValueTuple):
                text_node = self._serialize_primitive(prop_value, doc)
                prop_element.appendChild(text_node)

            elif isinstance(prop_value, generation.Instance):
                subsequence = self._serialize_instance(prop_value, doc)

                a_cls = self.symbol_table.must_find(prop_value.model_type)
                assert isinstance(
                    a_cls, (intermediate.AbstractClass, intermediate.ConcreteClass)
                )

                if (
                        a_cls.serialization is not None
                        and a_cls.serialization.with_model_type
                ):
                    model_type_node = doc.createElement(
                        aas_core_codegen.naming.xml_class_name(prop_value.model_type)
                    )
                    for node in subsequence:
                        model_type_node.appendChild(node)

                    prop_element.appendChild(model_type_node)
                else:
                    for node in subsequence:
                        prop_element.appendChild(node)

            elif isinstance(prop_value, generation.ListOfInstances):
                subsequence = self._serialize_list_of_instances(prop_value, doc)

                for node in subsequence:
                    prop_element.appendChild(node)
            else:
                aas_core_codegen.common.assert_never(prop_value)

            sequence.append(prop_element)

        return sequence

    def _serialize_list_of_instances(
            self, list_of_instances: generation.ListOfInstances, doc: minidom.Document
    ) -> List[minidom.Element]:
        sequence = []  # type: List[minidom.Element]

        for value in list_of_instances.values:
            model_type_node = doc.createElement(
                aas_core_codegen.naming.xml_class_name(value.model_type)
            )

            subsequence = self._serialize_instance(instance=value, doc=doc)

            for node in subsequence:
                model_type_node.appendChild(node)

            sequence.append(model_type_node)

        return sequence


_XML_1_0_TEXT_RE = re.compile(
    r'^[\x09\x0A\x0D\x20-\uD7FF\uE000-\uFFFD\U00010000-\U0010FFFF]*$')


def _conforms_to_xml_1_0(value: generation.ValueUnion) -> bool:
    """Check recursively that the value conforms to XML 1.0."""
    if isinstance(value, generation.PrimitiveValueTuple):
        if isinstance(value, str):
            return _XML_1_0_TEXT_RE.match(value) is not None
        else:
            return True
    elif isinstance(value, generation.Instance):
        for prop_value in value.properties.values():
            if not _conforms_to_xml_1_0(prop_value):
                return False

        return True
    elif isinstance(value, generation.ListOfInstances):
        for instance in value.values:
            if not _conforms_to_xml_1_0(instance):
                return False

        return True

    else:
        aas_core_codegen.common.assert_never(value)


def generate(test_data_dir: pathlib.Path) -> None:
    """Generate the XML files."""
    (
        symbol_table,
        constraints_by_class,
    ) = common.load_symbol_table_and_infer_constraints_for_schema()

    serializer = _Serializer(symbol_table=symbol_table)

    for test_case in generation.generate(
            symbol_table=symbol_table, constraints_by_class=constraints_by_class
    ):
        relative_pth = _relative_path(test_case=test_case)

        pth = test_data_dir / relative_pth

        if not _conforms_to_xml_1_0(test_case.environment):
            print(
                f"The test case can not be represented in XML 1.0, skipping: {relative_pth}")
            continue

        parent = pth.parent
        if not parent.exists():
            parent.mkdir(parents=True)

        element = serializer.serialize_environment(test_case.environment)

        pth.write_text(element.toprettyxml(), encoding="utf-8")


def main() -> None:
    """Execute the main routine."""
    this_path = pathlib.Path(os.path.realpath(__file__))
    test_data_dir = this_path.parent.parent / "test_data"

    generate(test_data_dir=test_data_dir)


if __name__ == "__main__":
    main()