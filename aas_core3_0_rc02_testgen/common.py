"""Provide common methods for generation of data in different formats."""

import io
import pathlib
import re
from typing import MutableMapping, Tuple

import aas_core_codegen.common
import aas_core_codegen.parse
import aas_core_codegen.run
import aas_core_meta.v3rc2
from aas_core_codegen import intermediate, infer_for_schema

from aas_core3_0_rc02_testgen import generation


def load_symbol_table_and_infer_constraints_for_schema() -> Tuple[
    intermediate.SymbolTable,
    MutableMapping[intermediate.ClassUnion, infer_for_schema.ConstraintsByProperty],
]:
    """
    Load the symbol table from the meta-model and infer the schema constraints.

    These constraints might not be sufficient to generate *some* of the instances.
    Further constraints in form of invariants might apply which are not represented
    in the schema constraints. However, this will help us cover *many* classes of the
    meta-model and spare us the work of manually writing many generators.
    """
    model_path = pathlib.Path(aas_core_meta.v3rc2.__file__)
    assert model_path.exists() and model_path.is_file(), model_path

    text = model_path.read_text(encoding="utf-8")

    atok, parse_exception = aas_core_codegen.parse.source_to_atok(source=text)
    if parse_exception:
        if isinstance(parse_exception, SyntaxError):
            raise RuntimeError(
                f"Failed to parse the meta-model {model_path}: "
                f"invalid syntax at line {parse_exception.lineno}\n"
            )
        else:
            raise RuntimeError(
                f"Failed to parse the meta-model {model_path}: " f"{parse_exception}\n"
            )

    assert atok is not None

    import_errors = aas_core_codegen.parse.check_expected_imports(atok=atok)
    if import_errors:
        writer = io.StringIO()
        aas_core_codegen.run.write_error_report(
            message="One or more unexpected imports in the meta-model",
            errors=import_errors,
            stderr=writer,
        )

        raise RuntimeError(writer.getvalue())

    lineno_columner = aas_core_codegen.common.LinenoColumner(atok=atok)

    parsed_symbol_table, error = aas_core_codegen.parse.atok_to_symbol_table(atok=atok)
    if error is not None:
        writer = io.StringIO()
        aas_core_codegen.run.write_error_report(
            message=f"Failed to construct the symbol table from {model_path}",
            errors=[lineno_columner.error_message(error)],
            stderr=writer,
        )

        raise RuntimeError(writer.getvalue())

    assert parsed_symbol_table is not None

    ir_symbol_table, error = intermediate.translate(
        parsed_symbol_table=parsed_symbol_table,
        atok=atok,
    )
    if error is not None:
        writer = io.StringIO()
        aas_core_codegen.run.write_error_report(
            message=f"Failed to translate the parsed symbol table "
            f"to intermediate symbol table "
            f"based on {model_path}",
            errors=[lineno_columner.error_message(error)],
            stderr=writer,
        )

        raise RuntimeError(writer.getvalue())

    assert ir_symbol_table is not None

    (
        constraints_by_class,
        inference_errors,
    ) = aas_core_codegen.infer_for_schema.infer_constraints_by_class(
        symbol_table=ir_symbol_table
    )

    if inference_errors is not None:
        writer = io.StringIO()
        aas_core_codegen.run.write_error_report(
            message=f"Failed to infer the constraints for the schema "
            f"based on {model_path}",
            errors=[lineno_columner.error_message(error) for error in inference_errors],
            stderr=writer,
        )

        raise RuntimeError(writer.getvalue())

    assert constraints_by_class is not None
    (
        constraints_by_class,
        merge_error,
    ) = aas_core_codegen.infer_for_schema.merge_constraints_with_ancestors(
        symbol_table=ir_symbol_table, constraints_by_class=constraints_by_class
    )

    if merge_error is not None:
        writer = io.StringIO()
        aas_core_codegen.run.write_error_report(
            message=f"Failed to infer the constraints for the schema "
            f"based on {model_path}",
            errors=[lineno_columner.error_message(merge_error)],
            stderr=writer,
        )

        raise RuntimeError(writer.getvalue())

    assert constraints_by_class is not None

    return ir_symbol_table, constraints_by_class


_XML_1_0_TEXT_RE = re.compile(
    r"^[\x09\x0A\x0D\x20-\uD7FF\uE000-\uFFFD\U00010000-\U0010FFFF]*$"
)


def conforms_to_xml_1_0(value: generation.ValueUnion) -> bool:
    """Check recursively that the value conforms to XML 1.0."""
    if isinstance(value, generation.PrimitiveValueTuple):
        if isinstance(value, str):
            return _XML_1_0_TEXT_RE.match(value) is not None
        else:
            return True
    elif isinstance(value, generation.Instance):
        # noinspection PyTypeChecker
        for prop_value in value.properties.values():
            if not conforms_to_xml_1_0(prop_value):
                return False

        return True
    elif isinstance(value, generation.ListOfInstances):
        for instance in value.values:
            if not conforms_to_xml_1_0(instance):
                return False

        return True

    else:
        aas_core_codegen.common.assert_never(value)
