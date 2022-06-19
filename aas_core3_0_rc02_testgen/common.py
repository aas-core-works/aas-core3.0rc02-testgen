"""Provide common methods for generation of data in different formats."""

import hashlib
import io
import pathlib
import re
from typing import MutableMapping, Tuple, Sequence, Union, List

import aas_core_meta.v3rc2
import aas_core_codegen.common
import aas_core_codegen.run
import aas_core_codegen.parse

from aas_core_codegen import intermediate, infer_for_schema
from icontract import ensure, require


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


HEX_RE = re.compile(r"[a-fA-F0-9]+")


@ensure(lambda result: HEX_RE.fullmatch(result))
def hash_path(path_segments: Sequence[Union[str, int]]) -> str:
    """Hash the given path to a value in the model."""
    hsh = hashlib.md5()
    hsh.update(("".join(repr(segment) for segment in path_segments)).encode("utf-8"))
    return hsh.hexdigest()[:8]


def posix_path(path_segments: Sequence[Union[str, int]]) -> pathlib.PurePosixPath:
    """Make a POSIX path out of the path segments."""
    pth = pathlib.PurePosixPath("/")
    for segment in path_segments:
        pth = pth / str(segment)

    return pth


@require(lambda length: length > 0)
@ensure(lambda result, length: len(result) == length)
def generate_long_string(
    length: int,
    path_segments: List[Union[int, str]],
) -> str:
    """
    Generate a string longer than the ``length``.

    >>> generate_long_string(2, ['some', 3, 'path'])
    'x9'

    >>> generate_long_string(9, ['some', 3, 'path'])
    'x99ef1573'

    >>> generate_long_string(10, ['some', 3, 'path'])
    'x99ef15730'

    >>> generate_long_string(15, ['some', 3, 'path'])
    'x99ef1573012345'

    >>> generate_long_string(20, ['some', 3, 'path'])
    'x99ef157301234567890'

    >>> generate_long_string(21, ['some', 3, 'path'])
    'x99ef1573012345678901'
    """
    prefix = f"x{hash_path(path_segments=path_segments)}"
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


def generate_time_of_day(path_segments: List[Union[int, str]]) -> str:
    """Generate a random time of the day based on the path to the value."""
    hsh = hash_path(path_segments=path_segments)

    hsh_as_int = int(hsh, base=16)

    remainder = hsh_as_int
    hours = (remainder // 3600) % 24
    remainder = remainder % 3600
    minutes = (remainder // 60) % 60
    seconds = remainder % 60

    fraction = hsh_as_int % 1000000

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{fraction}"
