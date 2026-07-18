"""Tree-sitter query strings — ported from xai-codebase-graph languages/*.rs.

Capture conventions (builder.rs extract_symbols_fast_inline):
  name.definition.*  → definitions
  name.reference.*   → references
  alias.original + alias.name → SymbolAlias (same match)
"""

from __future__ import annotations

# languages/python.rs
PYTHON_QUERY = r"""
(class_definition
    name: (identifier) @name.definition.class) @definition.class

(function_definition
    name: (identifier) @name.definition.function) @definition.function

(call
    function: [
        (identifier) @name.reference.call
        (attribute
            attribute: (identifier) @name.reference.call)
    ]) @reference.call
"""

# languages/javascript.rs (subset used for index defs/refs/aliases)
JAVASCRIPT_QUERY = r"""
(class_declaration
    name: (identifier) @name.definition.class) @definition.class

(function_declaration
    name: (identifier) @name.definition.function) @definition.function

(lexical_declaration
    (variable_declarator
        name: (identifier) @name.definition.function
        value: (arrow_function))) @definition.function

(method_definition
    name: (property_identifier) @name.definition.method) @definition.method

(lexical_declaration
    (variable_declarator
        name: (identifier) @name.definition.variable)) @definition.variable

(variable_declaration
    (variable_declarator
        name: (identifier) @name.definition.variable)) @definition.variable

(call_expression
    function: (identifier) @name.reference.call) @reference.call

(call_expression
    function: (member_expression
        property: (property_identifier) @name.reference.call)) @reference.call

(jsx_opening_element
    name: (identifier) @name.reference.jsx)

(jsx_self_closing_element
    name: (identifier) @name.reference.jsx)

(import_specifier
    name: (identifier) @name.reference.import)

(import_clause
    (identifier) @name.reference.import)

(import_specifier
    name: (identifier) @alias.original
    alias: (identifier) @alias.name)
"""

# languages/ts.rs — core defs + refs (not full destructuring surface)
TYPESCRIPT_QUERY = r"""
(function_signature
    name: (identifier) @name.definition.function) @definition.function

(method_signature
    name: (property_identifier) @name.definition.method) @definition.method

(abstract_method_signature
    name: (property_identifier) @name.definition.method) @definition.method

(abstract_class_declaration
    name: (type_identifier) @name.definition.class) @definition.class

(interface_declaration
    name: (type_identifier) @name.definition.interface) @definition.interface

(function_declaration
    name: (identifier) @name.definition.function) @definition.function

(method_definition
    name: (property_identifier) @name.definition.method) @definition.method

(class_declaration
    name: (type_identifier) @name.definition.class) @definition.class

(type_alias_declaration
    name: (type_identifier) @name.definition.type) @definition.type

(enum_declaration
    name: (identifier) @name.definition.enum) @definition.enum

(lexical_declaration
    (variable_declarator
        name: (identifier) @name.definition.function
        value: (arrow_function))) @definition.function

(lexical_declaration
    (variable_declarator
        name: (identifier) @name.definition.variable)) @definition.variable

(variable_declaration
    (variable_declarator
        name: (identifier) @name.definition.variable)) @definition.variable

(call_expression
    function: (identifier) @name.reference.call) @reference.call

(call_expression
    function: (member_expression
        property: (property_identifier) @name.reference.call)) @reference.call

(import_specifier
    name: (identifier) @name.reference.import)

(import_clause
    (identifier) @name.reference.import)

(import_specifier
    name: (identifier) @alias.original
    alias: (identifier) @alias.name)
"""

# languages/rust.rs
RUST_QUERY = r"""
(struct_item
    name: (type_identifier) @name.definition.class) @definition.class

(enum_item
    name: (type_identifier) @name.definition.class) @definition.class

(union_item
    name: (type_identifier) @name.definition.class) @definition.class

(type_item
    name: (type_identifier) @name.definition.class) @definition.class

(declaration_list
    (function_item
        name: (identifier) @name.definition.method)) @definition.method

(function_item
    name: (identifier) @name.definition.function) @definition.function

(trait_item
    name: (type_identifier) @name.definition.interface) @definition.interface

(mod_item
    name: (identifier) @name.definition.module) @definition.module

(macro_definition
    name: (identifier) @name.definition.macro) @definition.macro

(const_item
    name: (identifier) @name.definition.variable) @definition.variable

(static_item
    name: (identifier) @name.definition.variable) @definition.variable

(call_expression
    function: (identifier) @name.reference.call) @reference.call

(call_expression
    function: (field_expression
        field: (field_identifier) @name.reference.call)) @reference.call

(macro_invocation
    macro: (identifier) @name.reference.call) @reference.call

(impl_item
    trait: (type_identifier) @name.reference.implementation) @reference.implementation

(impl_item
    type: (type_identifier) @name.reference.implementation
    !trait) @reference.implementation

(use_declaration
    argument: (identifier) @name.reference.import) @reference.import

(use_declaration
    argument: (scoped_identifier
        name: (identifier) @name.reference.import)) @reference.import

(use_declaration
    argument: (scoped_use_list
        list: (use_list
            (identifier) @name.reference.import)))

(use_declaration
    argument: (scoped_use_list
        list: (use_list
            (scoped_identifier
                name: (identifier) @name.reference.import))))

(use_declaration
    argument: (use_as_clause
        path: (identifier) @alias.original
        alias: (identifier) @alias.name))

(use_declaration
    argument: (use_as_clause
        path: (scoped_identifier
            name: (identifier) @alias.original)
        alias: (identifier) @alias.name))
"""

# languages/golang.rs
GOLANG_QUERY = r"""
(function_declaration
    name: (identifier) @name.definition.function) @definition.function

(method_declaration
    name: (field_identifier) @name.definition.method) @definition.method

(type_declaration
    (type_spec
        name: (type_identifier) @name.definition.type)) @definition.type

(const_declaration
    (const_spec
        name: (identifier) @name.definition.const)) @definition.const

(var_declaration
    (var_spec
        name: (identifier) @name.definition.var)) @definition.var

(call_expression
    function: (identifier) @name.reference.call) @reference.call

(call_expression
    function: (selector_expression
        field: (field_identifier) @name.reference.call)) @reference.call

(type_identifier) @name.reference.type

(import_spec
    path: (interpreted_string_literal) @name.reference.import)

(import_spec
    name: (package_identifier) @alias.name
    path: (interpreted_string_literal) @alias.original)
"""

QUERIES_BY_LANG: dict[str, str] = {
    "python": PYTHON_QUERY,
    "javascript": JAVASCRIPT_QUERY,
    "typescript": TYPESCRIPT_QUERY,
    "rust": RUST_QUERY,
    "golang": GOLANG_QUERY,
}
