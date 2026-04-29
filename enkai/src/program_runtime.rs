use std::collections::BTreeMap;
use std::path::Path;

use enkai_runtime::Value;

pub(crate) fn run_program_value_with_env(
    entry: &Path,
    env_overrides: &BTreeMap<String, String>,
) -> Result<Value, String> {
    let env_scope = crate::runtime_exec::ScopedEnv {
        vars: env_overrides.clone(),
        std_override: crate::runtime_exec::bundled_std_override(),
    };
    crate::runtime_exec::with_scoped_env(&env_scope, || run_program_value_inner(entry))
}

fn run_program_value_inner(entry: &Path) -> Result<Value, String> {
    let package = enkai_compiler::modules::load_package(entry).map_err(|err| err.to_string())?;
    enkai_compiler::TypeChecker::check_package(&package).map_err(crate::type_error_message)?;
    let program = enkai_compiler::compiler::compile_package(&package)
        .map_err(crate::compile_error_message)?;
    let mut vm = enkai_runtime::VM::new(false, false, false, false);
    vm.run(&program).map_err(|err| err.to_string())
}
