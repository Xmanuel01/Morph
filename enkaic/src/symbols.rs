use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeKind {
    Global,
    Function,
    Block,
}

#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,
    pub index: u16,
    pub is_global: bool,
}

#[derive(Default)]
pub struct SymbolTable {
    scopes: Vec<HashMap<String, Symbol>>,
    kinds: Vec<ScopeKind>,
    next_local: Vec<u16>,
    globals: HashMap<String, u16>,
    global_list: Vec<String>,
}

impl SymbolTable {
    pub fn new() -> Self {
        let mut table = SymbolTable::default();
        table.push_scope(ScopeKind::Global);
        table
    }

    pub fn push_scope(&mut self, kind: ScopeKind) {
        self.scopes.push(HashMap::new());
        self.kinds.push(kind);
        self.next_local.push(0);
    }

    pub fn pop_scope(&mut self) {
        self.scopes.pop();
        self.kinds.pop();
        self.next_local.pop();
    }

    pub fn define(&mut self, name: &str) -> Symbol {
        let is_global = matches!(self.kinds.last(), Some(ScopeKind::Global));
        let index = if is_global {
            if let Some(idx) = self.globals.get(name) {
                *idx
            } else {
                let idx = self.global_list.len() as u16;
                self.global_list.push(name.to_string());
                self.globals.insert(name.to_string(), idx);
                idx
            }
        } else {
            let slot = self.next_local.last_mut().unwrap();
            let idx = *slot;
            *slot += 1;
            idx
        };
        let sym = Symbol {
            name: name.to_string(),
            index,
            is_global,
        };
        self.scopes
            .last_mut()
            .unwrap()
            .insert(name.to_string(), sym.clone());
        sym
    }

    pub fn resolve(&self, name: &str) -> Option<Symbol> {
        for scope in self.scopes.iter().rev() {
            if let Some(sym) = scope.get(name) {
                return Some(sym.clone());
            }
        }
        if let Some(idx) = self.globals.get(name) {
            return Some(Symbol {
                name: name.to_string(),
                index: *idx,
                is_global: true,
            });
        }
        None
    }

    pub fn current_local_count(&self) -> u16 {
        *self.next_local.last().unwrap_or(&0)
    }

    pub fn globals(&self) -> &[String] {
        &self.global_list
    }
}
