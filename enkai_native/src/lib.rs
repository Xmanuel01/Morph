use memmap2::Mmap;
use postgres::{types::Type as PgType, Client as PgClient, NoTls};
use rusqlite::types::ValueRef;
use rusqlite::Connection;
use sha2::{Digest, Sha256};
use std::cmp::{Ordering as CmpOrdering, Reverse};
use std::collections::{BTreeMap, BinaryHeap, HashMap};
use std::fs;
use std::net::{SocketAddr, TcpStream};
use std::process::{Child, Command, Stdio};
use std::sync::{
    atomic::{AtomicI64, Ordering},
    Mutex, OnceLock,
};

#[repr(C)]
pub struct FfiSlice {
    pub ptr: *mut u8,
    pub len: usize,
}

#[no_mangle]
/// # Safety
/// The caller must pass a pointer and length originally allocated by `enkai_native`
/// and must ensure the buffer is not freed more than once.
pub unsafe extern "C" fn enkai_free(ptr: *mut u8, len: usize) {
    if ptr.is_null() {
        return;
    }
    let raw = std::ptr::slice_from_raw_parts_mut(ptr, len);
    let _ = unsafe { Box::<[u8]>::from_raw(raw) };
}

fn slice_from_raw<'a>(ptr: *const u8, len: usize) -> Option<&'a [u8]> {
    if ptr.is_null() {
        return None;
    }
    unsafe { Some(std::slice::from_raw_parts(ptr, len)) }
}

fn string_from_raw(ptr: *const u8, len: usize) -> Option<String> {
    let bytes = slice_from_raw(ptr, len)?;
    std::str::from_utf8(bytes).ok().map(|s| s.to_string())
}

fn make_slice(bytes: Vec<u8>) -> FfiSlice {
    let boxed = bytes.into_boxed_slice();
    let len = boxed.len();
    let ptr = Box::into_raw(boxed) as *mut u8;
    FfiSlice { ptr, len }
}

fn null_slice() -> FfiSlice {
    FfiSlice {
        ptr: std::ptr::null_mut(),
        len: 0,
    }
}

fn string_slice(value: String) -> FfiSlice {
    make_slice(value.into_bytes())
}

#[no_mangle]
pub extern "C" fn add_i64(a: i64, b: i64) -> i64 {
    a.wrapping_add(b)
}

#[no_mangle]
pub extern "C" fn echo_string(ptr: *const u8, len: usize) -> FfiSlice {
    let bytes = match slice_from_raw(ptr, len) {
        Some(bytes) => bytes,
        None => return null_slice(),
    };
    make_slice(bytes.to_vec())
}

#[no_mangle]
pub extern "C" fn buffer_from_string(ptr: *const u8, len: usize) -> FfiSlice {
    let bytes = match slice_from_raw(ptr, len) {
        Some(bytes) => bytes,
        None => return null_slice(),
    };
    make_slice(bytes.to_vec())
}

#[no_mangle]
pub extern "C" fn buffer_to_string(ptr: *const u8, len: usize) -> FfiSlice {
    let bytes = match slice_from_raw(ptr, len) {
        Some(bytes) => bytes,
        None => return null_slice(),
    };
    let text = match std::str::from_utf8(bytes) {
        Ok(text) => text.to_string(),
        Err(_) => return null_slice(),
    };
    string_slice(text)
}

#[no_mangle]
pub extern "C" fn echo_buffer(ptr: *const u8, len: usize) -> FfiSlice {
    let bytes = match slice_from_raw(ptr, len) {
        Some(bytes) => bytes,
        None => return null_slice(),
    };
    make_slice(bytes.to_vec())
}

#[no_mangle]
pub extern "C" fn buffer_len(_ptr: *const u8, len: usize) -> i64 {
    len as i64
}

#[no_mangle]
pub extern "C" fn buffer_eq(a_ptr: *const u8, a_len: usize, b_ptr: *const u8, b_len: usize) -> u8 {
    let a = match slice_from_raw(a_ptr, a_len) {
        Some(bytes) => bytes,
        None => return 0,
    };
    let b = match slice_from_raw(b_ptr, b_len) {
        Some(bytes) => bytes,
        None => return 0,
    };
    if a == b {
        1
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn fsx_read_bytes(path_ptr: *const u8, path_len: usize) -> FfiSlice {
    let path = match string_from_raw(path_ptr, path_len) {
        Some(path) => path,
        None => return null_slice(),
    };
    match std::fs::read(&path) {
        Ok(bytes) => make_slice(bytes),
        Err(_) => null_slice(),
    }
}

#[no_mangle]
pub extern "C" fn fsx_write_bytes(
    path_ptr: *const u8,
    path_len: usize,
    buf_ptr: *const u8,
    buf_len: usize,
) -> u8 {
    let path = match string_from_raw(path_ptr, path_len) {
        Some(path) => path,
        None => return 0,
    };
    let bytes = match slice_from_raw(buf_ptr, buf_len) {
        Some(bytes) => bytes,
        None => return 0,
    };
    match std::fs::write(&path, bytes) {
        Ok(_) => 1,
        Err(_) => 0,
    }
}

#[no_mangle]
pub extern "C" fn fsx_mmap_read(path_ptr: *const u8, path_len: usize) -> FfiSlice {
    let path = match string_from_raw(path_ptr, path_len) {
        Some(path) => path,
        None => return null_slice(),
    };
    let file = match std::fs::File::open(&path) {
        Ok(file) => file,
        Err(_) => return null_slice(),
    };
    let mmap = unsafe { Mmap::map(&file) };
    match mmap {
        Ok(map) => make_slice(map.as_ref().to_vec()),
        Err(_) => null_slice(),
    }
}

#[no_mangle]
pub extern "C" fn io_read_stdin() -> FfiSlice {
    use std::io::Read;
    let mut buf = Vec::new();
    match std::io::stdin().read_to_end(&mut buf) {
        Ok(_) => make_slice(buf),
        Err(_) => null_slice(),
    }
}

#[no_mangle]
pub extern "C" fn io_write_stdout(ptr: *const u8, len: usize) -> u8 {
    use std::io::Write;
    let bytes = match slice_from_raw(ptr, len) {
        Some(bytes) => bytes,
        None => return 0,
    };
    match std::io::stdout().write_all(bytes) {
        Ok(_) => 1,
        Err(_) => 0,
    }
}

#[no_mangle]
pub extern "C" fn io_write_stderr(ptr: *const u8, len: usize) -> u8 {
    use std::io::Write;
    let bytes = match slice_from_raw(ptr, len) {
        Some(bytes) => bytes,
        None => return 0,
    };
    match std::io::stderr().write_all(bytes) {
        Ok(_) => 1,
        Err(_) => 0,
    }
}

#[no_mangle]
pub extern "C" fn zstd_compress(ptr: *const u8, len: usize, level: i64) -> FfiSlice {
    let bytes = match slice_from_raw(ptr, len) {
        Some(bytes) => bytes,
        None => return null_slice(),
    };
    let level = level.clamp(-7, 22) as i32;
    match zstd::stream::encode_all(bytes, level) {
        Ok(out) => make_slice(out),
        Err(_) => null_slice(),
    }
}

#[no_mangle]
pub extern "C" fn zstd_decompress(ptr: *const u8, len: usize) -> FfiSlice {
    let bytes = match slice_from_raw(ptr, len) {
        Some(bytes) => bytes,
        None => return null_slice(),
    };
    match zstd::stream::decode_all(bytes) {
        Ok(out) => make_slice(out),
        Err(_) => null_slice(),
    }
}

#[no_mangle]
pub extern "C" fn hash_sha256(ptr: *const u8, len: usize) -> FfiSlice {
    let bytes = match slice_from_raw(ptr, len) {
        Some(bytes) => bytes,
        None => return null_slice(),
    };
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let result = hasher.finalize();
    make_slice(result.to_vec())
}

#[no_mangle]
pub extern "C" fn hash_sha256_many(ptr: *const u8, len: usize, count: i64) -> FfiSlice {
    let bytes = match slice_from_raw(ptr, len) {
        Some(bytes) => bytes,
        None => return null_slice(),
    };
    if count <= 0 {
        return make_slice(Vec::new());
    }
    let repeat = count as usize;
    let mut out = Vec::with_capacity(repeat.saturating_mul(32));
    for _ in 0..repeat {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        out.extend_from_slice(&hasher.finalize());
    }
    make_slice(out)
}

#[no_mangle]
pub extern "C" fn env_get(key_ptr: *const u8, key_len: usize) -> FfiSlice {
    let key = match string_from_raw(key_ptr, key_len) {
        Some(key) => key,
        None => return null_slice(),
    };
    match std::env::var(&key) {
        Ok(value) => string_slice(value),
        Err(_) => null_slice(),
    }
}

#[no_mangle]
pub extern "C" fn env_set(
    key_ptr: *const u8,
    key_len: usize,
    value_ptr: *const u8,
    value_len: usize,
) -> u8 {
    let key = match string_from_raw(key_ptr, key_len) {
        Some(key) => key,
        None => return 0,
    };
    let value = match string_from_raw(value_ptr, value_len) {
        Some(value) => value,
        None => return 0,
    };
    std::env::set_var(key, value);
    1
}

#[no_mangle]
pub extern "C" fn env_remove(key_ptr: *const u8, key_len: usize) -> u8 {
    let key = match string_from_raw(key_ptr, key_len) {
        Some(key) => key,
        None => return 0,
    };
    std::env::remove_var(key);
    1
}

#[no_mangle]
pub extern "C" fn env_cwd() -> FfiSlice {
    match std::env::current_dir() {
        Ok(path) => string_slice(path.to_string_lossy().to_string()),
        Err(_) => null_slice(),
    }
}

#[no_mangle]
pub extern "C" fn env_set_cwd(path_ptr: *const u8, path_len: usize) -> u8 {
    let path = match string_from_raw(path_ptr, path_len) {
        Some(path) => path,
        None => return 0,
    };
    match std::env::set_current_dir(&path) {
        Ok(_) => 1,
        Err(_) => 0,
    }
}

#[no_mangle]
pub extern "C" fn path_join(
    a_ptr: *const u8,
    a_len: usize,
    b_ptr: *const u8,
    b_len: usize,
) -> FfiSlice {
    let a = match string_from_raw(a_ptr, a_len) {
        Some(a) => a,
        None => return null_slice(),
    };
    let b = match string_from_raw(b_ptr, b_len) {
        Some(b) => b,
        None => return null_slice(),
    };
    let joined = std::path::Path::new(&a).join(b);
    string_slice(joined.to_string_lossy().to_string())
}

#[no_mangle]
pub extern "C" fn path_dirname(path_ptr: *const u8, path_len: usize) -> FfiSlice {
    let path = match string_from_raw(path_ptr, path_len) {
        Some(path) => path,
        None => return null_slice(),
    };
    let path = std::path::Path::new(&path);
    match path.parent() {
        Some(parent) => string_slice(parent.to_string_lossy().to_string()),
        None => null_slice(),
    }
}

#[no_mangle]
pub extern "C" fn path_basename(path_ptr: *const u8, path_len: usize) -> FfiSlice {
    let path = match string_from_raw(path_ptr, path_len) {
        Some(path) => path,
        None => return null_slice(),
    };
    let path = std::path::Path::new(&path);
    match path.file_name().and_then(|name| name.to_str()) {
        Some(name) => string_slice(name.to_string()),
        None => null_slice(),
    }
}

#[no_mangle]
pub extern "C" fn path_extname(path_ptr: *const u8, path_len: usize) -> FfiSlice {
    let path = match string_from_raw(path_ptr, path_len) {
        Some(path) => path,
        None => return null_slice(),
    };
    let path = std::path::Path::new(&path);
    match path.extension().and_then(|name| name.to_str()) {
        Some(name) => string_slice(name.to_string()),
        None => null_slice(),
    }
}

#[no_mangle]
pub extern "C" fn path_normalize(path_ptr: *const u8, path_len: usize) -> FfiSlice {
    let path = match string_from_raw(path_ptr, path_len) {
        Some(path) => path,
        None => return null_slice(),
    };
    let normalized = normalize_path(std::path::Path::new(&path));
    string_slice(normalized.to_string_lossy().to_string())
}

#[no_mangle]
pub extern "C" fn time_now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(delta) => delta.as_millis().min(i64::MAX as u128) as i64,
        Err(_) => 0,
    }
}

#[no_mangle]
pub extern "C" fn time_sleep_ms(ms: i64) {
    let ms = if ms < 0 { 0 } else { ms } as u64;
    std::thread::sleep(std::time::Duration::from_millis(ms));
}

#[no_mangle]
pub extern "C" fn log_emit(
    level_ptr: *const u8,
    level_len: usize,
    msg_ptr: *const u8,
    msg_len: usize,
) -> u8 {
    let level = match string_from_raw(level_ptr, level_len) {
        Some(level) => level,
        None => return 0,
    };
    let msg = match string_from_raw(msg_ptr, msg_len) {
        Some(msg) => msg,
        None => return 0,
    };
    let level_norm = level.to_lowercase();
    if matches!(level_norm.as_str(), "error" | "warn") {
        eprintln!("[{}] {}", level_norm, msg);
    } else {
        println!("[{}] {}", level_norm, msg);
    }
    1
}

fn normalize_path(path: &std::path::Path) -> std::path::PathBuf {
    use std::path::Component;
    let mut parts: Vec<std::ffi::OsString> = Vec::new();
    let mut root_len = 0usize;
    for comp in path.components() {
        match comp {
            Component::Prefix(prefix) => {
                parts.push(prefix.as_os_str().to_os_string());
                root_len = parts.len();
            }
            Component::RootDir => {
                parts.push(comp.as_os_str().to_os_string());
                root_len = parts.len();
            }
            Component::CurDir => {}
            Component::ParentDir => {
                if parts.len() > root_len {
                    parts.pop();
                } else if root_len == 0 {
                    parts.push(std::ffi::OsString::from(".."));
                }
            }
            Component::Normal(name) => parts.push(name.to_os_string()),
        }
    }
    let mut out = std::path::PathBuf::new();
    for part in parts {
        out.push(part);
    }
    out
}

fn process_table() -> &'static Mutex<HashMap<i64, Child>> {
    static TABLE: OnceLock<Mutex<HashMap<i64, Child>>> = OnceLock::new();
    TABLE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn next_process_handle() -> i64 {
    static NEXT: AtomicI64 = AtomicI64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

fn db_table() -> &'static Mutex<HashMap<i64, Connection>> {
    static TABLE: OnceLock<Mutex<HashMap<i64, Connection>>> = OnceLock::new();
    TABLE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn next_db_handle() -> i64 {
    static NEXT: AtomicI64 = AtomicI64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

fn pg_table() -> &'static Mutex<HashMap<i64, PgClient>> {
    static TABLE: OnceLock<Mutex<HashMap<i64, PgClient>>> = OnceLock::new();
    TABLE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn next_pg_handle() -> i64 {
    static NEXT: AtomicI64 = AtomicI64::new(1_000_000);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

fn parse_args_json(text: &str) -> Option<Vec<String>> {
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    let arr = value.as_array()?;
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        if let Some(s) = item.as_str() {
            out.push(s.to_string());
        }
    }
    Some(out)
}

#[no_mangle]
pub extern "C" fn process_spawn(
    cmd_ptr: *const u8,
    cmd_len: usize,
    args_ptr: *const u8,
    args_len: usize,
    cwd_ptr: *const u8,
    cwd_len: usize,
) -> i64 {
    let cmd = match string_from_raw(cmd_ptr, cmd_len) {
        Some(cmd) => cmd,
        None => return 0,
    };
    let args_text = string_from_raw(args_ptr, args_len).unwrap_or_default();
    let args = parse_args_json(&args_text).unwrap_or_default();
    let cwd = string_from_raw(cwd_ptr, cwd_len);
    let mut command = Command::new(cmd);
    command.args(args);
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }
    command.stdin(Stdio::null());
    command.stdout(Stdio::null());
    command.stderr(Stdio::null());
    match command.spawn() {
        Ok(child) => {
            let handle = next_process_handle();
            if let Ok(mut table) = process_table().lock() {
                table.insert(handle, child);
            }
            handle
        }
        Err(_) => 0,
    }
}

#[no_mangle]
pub extern "C" fn process_wait(handle: i64) -> i64 {
    let child = if let Ok(mut table) = process_table().lock() {
        table.remove(&handle)
    } else {
        None
    };
    let mut child = match child {
        Some(child) => child,
        None => return -1,
    };
    match child.wait() {
        Ok(status) => status.code().unwrap_or(-1) as i64,
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn process_kill(handle: i64) -> u8 {
    let child = if let Ok(mut table) = process_table().lock() {
        table.remove(&handle)
    } else {
        None
    };
    let mut child = match child {
        Some(child) => child,
        None => return 0,
    };
    match child.kill() {
        Ok(_) => 1,
        Err(_) => 0,
    }
}

#[no_mangle]
pub extern "C" fn process_run(
    cmd_ptr: *const u8,
    cmd_len: usize,
    args_ptr: *const u8,
    args_len: usize,
    cwd_ptr: *const u8,
    cwd_len: usize,
) -> FfiSlice {
    let cmd = match string_from_raw(cmd_ptr, cmd_len) {
        Some(cmd) => cmd,
        None => return null_slice(),
    };
    let args_text = string_from_raw(args_ptr, args_len).unwrap_or_default();
    let args = parse_args_json(&args_text).unwrap_or_default();
    let cwd = string_from_raw(cwd_ptr, cwd_len);
    let mut command = Command::new(cmd);
    command.args(args);
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }
    command.stdout(Stdio::piped());
    command.stderr(Stdio::piped());
    match command.output() {
        Ok(output) => {
            let status = output.status.code().unwrap_or(-1);
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            let json = serde_json::json!({
                "status": status,
                "stdout": stdout,
                "stderr": stderr
            });
            match serde_json::to_string(&json) {
                Ok(text) => string_slice(text),
                Err(_) => null_slice(),
            }
        }
        Err(_) => null_slice(),
    }
}

#[no_mangle]
pub extern "C" fn process_exit(code: i64) {
    std::process::exit(code as i32);
}

#[no_mangle]
pub extern "C" fn db_sqlite_open(path_ptr: *const u8, path_len: usize) -> i64 {
    let path = match string_from_raw(path_ptr, path_len) {
        Some(path) => path,
        None => return 0,
    };
    let conn = match Connection::open(path) {
        Ok(conn) => conn,
        Err(_) => return 0,
    };
    let _ = conn.busy_timeout(std::time::Duration::from_secs(5));
    let handle = next_db_handle();
    if let Ok(mut table) = db_table().lock() {
        table.insert(handle, conn);
        handle
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn db_sqlite_close(handle: i64) -> u8 {
    if handle <= 0 {
        return 0;
    }
    if let Ok(mut table) = db_table().lock() {
        if table.remove(&handle).is_some() {
            1
        } else {
            0
        }
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn db_sqlite_exec(handle: i64, sql_ptr: *const u8, sql_len: usize) -> i64 {
    if handle <= 0 {
        return -1;
    }
    let sql = match string_from_raw(sql_ptr, sql_len) {
        Some(sql) => sql,
        None => return -1,
    };
    let mut table = match db_table().lock() {
        Ok(table) => table,
        Err(_) => return -1,
    };
    let conn = match table.get_mut(&handle) {
        Some(conn) => conn,
        None => return -1,
    };
    match conn.execute_batch(&sql) {
        Ok(_) => conn.changes() as i64,
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn db_sqlite_transaction_begin(handle: i64) -> u8 {
    if handle <= 0 {
        return 0;
    }
    let mut table = match db_table().lock() {
        Ok(table) => table,
        Err(_) => return 0,
    };
    let conn = match table.get_mut(&handle) {
        Some(conn) => conn,
        None => return 0,
    };
    match conn.execute_batch("BEGIN IMMEDIATE TRANSACTION") {
        Ok(_) => 1,
        Err(_) => 0,
    }
}

#[no_mangle]
pub extern "C" fn db_sqlite_transaction_commit(handle: i64) -> u8 {
    if handle <= 0 {
        return 0;
    }
    let mut table = match db_table().lock() {
        Ok(table) => table,
        Err(_) => return 0,
    };
    let conn = match table.get_mut(&handle) {
        Some(conn) => conn,
        None => return 0,
    };
    match conn.execute_batch("COMMIT") {
        Ok(_) => 1,
        Err(_) => 0,
    }
}

#[no_mangle]
pub extern "C" fn db_sqlite_exec_many(
    handle: i64,
    sql_ptr: *const u8,
    sql_len: usize,
    count: i64,
) -> i64 {
    if handle <= 0 || count < 0 {
        return -1;
    }
    let sql = match string_from_raw(sql_ptr, sql_len) {
        Some(sql) => sql,
        None => return -1,
    };
    let mut table = match db_table().lock() {
        Ok(table) => table,
        Err(_) => return -1,
    };
    let conn = match table.get_mut(&handle) {
        Some(conn) => conn,
        None => return -1,
    };
    let mut stmt = match conn.prepare(&sql) {
        Ok(stmt) => stmt,
        Err(_) => return -1,
    };
    let mut total = 0i64;
    for _ in 0..(count as usize) {
        let changed = match stmt.execute([]) {
            Ok(changed) => changed as i64,
            Err(_) => return -1,
        };
        total = total.saturating_add(changed);
    }
    total
}

#[no_mangle]
pub extern "C" fn db_sqlite_query(handle: i64, sql_ptr: *const u8, sql_len: usize) -> FfiSlice {
    if handle <= 0 {
        return string_slice("[]".to_string());
    }
    let sql = match string_from_raw(sql_ptr, sql_len) {
        Some(sql) => sql,
        None => return string_slice("[]".to_string()),
    };
    let mut table = match db_table().lock() {
        Ok(table) => table,
        Err(_) => return string_slice("[]".to_string()),
    };
    let conn = match table.get_mut(&handle) {
        Some(conn) => conn,
        None => return string_slice("[]".to_string()),
    };
    let mut stmt = match conn.prepare(&sql) {
        Ok(stmt) => stmt,
        Err(_) => return string_slice("[]".to_string()),
    };
    let col_count = stmt.column_count();
    let mut col_names = Vec::with_capacity(col_count);
    for idx in 0..col_count {
        let name = stmt
            .column_name(idx)
            .map(|s| s.to_string())
            .unwrap_or_else(|_| format!("c{}", idx));
        col_names.push(name);
    }
    let mut rows = match stmt.query([]) {
        Ok(rows) => rows,
        Err(_) => return string_slice("[]".to_string()),
    };
    let mut out = Vec::new();
    loop {
        let row = match rows.next() {
            Ok(Some(row)) => row,
            Ok(None) => break,
            Err(_) => return string_slice("[]".to_string()),
        };
        let mut obj = serde_json::Map::new();
        for (idx, key) in col_names.iter().enumerate() {
            let value = match row.get_ref(idx) {
                Ok(ValueRef::Null) => serde_json::Value::Null,
                Ok(ValueRef::Integer(i)) => serde_json::json!(i),
                Ok(ValueRef::Real(f)) => serde_json::json!(f),
                Ok(ValueRef::Text(bytes)) => {
                    serde_json::Value::String(String::from_utf8_lossy(bytes).to_string())
                }
                Ok(ValueRef::Blob(bytes)) => {
                    serde_json::Value::String(String::from_utf8_lossy(bytes).to_string())
                }
                Err(_) => serde_json::Value::Null,
            };
            obj.insert(key.clone(), value);
        }
        out.push(serde_json::Value::Object(obj));
    }
    match serde_json::to_string(&out) {
        Ok(text) => string_slice(text),
        Err(_) => string_slice("[]".to_string()),
    }
}

#[no_mangle]
pub extern "C" fn db_postgres_open(conn_ptr: *const u8, conn_len: usize) -> i64 {
    let conn = match string_from_raw(conn_ptr, conn_len) {
        Some(conn) => conn,
        None => return 0,
    };
    let client = match PgClient::connect(&conn, NoTls) {
        Ok(client) => client,
        Err(_) => return 0,
    };
    let handle = next_pg_handle();
    if let Ok(mut table) = pg_table().lock() {
        table.insert(handle, client);
        handle
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn db_postgres_close(handle: i64) -> u8 {
    if handle <= 0 {
        return 0;
    }
    if let Ok(mut table) = pg_table().lock() {
        if table.remove(&handle).is_some() {
            1
        } else {
            0
        }
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn db_postgres_exec(handle: i64, sql_ptr: *const u8, sql_len: usize) -> i64 {
    if handle <= 0 {
        return -1;
    }
    let sql = match string_from_raw(sql_ptr, sql_len) {
        Some(sql) => sql,
        None => return -1,
    };
    let mut table = match pg_table().lock() {
        Ok(table) => table,
        Err(_) => return -1,
    };
    let client = match table.get_mut(&handle) {
        Some(client) => client,
        None => return -1,
    };
    match client.execute(&sql, &[]) {
        Ok(count) => count as i64,
        Err(_) => -1,
    }
}

fn pg_cell_to_json(row: &postgres::Row, idx: usize, ty: &PgType) -> serde_json::Value {
    if *ty == PgType::BOOL {
        return match row.try_get::<usize, Option<bool>>(idx) {
            Ok(Some(value)) => serde_json::json!(value),
            Ok(None) => serde_json::Value::Null,
            Err(_) => serde_json::Value::Null,
        };
    }
    if *ty == PgType::INT2 {
        return match row.try_get::<usize, Option<i16>>(idx) {
            Ok(Some(value)) => serde_json::json!(value),
            Ok(None) => serde_json::Value::Null,
            Err(_) => serde_json::Value::Null,
        };
    }
    if *ty == PgType::INT4 {
        return match row.try_get::<usize, Option<i32>>(idx) {
            Ok(Some(value)) => serde_json::json!(value),
            Ok(None) => serde_json::Value::Null,
            Err(_) => serde_json::Value::Null,
        };
    }
    if *ty == PgType::INT8 {
        return match row.try_get::<usize, Option<i64>>(idx) {
            Ok(Some(value)) => serde_json::json!(value),
            Ok(None) => serde_json::Value::Null,
            Err(_) => serde_json::Value::Null,
        };
    }
    if *ty == PgType::FLOAT4 {
        return match row.try_get::<usize, Option<f32>>(idx) {
            Ok(Some(value)) => serde_json::json!(value),
            Ok(None) => serde_json::Value::Null,
            Err(_) => serde_json::Value::Null,
        };
    }
    if *ty == PgType::FLOAT8 {
        return match row.try_get::<usize, Option<f64>>(idx) {
            Ok(Some(value)) => serde_json::json!(value),
            Ok(None) => serde_json::Value::Null,
            Err(_) => serde_json::Value::Null,
        };
    }
    if *ty == PgType::JSON || *ty == PgType::JSONB {
        return match row.try_get::<usize, Option<serde_json::Value>>(idx) {
            Ok(Some(value)) => value,
            Ok(None) => serde_json::Value::Null,
            Err(_) => serde_json::Value::Null,
        };
    }
    if *ty == PgType::BYTEA {
        return match row.try_get::<usize, Option<Vec<u8>>>(idx) {
            Ok(Some(bytes)) => serde_json::Value::String(hex_bytes(&bytes)),
            Ok(None) => serde_json::Value::Null,
            Err(_) => serde_json::Value::Null,
        };
    }
    match row.try_get::<usize, Option<String>>(idx) {
        Ok(Some(value)) => serde_json::Value::String(value),
        Ok(None) => serde_json::Value::Null,
        Err(_) => serde_json::Value::Null,
    }
}

fn hex_bytes(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

#[no_mangle]
pub extern "C" fn db_postgres_query(handle: i64, sql_ptr: *const u8, sql_len: usize) -> FfiSlice {
    if handle <= 0 {
        return string_slice("[]".to_string());
    }
    let sql = match string_from_raw(sql_ptr, sql_len) {
        Some(sql) => sql,
        None => return string_slice("[]".to_string()),
    };
    let mut table = match pg_table().lock() {
        Ok(table) => table,
        Err(_) => return string_slice("[]".to_string()),
    };
    let client = match table.get_mut(&handle) {
        Some(client) => client,
        None => return string_slice("[]".to_string()),
    };
    let rows = match client.query(&sql, &[]) {
        Ok(rows) => rows,
        Err(_) => return string_slice("[]".to_string()),
    };
    let mut out = Vec::new();
    for row in rows {
        let mut obj = serde_json::Map::new();
        for (idx, col) in row.columns().iter().enumerate() {
            let value = pg_cell_to_json(&row, idx, col.type_());
            obj.insert(col.name().to_string(), value);
        }
        out.push(serde_json::Value::Object(obj));
    }
    match serde_json::to_string(&out) {
        Ok(text) => string_slice(text),
        Err(_) => string_slice("[]".to_string()),
    }
}

#[no_mangle]
pub extern "C" fn tls_fetch_server_info(
    host_ptr: *const u8,
    host_len: usize,
    port: i64,
    timeout_ms: i64,
) -> FfiSlice {
    let host = match string_from_raw(host_ptr, host_len) {
        Some(host) => host,
        None => return null_slice(),
    };
    let port = if port <= 0 || port > u16::MAX as i64 {
        return null_slice();
    } else {
        port as u16
    };
    let timeout_ms = timeout_ms.clamp(100, 60_000) as u64;
    let addr = format!("{}:{}", host, port);
    let socket: SocketAddr = match addr.parse() {
        Ok(addr) => addr,
        Err(_) => return null_slice(),
    };
    let stream =
        match TcpStream::connect_timeout(&socket, std::time::Duration::from_millis(timeout_ms)) {
            Ok(stream) => stream,
            Err(_) => return null_slice(),
        };
    let _ = stream.set_read_timeout(Some(std::time::Duration::from_millis(timeout_ms)));
    let _ = stream.set_write_timeout(Some(std::time::Duration::from_millis(timeout_ms)));
    let connector = match native_tls::TlsConnector::new() {
        Ok(conn) => conn,
        Err(_) => return null_slice(),
    };
    let stream = match connector.connect(&host, stream) {
        Ok(stream) => stream,
        Err(_) => return null_slice(),
    };
    let cert_sha256 = stream
        .peer_certificate()
        .ok()
        .flatten()
        .and_then(|cert| cert.to_der().ok())
        .map(|der| {
            let mut hasher = Sha256::new();
            hasher.update(&der);
            format!("{:x}", hasher.finalize())
        })
        .unwrap_or_default();
    let json = serde_json::json!({
        "host": host,
        "port": port,
        "cert_sha256": cert_sha256
    });
    match serde_json::to_string(&json) {
        Ok(text) => string_slice(text),
        Err(_) => null_slice(),
    }
}

fn parse_json_numbers(text: &str) -> Option<Vec<f64>> {
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    let arr = value.as_array()?;
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let num = item.as_f64()?;
        if !num.is_finite() {
            return None;
        }
        out.push(num);
    }
    Some(out)
}

fn parse_json_i64(text: &str) -> Option<Vec<i64>> {
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    let arr = value.as_array()?;
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        if let Some(value) = item.as_i64() {
            out.push(value);
            continue;
        }
        let value = item.as_f64()?;
        if !value.is_finite() {
            return None;
        }
        out.push(value as i64);
    }
    Some(out)
}

fn parse_json_i64_map(text: &str) -> Option<BTreeMap<String, i64>> {
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    let obj = value.as_object()?;
    let mut out = BTreeMap::new();
    for (key, raw) in obj {
        let num = if let Some(v) = raw.as_i64() {
            v
        } else if let Some(v) = raw.as_f64() {
            if !v.is_finite() {
                return None;
            }
            v as i64
        } else {
            return None;
        };
        out.insert(key.clone(), num);
    }
    Some(out)
}

fn parse_json_array(text: &str) -> Option<Vec<serde_json::Value>> {
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    Some(value.as_array()?.clone())
}

fn deterministic_order_key(seed: u64, index: usize) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(seed.to_le_bytes());
    hasher.update((index as u64).to_le_bytes());
    let digest = hasher.finalize();
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    u64::from_le_bytes(bytes)
}

fn parse_json_records(text: &str) -> Option<Vec<serde_json::Map<String, serde_json::Value>>> {
    let list = parse_json_array(text)?;
    let mut out = Vec::with_capacity(list.len());
    for item in list {
        let obj = item.as_object()?.clone();
        out.push(obj);
    }
    Some(out)
}

fn normalize_schema_type(value: &str) -> String {
    match value.trim().to_ascii_lowercase().as_str() {
        "integer" => "int".to_string(),
        "number" => "float".to_string(),
        "str" => "string".to_string(),
        "boolean" => "bool".to_string(),
        "any" => "mixed".to_string(),
        other => other.to_string(),
    }
}

fn classify_json_type(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(number) => {
            if number.is_i64() || number.is_u64() {
                "int"
            } else {
                "float"
            }
        }
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

fn parse_schema_definition(text: &str) -> Option<BTreeMap<String, (String, bool)>> {
    let raw: serde_json::Value = serde_json::from_str(text).ok()?;
    let map = raw.as_object()?;
    let mut out = BTreeMap::new();
    for (field, entry) in map {
        match entry {
            serde_json::Value::String(ty) => {
                out.insert(field.clone(), (normalize_schema_type(ty), false));
            }
            serde_json::Value::Object(obj) => {
                let ty = obj
                    .get("type")
                    .and_then(|value| value.as_str())
                    .map(normalize_schema_type)
                    .unwrap_or_else(|| "mixed".to_string());
                let nullable = obj
                    .get("nullable")
                    .and_then(|value| value.as_bool())
                    .unwrap_or(false);
                out.insert(field.clone(), (ty, nullable));
            }
            _ => return None,
        }
    }
    Some(out)
}

fn cast_to_schema_type(value: &serde_json::Value, ty: &str) -> Option<serde_json::Value> {
    match ty {
        "mixed" | "any" => Some(value.clone()),
        "null" => {
            if value.is_null() {
                Some(serde_json::Value::Null)
            } else {
                None
            }
        }
        "int" => {
            if let Some(num) = value.as_i64() {
                return Some(serde_json::json!(num));
            }
            if let Some(num) = value.as_u64() {
                if num <= i64::MAX as u64 {
                    return Some(serde_json::json!(num as i64));
                }
                return None;
            }
            if let Some(text) = value.as_str() {
                let parsed = text.trim().parse::<i64>().ok()?;
                return Some(serde_json::json!(parsed));
            }
            None
        }
        "float" => {
            if let Some(num) = value.as_f64() {
                if num.is_finite() {
                    return Some(serde_json::json!(num));
                }
                return None;
            }
            if let Some(text) = value.as_str() {
                let parsed = text.trim().parse::<f64>().ok()?;
                if parsed.is_finite() {
                    return Some(serde_json::json!(parsed));
                }
                return None;
            }
            None
        }
        "bool" => {
            if let Some(flag) = value.as_bool() {
                return Some(serde_json::json!(flag));
            }
            if let Some(text) = value.as_str() {
                let lowered = text.trim().to_ascii_lowercase();
                let parsed = match lowered.as_str() {
                    "1" | "true" | "yes" | "on" => Some(true),
                    "0" | "false" | "no" | "off" => Some(false),
                    _ => None,
                }?;
                return Some(serde_json::json!(parsed));
            }
            None
        }
        "string" => {
            if let Some(text) = value.as_str() {
                Some(serde_json::json!(text))
            } else {
                Some(serde_json::json!(value.to_string()))
            }
        }
        "array" => {
            if value.is_array() {
                Some(value.clone())
            } else {
                None
            }
        }
        "object" => {
            if value.is_object() {
                Some(value.clone())
            } else {
                None
            }
        }
        _ => None,
    }
}

fn value_to_key_string(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Bool(flag) => {
            if *flag {
                "true".to_string()
            } else {
                "false".to_string()
            }
        }
        serde_json::Value::Number(number) => number.to_string(),
        serde_json::Value::String(text) => text.clone(),
        serde_json::Value::Array(_) | serde_json::Value::Object(_) => value.to_string(),
    }
}

fn merged_join_row(
    left: Option<&serde_json::Map<String, serde_json::Value>>,
    right: Option<&serde_json::Map<String, serde_json::Value>>,
    left_columns: &[String],
    right_columns: &[String],
) -> serde_json::Map<String, serde_json::Value> {
    let mut out = serde_json::Map::new();
    for key in left_columns {
        let value = left
            .and_then(|row| row.get(key))
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        out.insert(key.clone(), value);
    }
    for key in right_columns {
        let out_key = if out.contains_key(key) {
            format!("right_{}", key)
        } else {
            key.clone()
        };
        let value = right
            .and_then(|row| row.get(key))
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        out.insert(out_key, value);
    }
    out
}

fn parse_quantiles(raw: &str) -> Option<Vec<f64>> {
    let mut out = parse_json_numbers(raw)?;
    for value in &out {
        if !(*value >= 0.0 && *value <= 1.0) {
            return None;
        }
    }
    if out.is_empty() {
        return None;
    }
    out.shrink_to_fit();
    Some(out)
}

fn quantile_interpolate(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let clamped = q.clamp(0.0, 1.0);
    let pos = clamped * (sorted.len() - 1) as f64;
    let low = pos.floor() as usize;
    let high = pos.ceil() as usize;
    if low == high {
        return sorted[low];
    }
    let weight = pos - low as f64;
    sorted[low] * (1.0 - weight) + sorted[high] * weight
}

fn json_string(value: serde_json::Value) -> FfiSlice {
    match serde_json::to_string(&value) {
        Ok(text) => string_slice(text),
        Err(_) => null_slice(),
    }
}

#[no_mangle]
pub extern "C" fn analysis_csv_read(
    path_ptr: *const u8,
    path_len: usize,
    delimiter_ptr: *const u8,
    delimiter_len: usize,
    has_header: u8,
) -> FfiSlice {
    let path = match string_from_raw(path_ptr, path_len) {
        Some(path) => path,
        None => return null_slice(),
    };
    let delimiter_raw =
        string_from_raw(delimiter_ptr, delimiter_len).unwrap_or_else(|| ",".to_string());
    let delimiter = delimiter_raw.as_bytes().first().copied().unwrap_or(b',');
    let mut builder = csv::ReaderBuilder::new();
    builder.delimiter(delimiter);
    builder.has_headers(has_header != 0);
    let mut reader = match builder.from_path(path) {
        Ok(reader) => reader,
        Err(_) => return string_slice("[]".to_string()),
    };
    let headers: Vec<String> = if has_header != 0 {
        match reader.headers() {
            Ok(header) => header.iter().map(|value| value.to_string()).collect(),
            Err(_) => return string_slice("[]".to_string()),
        }
    } else {
        Vec::new()
    };
    let mut out = Vec::new();
    for (idx, row) in reader.records().enumerate() {
        let Ok(row) = row else {
            return string_slice("[]".to_string());
        };
        let mut obj = serde_json::Map::new();
        for (col_idx, value) in row.iter().enumerate() {
            let key = if has_header != 0 {
                headers
                    .get(col_idx)
                    .cloned()
                    .unwrap_or_else(|| format!("c{}", col_idx))
            } else {
                format!("c{}", col_idx)
            };
            obj.insert(key, serde_json::Value::String(value.to_string()));
        }
        obj.insert(
            "_row".to_string(),
            serde_json::Value::Number((idx as u64).into()),
        );
        out.push(serde_json::Value::Object(obj));
    }
    json_string(serde_json::Value::Array(out))
}

#[no_mangle]
pub extern "C" fn analysis_jsonl_read(path_ptr: *const u8, path_len: usize) -> FfiSlice {
    let path = match string_from_raw(path_ptr, path_len) {
        Some(path) => path,
        None => return null_slice(),
    };
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(_) => return string_slice("[]".to_string()),
    };
    let mut out = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value: serde_json::Value = match serde_json::from_str(trimmed) {
            Ok(value) => value,
            Err(_) => return string_slice("[]".to_string()),
        };
        out.push(value);
    }
    json_string(serde_json::Value::Array(out))
}

#[no_mangle]
pub extern "C" fn analysis_infer_schema(rows_ptr: *const u8, rows_len: usize) -> FfiSlice {
    let raw = match string_from_raw(rows_ptr, rows_len) {
        Some(raw) => raw,
        None => return null_slice(),
    };
    let rows: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(value) => value,
        Err(_) => return string_slice("{}".to_string()),
    };
    let Some(list) = rows.as_array() else {
        return string_slice("{}".to_string());
    };
    let mut schema = serde_json::Map::new();
    for row in list {
        let Some(obj) = row.as_object() else {
            continue;
        };
        for (key, value) in obj {
            let ty = match value {
                serde_json::Value::Null => "null",
                serde_json::Value::Bool(_) => "bool",
                serde_json::Value::Number(number) => {
                    if number.is_i64() || number.is_u64() {
                        "int"
                    } else {
                        "float"
                    }
                }
                serde_json::Value::String(_) => "string",
                serde_json::Value::Array(_) => "array",
                serde_json::Value::Object(_) => "object",
            };
            schema
                .entry(key.clone())
                .or_insert_with(|| serde_json::Value::String(ty.to_string()));
        }
    }
    json_string(serde_json::Value::Object(schema))
}

#[no_mangle]
pub extern "C" fn analysis_infer_schema_typed(rows_ptr: *const u8, rows_len: usize) -> FfiSlice {
    let raw = match string_from_raw(rows_ptr, rows_len) {
        Some(raw) => raw,
        None => return null_slice(),
    };
    let rows = match parse_json_records(&raw) {
        Some(rows) => rows,
        None => return string_slice("{\"fields\":{}}".to_string()),
    };
    #[derive(Default)]
    struct FieldSummary {
        nullable: bool,
        count: usize,
        types: BTreeMap<String, usize>,
    }
    let mut fields: BTreeMap<String, FieldSummary> = BTreeMap::new();
    for row in rows {
        for (key, value) in row {
            let entry = fields.entry(key).or_default();
            entry.count += 1;
            let ty = classify_json_type(&value).to_string();
            if ty == "null" {
                entry.nullable = true;
                continue;
            }
            *entry.types.entry(ty).or_insert(0) += 1;
        }
    }
    let mut fields_json = serde_json::Map::new();
    for (field, summary) in fields {
        let mut observed_types: Vec<String> = summary.types.keys().cloned().collect();
        observed_types.sort();
        let resolved_type = if observed_types.is_empty() {
            "null".to_string()
        } else if observed_types.len() == 1 {
            observed_types[0].clone()
        } else if observed_types.len() == 2
            && observed_types.iter().any(|value| value == "int")
            && observed_types.iter().any(|value| value == "float")
        {
            "float".to_string()
        } else {
            "mixed".to_string()
        };
        fields_json.insert(
            field,
            serde_json::json!({
                "type": resolved_type,
                "nullable": summary.nullable,
                "observed_types": observed_types,
                "count": summary.count
            }),
        );
    }
    json_string(serde_json::json!({
        "schema_version": 1,
        "fields": fields_json
    }))
}

#[no_mangle]
pub extern "C" fn analysis_validate_schema(
    rows_ptr: *const u8,
    rows_len: usize,
    schema_ptr: *const u8,
    schema_len: usize,
) -> FfiSlice {
    let raw_rows = match string_from_raw(rows_ptr, rows_len) {
        Some(raw) => raw,
        None => return null_slice(),
    };
    let raw_schema = match string_from_raw(schema_ptr, schema_len) {
        Some(raw) => raw,
        None => return null_slice(),
    };
    let rows = match parse_json_records(&raw_rows) {
        Some(rows) => rows,
        None => {
            return string_slice(
                "{\"ok\":false,\"errors\":[{\"code\":\"invalid_rows\"}],\"rows\":[]}".to_string(),
            )
        }
    };
    let schema = match parse_schema_definition(&raw_schema) {
        Some(schema) => schema,
        None => {
            return string_slice(
                "{\"ok\":false,\"errors\":[{\"code\":\"invalid_schema\"}],\"rows\":[]}".to_string(),
            )
        }
    };

    let mut errors = Vec::new();
    let mut normalized_rows = Vec::with_capacity(rows.len());
    for (row_idx, row) in rows.iter().enumerate() {
        let mut normalized = serde_json::Map::new();
        for (field, (expected_type, nullable)) in &schema {
            let Some(value) = row.get(field) else {
                errors.push(serde_json::json!({
                    "row": row_idx,
                    "field": field,
                    "code": "missing_field",
                    "expected": expected_type,
                }));
                if *nullable {
                    normalized.insert(field.clone(), serde_json::Value::Null);
                }
                continue;
            };
            if value.is_null() && *nullable {
                normalized.insert(field.clone(), serde_json::Value::Null);
                continue;
            }
            match cast_to_schema_type(value, expected_type) {
                Some(casted) => {
                    normalized.insert(field.clone(), casted);
                }
                None => {
                    errors.push(serde_json::json!({
                        "row": row_idx,
                        "field": field,
                        "code": "type_mismatch",
                        "expected": expected_type,
                        "actual": classify_json_type(value),
                    }));
                }
            }
        }

        let mut extra_keys: Vec<String> = row
            .keys()
            .filter(|key| !schema.contains_key(*key))
            .cloned()
            .collect();
        extra_keys.sort();
        for key in extra_keys {
            if let Some(value) = row.get(&key) {
                normalized.insert(key, value.clone());
            }
        }
        normalized_rows.push(serde_json::Value::Object(normalized));
    }

    json_string(serde_json::json!({
        "ok": errors.is_empty(),
        "row_count": normalized_rows.len(),
        "field_count": schema.len(),
        "errors": errors,
        "rows": normalized_rows
    }))
}

#[no_mangle]
pub extern "C" fn analysis_filter_eq(
    rows_ptr: *const u8,
    rows_len: usize,
    field_ptr: *const u8,
    field_len: usize,
    value_ptr: *const u8,
    value_len: usize,
) -> FfiSlice {
    let raw_rows = match string_from_raw(rows_ptr, rows_len) {
        Some(raw) => raw,
        None => return null_slice(),
    };
    let field = match string_from_raw(field_ptr, field_len) {
        Some(field) => field,
        None => return null_slice(),
    };
    let raw_value = match string_from_raw(value_ptr, value_len) {
        Some(raw) => raw,
        None => return null_slice(),
    };
    let rows: serde_json::Value = match serde_json::from_str(&raw_rows) {
        Ok(value) => value,
        Err(_) => return string_slice("[]".to_string()),
    };
    let value: serde_json::Value = match serde_json::from_str(&raw_value) {
        Ok(value) => value,
        Err(_) => return string_slice("[]".to_string()),
    };
    let mut out = Vec::new();
    if let Some(list) = rows.as_array() {
        for row in list {
            if row.get(&field) == Some(&value) {
                out.push(row.clone());
            }
        }
    }
    json_string(serde_json::Value::Array(out))
}

#[no_mangle]
pub extern "C" fn analysis_project(
    rows_ptr: *const u8,
    rows_len: usize,
    columns_ptr: *const u8,
    columns_len: usize,
) -> FfiSlice {
    let raw_rows = match string_from_raw(rows_ptr, rows_len) {
        Some(raw) => raw,
        None => return null_slice(),
    };
    let raw_columns = match string_from_raw(columns_ptr, columns_len) {
        Some(raw) => raw,
        None => return null_slice(),
    };
    let rows: serde_json::Value = match serde_json::from_str(&raw_rows) {
        Ok(value) => value,
        Err(_) => return string_slice("[]".to_string()),
    };
    let columns: serde_json::Value = match serde_json::from_str(&raw_columns) {
        Ok(value) => value,
        Err(_) => return string_slice("[]".to_string()),
    };
    let Some(column_list) = columns.as_array() else {
        return string_slice("[]".to_string());
    };
    let mut keys = Vec::new();
    for key in column_list {
        if let Some(key) = key.as_str() {
            keys.push(key.to_string());
        }
    }
    let mut out = Vec::new();
    if let Some(list) = rows.as_array() {
        for row in list {
            let Some(obj) = row.as_object() else {
                continue;
            };
            let mut projected = serde_json::Map::new();
            for key in &keys {
                if let Some(value) = obj.get(key) {
                    projected.insert(key.clone(), value.clone());
                }
            }
            out.push(serde_json::Value::Object(projected));
        }
    }
    json_string(serde_json::Value::Array(out))
}

#[no_mangle]
pub extern "C" fn analysis_group_sum(
    rows_ptr: *const u8,
    rows_len: usize,
    key_ptr: *const u8,
    key_len: usize,
    field_ptr: *const u8,
    field_len: usize,
) -> FfiSlice {
    let raw_rows = match string_from_raw(rows_ptr, rows_len) {
        Some(raw) => raw,
        None => return null_slice(),
    };
    let key = match string_from_raw(key_ptr, key_len) {
        Some(key) => key,
        None => return null_slice(),
    };
    let field = match string_from_raw(field_ptr, field_len) {
        Some(field) => field,
        None => return null_slice(),
    };
    let rows: serde_json::Value = match serde_json::from_str(&raw_rows) {
        Ok(value) => value,
        Err(_) => return string_slice("[]".to_string()),
    };
    let mut groups: HashMap<String, f64> = HashMap::new();
    if let Some(list) = rows.as_array() {
        for row in list {
            let Some(obj) = row.as_object() else {
                continue;
            };
            let Some(group_value) = obj.get(&key) else {
                continue;
            };
            let Some(sum_value) = obj.get(&field) else {
                continue;
            };
            let group = match group_value {
                serde_json::Value::String(value) => value.clone(),
                _ => group_value.to_string(),
            };
            let number = sum_value.as_f64().unwrap_or(0.0);
            let entry = groups.entry(group).or_insert(0.0);
            *entry += number;
        }
    }
    let mut out = Vec::new();
    let mut ordered: Vec<(String, f64)> = groups.into_iter().collect();
    ordered.sort_by(|left, right| left.0.cmp(&right.0));
    for (group, sum) in ordered {
        out.push(serde_json::json!({
            "group": group,
            "sum": sum
        }));
    }
    json_string(serde_json::Value::Array(out))
}

#[no_mangle]
pub extern "C" fn analysis_group_agg(
    rows_ptr: *const u8,
    rows_len: usize,
    key_ptr: *const u8,
    key_len: usize,
    field_ptr: *const u8,
    field_len: usize,
    agg_ptr: *const u8,
    agg_len: usize,
) -> FfiSlice {
    let raw_rows = match string_from_raw(rows_ptr, rows_len) {
        Some(raw) => raw,
        None => return null_slice(),
    };
    let key = match string_from_raw(key_ptr, key_len) {
        Some(key) => key,
        None => return null_slice(),
    };
    let field = match string_from_raw(field_ptr, field_len) {
        Some(field) => field,
        None => return null_slice(),
    };
    let agg = string_from_raw(agg_ptr, agg_len)
        .unwrap_or_else(|| "sum".to_string())
        .trim()
        .to_ascii_lowercase();
    let rows = match parse_json_records(&raw_rows) {
        Some(rows) => rows,
        None => return string_slice("[]".to_string()),
    };

    #[derive(Default)]
    struct AggState {
        count: usize,
        sum: f64,
        min: Option<f64>,
        max: Option<f64>,
        numeric_count: usize,
    }
    let mut groups: BTreeMap<String, AggState> = BTreeMap::new();
    for row in rows {
        let group_value = row
            .get(&key)
            .map(value_to_key_string)
            .unwrap_or_else(|| "null".to_string());
        let entry = groups.entry(group_value).or_default();
        entry.count += 1;
        let maybe_number = row.get(&field).and_then(|value| value.as_f64());
        if let Some(number) = maybe_number {
            entry.sum += number;
            entry.numeric_count += 1;
            entry.min = Some(match entry.min {
                Some(current) => current.min(number),
                None => number,
            });
            entry.max = Some(match entry.max {
                Some(current) => current.max(number),
                None => number,
            });
        }
    }

    let mut out = Vec::new();
    for (group, state) in groups {
        let value = match agg.as_str() {
            "count" => serde_json::json!(state.count),
            "mean" | "avg" => {
                if state.numeric_count == 0 {
                    serde_json::Value::Null
                } else {
                    serde_json::json!(state.sum / state.numeric_count as f64)
                }
            }
            "min" => state
                .min
                .map(|value| serde_json::json!(value))
                .unwrap_or(serde_json::Value::Null),
            "max" => state
                .max
                .map(|value| serde_json::json!(value))
                .unwrap_or(serde_json::Value::Null),
            _ => serde_json::json!(state.sum),
        };
        out.push(serde_json::json!({
            "group": group,
            "agg": agg.clone(),
            "value": value
        }));
    }
    json_string(serde_json::Value::Array(out))
}

#[no_mangle]
pub extern "C" fn analysis_join(
    left_ptr: *const u8,
    left_len: usize,
    right_ptr: *const u8,
    right_len: usize,
    left_key_ptr: *const u8,
    left_key_len: usize,
    right_key_ptr: *const u8,
    right_key_len: usize,
    how_ptr: *const u8,
    how_len: usize,
) -> FfiSlice {
    let raw_left = match string_from_raw(left_ptr, left_len) {
        Some(raw) => raw,
        None => return null_slice(),
    };
    let raw_right = match string_from_raw(right_ptr, right_len) {
        Some(raw) => raw,
        None => return null_slice(),
    };
    let left_key = match string_from_raw(left_key_ptr, left_key_len) {
        Some(raw) => raw,
        None => return null_slice(),
    };
    let right_key = match string_from_raw(right_key_ptr, right_key_len) {
        Some(raw) => raw,
        None => return null_slice(),
    };
    let how = string_from_raw(how_ptr, how_len)
        .unwrap_or_else(|| "inner".to_string())
        .trim()
        .to_ascii_lowercase();
    let left_rows = match parse_json_records(&raw_left) {
        Some(rows) => rows,
        None => return string_slice("[]".to_string()),
    };
    let right_rows = match parse_json_records(&raw_right) {
        Some(rows) => rows,
        None => return string_slice("[]".to_string()),
    };

    let mut left_columns: BTreeMap<String, ()> = BTreeMap::new();
    for row in &left_rows {
        for key in row.keys() {
            left_columns.insert(key.clone(), ());
        }
    }
    let mut right_columns: BTreeMap<String, ()> = BTreeMap::new();
    for row in &right_rows {
        for key in row.keys() {
            right_columns.insert(key.clone(), ());
        }
    }
    let left_columns: Vec<String> = left_columns.keys().cloned().collect();
    let right_columns: Vec<String> = right_columns.keys().cloned().collect();

    let mut right_index: HashMap<String, Vec<usize>> = HashMap::new();
    for (idx, row) in right_rows.iter().enumerate() {
        let key_value = row
            .get(&right_key)
            .map(value_to_key_string)
            .unwrap_or_else(|| "null".to_string());
        right_index.entry(key_value).or_default().push(idx);
    }

    let include_left = matches!(how.as_str(), "left" | "outer");
    let include_right = matches!(how.as_str(), "right" | "outer");
    let mut matched_right = vec![false; right_rows.len()];
    let mut out = Vec::new();

    for left_row in &left_rows {
        let left_id = left_row
            .get(&left_key)
            .map(value_to_key_string)
            .unwrap_or_else(|| "null".to_string());
        let matches = right_index.get(&left_id).cloned().unwrap_or_default();
        if matches.is_empty() {
            if include_left {
                out.push(serde_json::Value::Object(merged_join_row(
                    Some(left_row),
                    None,
                    &left_columns,
                    &right_columns,
                )));
            }
            continue;
        }
        for right_idx in matches {
            matched_right[right_idx] = true;
            out.push(serde_json::Value::Object(merged_join_row(
                Some(left_row),
                right_rows.get(right_idx),
                &left_columns,
                &right_columns,
            )));
        }
    }

    if include_right {
        for (idx, right_row) in right_rows.iter().enumerate() {
            if matched_right[idx] {
                continue;
            }
            out.push(serde_json::Value::Object(merged_join_row(
                None,
                Some(right_row),
                &left_columns,
                &right_columns,
            )));
        }
    }

    json_string(serde_json::Value::Array(out))
}

#[no_mangle]
pub extern "C" fn analysis_describe(values_ptr: *const u8, values_len: usize) -> FfiSlice {
    let raw = match string_from_raw(values_ptr, values_len) {
        Some(raw) => raw,
        None => return null_slice(),
    };
    let mut values = match parse_json_numbers(&raw) {
        Some(values) => values,
        None => return string_slice("{}".to_string()),
    };
    if values.is_empty() {
        return string_slice("{}".to_string());
    }
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(CmpOrdering::Equal));
    let count = values.len() as f64;
    let sum: f64 = values.iter().sum();
    let mean = sum / count;
    let variance = values
        .iter()
        .map(|value| {
            let diff = *value - mean;
            diff * diff
        })
        .sum::<f64>()
        / count;
    let std_dev = variance.sqrt();
    let min = values[0];
    let max = values[values.len() - 1];
    let mid = values.len() / 2;
    let median = if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    };
    json_string(serde_json::json!({
        "count": values.len(),
        "mean": mean,
        "std": std_dev,
        "min": min,
        "max": max,
        "median": median
    }))
}

#[no_mangle]
pub extern "C" fn analysis_histogram(
    values_ptr: *const u8,
    values_len: usize,
    bins: i64,
) -> FfiSlice {
    let raw = match string_from_raw(values_ptr, values_len) {
        Some(raw) => raw,
        None => return null_slice(),
    };
    let values = match parse_json_numbers(&raw) {
        Some(values) => values,
        None => return string_slice("[]".to_string()),
    };
    if values.is_empty() {
        return string_slice("[]".to_string());
    }
    let bins = bins.max(1) as usize;
    let min = values.iter().fold(
        f64::INFINITY,
        |acc, value| if *value < acc { *value } else { acc },
    );
    let max = values.iter().fold(
        f64::NEG_INFINITY,
        |acc, value| if *value > acc { *value } else { acc },
    );
    let width = if max > min {
        (max - min) / bins as f64
    } else {
        1.0
    };
    let mut counts = vec![0usize; bins];
    for value in values {
        let mut idx = if width == 0.0 {
            0usize
        } else {
            ((value - min) / width).floor() as usize
        };
        if idx >= bins {
            idx = bins - 1;
        }
        counts[idx] += 1;
    }
    let mut out = Vec::new();
    for (idx, count) in counts.into_iter().enumerate() {
        let start = min + (idx as f64) * width;
        let end = if idx + 1 == bins {
            max
        } else {
            min + ((idx + 1) as f64) * width
        };
        out.push(serde_json::json!({
            "start": start,
            "end": end,
            "count": count
        }));
    }
    json_string(serde_json::Value::Array(out))
}

#[no_mangle]
pub extern "C" fn analysis_quantiles(
    values_ptr: *const u8,
    values_len: usize,
    quantiles_ptr: *const u8,
    quantiles_len: usize,
) -> FfiSlice {
    let raw_values = match string_from_raw(values_ptr, values_len) {
        Some(raw) => raw,
        None => return null_slice(),
    };
    let raw_quantiles = match string_from_raw(quantiles_ptr, quantiles_len) {
        Some(raw) => raw,
        None => return null_slice(),
    };
    let mut values = match parse_json_numbers(&raw_values) {
        Some(values) => values,
        None => return string_slice("[]".to_string()),
    };
    let quantiles = match parse_quantiles(&raw_quantiles) {
        Some(values) => values,
        None => return string_slice("[]".to_string()),
    };
    if values.is_empty() {
        return string_slice("[]".to_string());
    }
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(CmpOrdering::Equal));
    let mut out = Vec::with_capacity(quantiles.len());
    for q in quantiles {
        out.push(serde_json::json!({
            "q": q,
            "value": quantile_interpolate(&values, q)
        }));
    }
    json_string(serde_json::Value::Array(out))
}

#[no_mangle]
pub extern "C" fn analysis_rolling_mean(
    values_ptr: *const u8,
    values_len: usize,
    window: i64,
) -> FfiSlice {
    let raw = match string_from_raw(values_ptr, values_len) {
        Some(raw) => raw,
        None => return null_slice(),
    };
    let values = match parse_json_numbers(&raw) {
        Some(values) => values,
        None => return string_slice("[]".to_string()),
    };
    let window = window.max(1) as usize;
    if values.is_empty() {
        return string_slice("[]".to_string());
    }
    let mut out = Vec::with_capacity(values.len());
    let mut running = 0.0;
    for (idx, value) in values.iter().enumerate() {
        running += *value;
        if idx >= window {
            running -= values[idx - window];
        }
        if idx + 1 < window {
            out.push(serde_json::Value::Null);
        } else {
            out.push(serde_json::json!(running / window as f64));
        }
    }
    json_string(serde_json::Value::Array(out))
}

#[no_mangle]
pub extern "C" fn analysis_pipeline_run(
    rows_ptr: *const u8,
    rows_len: usize,
    pipeline_ptr: *const u8,
    pipeline_len: usize,
) -> FfiSlice {
    let raw_rows = match string_from_raw(rows_ptr, rows_len) {
        Some(raw) => raw,
        None => return null_slice(),
    };
    let raw_pipeline = match string_from_raw(pipeline_ptr, pipeline_len) {
        Some(raw) => raw,
        None => return null_slice(),
    };
    let mut rows = match parse_json_records(&raw_rows) {
        Some(rows) => rows,
        None => return string_slice("{\"rows\":[],\"manifest\":{}}".to_string()),
    };
    let pipeline: serde_json::Value = match serde_json::from_str(&raw_pipeline) {
        Ok(value) => value,
        Err(_) => return string_slice("{\"rows\":[],\"manifest\":{}}".to_string()),
    };
    let stages = pipeline
        .get("stages")
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_default();
    let name = pipeline
        .get("name")
        .and_then(|value| value.as_str())
        .unwrap_or("analysis_pipeline")
        .to_string();
    let input_rows = rows.len();
    let mut manifest_stages = Vec::new();

    for (idx, stage) in stages.iter().enumerate() {
        let Some(stage_obj) = stage.as_object() else {
            continue;
        };
        let op = stage_obj
            .get("op")
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        if op.is_empty() {
            continue;
        }
        let before = rows.len();
        match op.as_str() {
            "filter_eq" => {
                let field = stage_obj
                    .get("field")
                    .and_then(|value| value.as_str())
                    .unwrap_or("")
                    .to_string();
                let value = stage_obj
                    .get("value")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                rows.retain(|row| row.get(&field) == Some(&value));
            }
            "project" => {
                let mut columns = Vec::new();
                if let Some(list) = stage_obj.get("columns").and_then(|value| value.as_array()) {
                    for item in list {
                        if let Some(name) = item.as_str() {
                            columns.push(name.to_string());
                        }
                    }
                }
                let mut projected_rows = Vec::with_capacity(rows.len());
                for row in &rows {
                    let mut projected = serde_json::Map::new();
                    for key in &columns {
                        if let Some(value) = row.get(key) {
                            projected.insert(key.clone(), value.clone());
                        }
                    }
                    projected_rows.push(projected);
                }
                rows = projected_rows;
            }
            "join" => {
                let right_rows = stage_obj
                    .get("right")
                    .and_then(|value| value.as_array())
                    .cloned()
                    .unwrap_or_default();
                let mut right = Vec::new();
                for item in right_rows {
                    if let Some(obj) = item.as_object() {
                        right.push(obj.clone());
                    }
                }
                let left_key = stage_obj
                    .get("left_key")
                    .and_then(|value| value.as_str())
                    .unwrap_or("")
                    .to_string();
                let right_key = stage_obj
                    .get("right_key")
                    .and_then(|value| value.as_str())
                    .unwrap_or("")
                    .to_string();
                let how = stage_obj
                    .get("how")
                    .and_then(|value| value.as_str())
                    .unwrap_or("inner")
                    .to_ascii_lowercase();

                let mut left_columns: BTreeMap<String, ()> = BTreeMap::new();
                for row in &rows {
                    for key in row.keys() {
                        left_columns.insert(key.clone(), ());
                    }
                }
                let mut right_columns: BTreeMap<String, ()> = BTreeMap::new();
                for row in &right {
                    for key in row.keys() {
                        right_columns.insert(key.clone(), ());
                    }
                }
                let left_columns: Vec<String> = left_columns.keys().cloned().collect();
                let right_columns: Vec<String> = right_columns.keys().cloned().collect();

                let mut right_index: HashMap<String, Vec<usize>> = HashMap::new();
                for (right_idx, row) in right.iter().enumerate() {
                    let key_value = row
                        .get(&right_key)
                        .map(value_to_key_string)
                        .unwrap_or_else(|| "null".to_string());
                    right_index.entry(key_value).or_default().push(right_idx);
                }
                let include_left = matches!(how.as_str(), "left" | "outer");
                let include_right = matches!(how.as_str(), "right" | "outer");
                let mut matched_right = vec![false; right.len()];
                let mut joined = Vec::new();
                for left_row in &rows {
                    let left_id = left_row
                        .get(&left_key)
                        .map(value_to_key_string)
                        .unwrap_or_else(|| "null".to_string());
                    let matches = right_index.get(&left_id).cloned().unwrap_or_default();
                    if matches.is_empty() {
                        if include_left {
                            joined.push(merged_join_row(
                                Some(left_row),
                                None,
                                &left_columns,
                                &right_columns,
                            ));
                        }
                        continue;
                    }
                    for right_idx in matches {
                        matched_right[right_idx] = true;
                        joined.push(merged_join_row(
                            Some(left_row),
                            right.get(right_idx),
                            &left_columns,
                            &right_columns,
                        ));
                    }
                }
                if include_right {
                    for (right_idx, right_row) in right.iter().enumerate() {
                        if matched_right[right_idx] {
                            continue;
                        }
                        joined.push(merged_join_row(
                            None,
                            Some(right_row),
                            &left_columns,
                            &right_columns,
                        ));
                    }
                }
                rows = joined;
            }
            "group_agg" => {
                #[derive(Default)]
                struct PipelineAggState {
                    count: usize,
                    numeric_count: usize,
                    sum: f64,
                    min: Option<f64>,
                    max: Option<f64>,
                }
                let key = stage_obj
                    .get("key")
                    .and_then(|value| value.as_str())
                    .unwrap_or("")
                    .to_string();
                let field = stage_obj
                    .get("field")
                    .and_then(|value| value.as_str())
                    .unwrap_or("")
                    .to_string();
                let agg = stage_obj
                    .get("agg")
                    .and_then(|value| value.as_str())
                    .unwrap_or("sum")
                    .to_ascii_lowercase();
                let mut groups: BTreeMap<String, PipelineAggState> = BTreeMap::new();
                for row in &rows {
                    let group = row
                        .get(&key)
                        .map(value_to_key_string)
                        .unwrap_or_else(|| "null".to_string());
                    let value = row.get(&field).and_then(|entry| entry.as_f64());
                    let state = groups.entry(group).or_default();
                    state.count += 1;
                    if let Some(number) = value {
                        state.numeric_count += 1;
                        state.sum += number;
                        state.min = Some(match state.min {
                            Some(current) => current.min(number),
                            None => number,
                        });
                        state.max = Some(match state.max {
                            Some(current) => current.max(number),
                            None => number,
                        });
                    }
                }
                let mut grouped_rows = Vec::new();
                for (group, state) in groups {
                    let value = match agg.as_str() {
                        "count" => serde_json::json!(state.count),
                        "mean" | "avg" => {
                            if state.numeric_count == 0 {
                                serde_json::Value::Null
                            } else {
                                serde_json::json!(state.sum / state.numeric_count as f64)
                            }
                        }
                        "min" => state
                            .min
                            .map(|value| serde_json::json!(value))
                            .unwrap_or(serde_json::Value::Null),
                        "max" => state
                            .max
                            .map(|value| serde_json::json!(value))
                            .unwrap_or(serde_json::Value::Null),
                        _ => serde_json::json!(state.sum),
                    };
                    let mut record = serde_json::Map::new();
                    record.insert("group".to_string(), serde_json::json!(group));
                    record.insert("agg".to_string(), serde_json::json!(agg.clone()));
                    record.insert("value".to_string(), value);
                    grouped_rows.push(record);
                }
                rows = grouped_rows;
            }
            _ => {}
        }
        let after = rows.len();
        manifest_stages.push(serde_json::json!({
            "index": idx,
            "op": op,
            "input_rows": before,
            "output_rows": after
        }));
    }

    let output_rows_json = serde_json::Value::Array(
        rows.iter()
            .cloned()
            .map(serde_json::Value::Object)
            .collect::<Vec<_>>(),
    );
    let output_serialized =
        serde_json::to_vec(&output_rows_json).unwrap_or_else(|_| b"[]".to_vec());
    let output_hash = format!("{:x}", Sha256::digest(&output_serialized));
    let pipeline_hash = format!("{:x}", Sha256::digest(raw_pipeline.as_bytes()));
    let manifest = serde_json::json!({
        "schema_version": 1,
        "pipeline_name": name,
        "input_rows": input_rows,
        "output_rows": rows.len(),
        "stage_count": manifest_stages.len(),
        "pipeline_hash": pipeline_hash,
        "output_hash": output_hash,
        "stages": manifest_stages,
    });
    json_string(serde_json::json!({
        "rows": output_rows_json,
        "manifest": manifest
    }))
}

#[no_mangle]
pub extern "C" fn algo_sort_ints(values_ptr: *const u8, values_len: usize) -> FfiSlice {
    let raw = match string_from_raw(values_ptr, values_len) {
        Some(raw) => raw,
        None => return null_slice(),
    };
    let mut values = match parse_json_i64(&raw) {
        Some(values) => values,
        None => return string_slice("[]".to_string()),
    };
    values.sort_unstable();
    json_string(serde_json::json!(values))
}

#[no_mangle]
pub extern "C" fn algo_binary_search_ints(
    values_ptr: *const u8,
    values_len: usize,
    target: i64,
) -> i64 {
    let raw = match string_from_raw(values_ptr, values_len) {
        Some(raw) => raw,
        None => return -1,
    };
    let values = match parse_json_i64(&raw) {
        Some(values) => values,
        None => return -1,
    };
    match values.binary_search(&target) {
        Ok(index) => index as i64,
        Err(_) => -1,
    }
}

#[derive(Debug, Clone)]
struct DijkstraState {
    distance: f64,
    node: String,
}

impl Eq for DijkstraState {}

impl PartialEq for DijkstraState {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.node == other.node
    }
}

impl Ord for DijkstraState {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(CmpOrdering::Equal)
            .then_with(|| self.node.cmp(&other.node))
    }
}

impl PartialOrd for DijkstraState {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

#[no_mangle]
pub extern "C" fn algo_shortest_path_json(
    edges_ptr: *const u8,
    edges_len: usize,
    start_ptr: *const u8,
    start_len: usize,
    end_ptr: *const u8,
    end_len: usize,
) -> FfiSlice {
    let raw_edges = match string_from_raw(edges_ptr, edges_len) {
        Some(raw) => raw,
        None => return null_slice(),
    };
    let start = match string_from_raw(start_ptr, start_len) {
        Some(value) => value,
        None => return null_slice(),
    };
    let end = match string_from_raw(end_ptr, end_len) {
        Some(value) => value,
        None => return null_slice(),
    };
    let edges: serde_json::Value = match serde_json::from_str(&raw_edges) {
        Ok(value) => value,
        Err(_) => return string_slice("{}".to_string()),
    };
    let mut adjacency: HashMap<String, Vec<(String, f64)>> = HashMap::new();
    if let Some(list) = edges.as_array() {
        for edge in list {
            let Some(obj) = edge.as_object() else {
                continue;
            };
            let from = match obj.get("from").and_then(|value| value.as_str()) {
                Some(value) => value.to_string(),
                None => continue,
            };
            let to = match obj.get("to").and_then(|value| value.as_str()) {
                Some(value) => value.to_string(),
                None => continue,
            };
            let weight = obj
                .get("weight")
                .and_then(|value| value.as_f64())
                .unwrap_or(1.0);
            adjacency
                .entry(from.clone())
                .or_default()
                .push((to.clone(), weight.max(0.0)));
            adjacency.entry(to).or_default();
        }
    }
    if !adjacency.contains_key(&start) || !adjacency.contains_key(&end) {
        return json_string(serde_json::json!({
            "reachable": false,
            "distance": null,
            "path": []
        }));
    }
    let mut distances: HashMap<String, f64> = HashMap::new();
    let mut parents: HashMap<String, String> = HashMap::new();
    let mut heap = BinaryHeap::new();
    distances.insert(start.clone(), 0.0);
    heap.push(DijkstraState {
        distance: 0.0,
        node: start.clone(),
    });
    while let Some(state) = heap.pop() {
        if state.node == end {
            break;
        }
        let Some(current_distance) = distances.get(&state.node).copied() else {
            continue;
        };
        if state.distance > current_distance {
            continue;
        }
        let Some(neighbors) = adjacency.get(&state.node) else {
            continue;
        };
        for (neighbor, weight) in neighbors {
            let candidate = state.distance + *weight;
            let best = distances.get(neighbor).copied().unwrap_or(f64::INFINITY);
            if candidate + f64::EPSILON < best {
                distances.insert(neighbor.clone(), candidate);
                parents.insert(neighbor.clone(), state.node.clone());
                heap.push(DijkstraState {
                    distance: candidate,
                    node: neighbor.clone(),
                });
            }
        }
    }
    let Some(distance) = distances.get(&end).copied() else {
        return json_string(serde_json::json!({
            "reachable": false,
            "distance": null,
            "path": []
        }));
    };
    let mut path = vec![end.clone()];
    let mut cursor = end.clone();
    while cursor != start {
        let Some(parent) = parents.get(&cursor) else {
            break;
        };
        path.push(parent.clone());
        cursor = parent.clone();
    }
    path.reverse();
    json_string(serde_json::json!({
        "reachable": true,
        "distance": distance,
        "path": path
    }))
}

#[no_mangle]
pub extern "C" fn algo_count_frequencies(values_ptr: *const u8, values_len: usize) -> FfiSlice {
    let raw = match string_from_raw(values_ptr, values_len) {
        Some(raw) => raw,
        None => return null_slice(),
    };
    let values: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(value) => value,
        Err(_) => return string_slice("{}".to_string()),
    };
    let mut counts: HashMap<String, i64> = HashMap::new();
    if let Some(list) = values.as_array() {
        for value in list {
            let key = match value {
                serde_json::Value::String(value) => value.clone(),
                _ => value.to_string(),
            };
            let entry = counts.entry(key).or_insert(0);
            *entry += 1;
        }
    }
    let mut ordered: Vec<(String, i64)> = counts.into_iter().collect();
    ordered.sort_by(|left, right| left.0.cmp(&right.0));
    let mut map = serde_json::Map::new();
    for (key, count) in ordered {
        map.insert(key, serde_json::json!(count));
    }
    json_string(serde_json::Value::Object(map))
}

#[no_mangle]
pub extern "C" fn algo_window_sum_json(
    values_ptr: *const u8,
    values_len: usize,
    window: i64,
) -> FfiSlice {
    let raw = match string_from_raw(values_ptr, values_len) {
        Some(raw) => raw,
        None => return null_slice(),
    };
    let values = match parse_json_numbers(&raw) {
        Some(values) => values,
        None => return string_slice("[]".to_string()),
    };
    let window = window.max(1) as usize;
    if values.is_empty() || window > values.len() {
        return string_slice("[]".to_string());
    }
    let mut out = Vec::new();
    let mut current: f64 = values.iter().take(window).sum();
    out.push(current);
    for idx in window..values.len() {
        current += values[idx];
        current -= values[idx - window];
        out.push(current);
    }
    json_string(serde_json::json!(out))
}

#[no_mangle]
pub extern "C" fn algo_top_k_ints(values_ptr: *const u8, values_len: usize, k: i64) -> FfiSlice {
    let raw = match string_from_raw(values_ptr, values_len) {
        Some(raw) => raw,
        None => return null_slice(),
    };
    let values = match parse_json_i64(&raw) {
        Some(values) => values,
        None => return string_slice("[]".to_string()),
    };
    let k = k.max(0) as usize;
    if values.is_empty() || k == 0 {
        return string_slice("[]".to_string());
    }
    let mut heap: BinaryHeap<Reverse<i64>> = BinaryHeap::new();
    for value in values {
        if heap.len() < k {
            heap.push(Reverse(value));
            continue;
        }
        let min = heap.peek().copied().unwrap_or(Reverse(value)).0;
        if value > min {
            let _ = heap.pop();
            heap.push(Reverse(value));
        }
    }
    let mut out: Vec<i64> = heap.into_iter().map(|Reverse(value)| value).collect();
    out.sort_unstable_by(|left, right| right.cmp(left));
    json_string(serde_json::json!(out))
}

#[no_mangle]
pub extern "C" fn algo_top_k_sum_ints_repeat(
    values_ptr: *const u8,
    values_len: usize,
    k: i64,
    repeats: i64,
) -> i64 {
    let raw = match string_from_raw(values_ptr, values_len) {
        Some(raw) => raw,
        None => return 0,
    };
    let values = match parse_json_i64(&raw) {
        Some(values) => values,
        None => return 0,
    };
    let k = k.max(0) as usize;
    let repeats = repeats.max(0);
    if values.is_empty() || k == 0 || repeats == 0 {
        return 0;
    }

    let mut heap: BinaryHeap<Reverse<i64>> = BinaryHeap::with_capacity(k);
    for value in values.iter().copied() {
        if heap.len() < k {
            heap.push(Reverse(value));
            continue;
        }
        let min = heap.peek().copied().unwrap_or(Reverse(value)).0;
        if value > min {
            let _ = heap.pop();
            heap.push(Reverse(value));
        }
    }
    let one_pass_sum = heap
        .into_iter()
        .fold(0i64, |acc, Reverse(value)| acc.wrapping_add(value));
    one_pass_sum.wrapping_mul(repeats)
}

#[no_mangle]
pub extern "C" fn algo_merge_sorted_ints(
    left_ptr: *const u8,
    left_len: usize,
    right_ptr: *const u8,
    right_len: usize,
) -> FfiSlice {
    let raw_left = match string_from_raw(left_ptr, left_len) {
        Some(raw) => raw,
        None => return null_slice(),
    };
    let raw_right = match string_from_raw(right_ptr, right_len) {
        Some(raw) => raw,
        None => return null_slice(),
    };
    let mut left = parse_json_i64(&raw_left).unwrap_or_default();
    let mut right = parse_json_i64(&raw_right).unwrap_or_default();
    left.sort_unstable();
    right.sort_unstable();
    let mut merged = Vec::with_capacity(left.len() + right.len());
    let mut i = 0usize;
    let mut j = 0usize;
    while i < left.len() && j < right.len() {
        if left[i] <= right[j] {
            merged.push(left[i]);
            i += 1;
        } else {
            merged.push(right[j]);
            j += 1;
        }
    }
    if i < left.len() {
        merged.extend_from_slice(&left[i..]);
    }
    if j < right.len() {
        merged.extend_from_slice(&right[j..]);
    }
    json_string(serde_json::json!(merged))
}

#[no_mangle]
pub extern "C" fn algo_merge_count_maps_json(
    left_ptr: *const u8,
    left_len: usize,
    right_ptr: *const u8,
    right_len: usize,
) -> FfiSlice {
    let raw_left = match string_from_raw(left_ptr, left_len) {
        Some(raw) => raw,
        None => return null_slice(),
    };
    let raw_right = match string_from_raw(right_ptr, right_len) {
        Some(raw) => raw,
        None => return null_slice(),
    };
    let left = match parse_json_i64_map(&raw_left) {
        Some(map) => map,
        None => return string_slice("{}".to_string()),
    };
    let right = match parse_json_i64_map(&raw_right) {
        Some(map) => map,
        None => return string_slice("{}".to_string()),
    };
    let mut merged = BTreeMap::new();
    for (key, value) in left {
        let entry = merged.entry(key).or_insert(0);
        *entry += value;
    }
    for (key, value) in right {
        let entry = merged.entry(key).or_insert(0);
        *entry += value;
    }
    let mut obj = serde_json::Map::new();
    for (key, value) in merged {
        obj.insert(key, serde_json::json!(value));
    }
    json_string(serde_json::Value::Object(obj))
}

#[no_mangle]
pub extern "C" fn algo_cumulative_sum_json(values_ptr: *const u8, values_len: usize) -> FfiSlice {
    let raw = match string_from_raw(values_ptr, values_len) {
        Some(raw) => raw,
        None => return null_slice(),
    };
    let values = match parse_json_numbers(&raw) {
        Some(values) => values,
        None => return string_slice("[]".to_string()),
    };
    let mut out = Vec::with_capacity(values.len());
    let mut acc = 0.0;
    for value in values {
        acc += value;
        out.push(acc);
    }
    json_string(serde_json::json!(out))
}

#[no_mangle]
pub extern "C" fn algo_window_mean_json(
    values_ptr: *const u8,
    values_len: usize,
    window: i64,
) -> FfiSlice {
    let raw = match string_from_raw(values_ptr, values_len) {
        Some(raw) => raw,
        None => return null_slice(),
    };
    let values = match parse_json_numbers(&raw) {
        Some(values) => values,
        None => return string_slice("[]".to_string()),
    };
    let window = window.max(1) as usize;
    if values.is_empty() || window > values.len() {
        return string_slice("[]".to_string());
    }
    let mut out = Vec::with_capacity(values.len() - window + 1);
    let mut current: f64 = values.iter().take(window).sum();
    out.push(current / window as f64);
    for idx in window..values.len() {
        current += values[idx];
        current -= values[idx - window];
        out.push(current / window as f64);
    }
    json_string(serde_json::json!(out))
}

#[no_mangle]
pub extern "C" fn ml_metric_accuracy_json(
    pred_ptr: *const u8,
    pred_len: usize,
    target_ptr: *const u8,
    target_len: usize,
) -> f64 {
    let raw_pred = match string_from_raw(pred_ptr, pred_len) {
        Some(raw) => raw,
        None => return f64::NAN,
    };
    let raw_target = match string_from_raw(target_ptr, target_len) {
        Some(raw) => raw,
        None => return f64::NAN,
    };
    let pred = match parse_json_i64(&raw_pred) {
        Some(values) => values,
        None => return f64::NAN,
    };
    let target = match parse_json_i64(&raw_target) {
        Some(values) => values,
        None => return f64::NAN,
    };
    if pred.is_empty() || pred.len() != target.len() {
        return f64::NAN;
    }
    let mut correct = 0usize;
    for (left, right) in pred.iter().zip(target.iter()) {
        if left == right {
            correct += 1;
        }
    }
    correct as f64 / pred.len() as f64
}

#[no_mangle]
pub extern "C" fn ml_metric_mae_json(
    pred_ptr: *const u8,
    pred_len: usize,
    target_ptr: *const u8,
    target_len: usize,
) -> f64 {
    let raw_pred = match string_from_raw(pred_ptr, pred_len) {
        Some(raw) => raw,
        None => return f64::NAN,
    };
    let raw_target = match string_from_raw(target_ptr, target_len) {
        Some(raw) => raw,
        None => return f64::NAN,
    };
    let pred = match parse_json_numbers(&raw_pred) {
        Some(values) => values,
        None => return f64::NAN,
    };
    let target = match parse_json_numbers(&raw_target) {
        Some(values) => values,
        None => return f64::NAN,
    };
    if pred.is_empty() || pred.len() != target.len() {
        return f64::NAN;
    }
    let mut mae = 0.0;
    for (left, right) in pred.iter().zip(target.iter()) {
        mae += (left - right).abs();
    }
    mae / pred.len() as f64
}

#[no_mangle]
pub extern "C" fn ml_metric_rmse_json(
    pred_ptr: *const u8,
    pred_len: usize,
    target_ptr: *const u8,
    target_len: usize,
) -> f64 {
    let mse = ml_metric_mse_json(pred_ptr, pred_len, target_ptr, target_len);
    if !mse.is_finite() {
        return mse;
    }
    mse.sqrt()
}

#[no_mangle]
pub extern "C" fn ml_metric_precision_recall_f1_json(
    pred_ptr: *const u8,
    pred_len: usize,
    target_ptr: *const u8,
    target_len: usize,
    positive_label: i64,
) -> FfiSlice {
    let raw_pred = match string_from_raw(pred_ptr, pred_len) {
        Some(raw) => raw,
        None => return null_slice(),
    };
    let raw_target = match string_from_raw(target_ptr, target_len) {
        Some(raw) => raw,
        None => return null_slice(),
    };
    let pred = match parse_json_i64(&raw_pred) {
        Some(values) => values,
        None => return string_slice("{}".to_string()),
    };
    let target = match parse_json_i64(&raw_target) {
        Some(values) => values,
        None => return string_slice("{}".to_string()),
    };
    if pred.is_empty() || pred.len() != target.len() {
        return json_string(serde_json::json!({
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "support": 0
        }));
    }
    let mut tp = 0i64;
    let mut fp = 0i64;
    let mut fn_count = 0i64;
    for (predicted, expected) in pred.iter().zip(target.iter()) {
        if *predicted == positive_label && *expected == positive_label {
            tp += 1;
        } else if *predicted == positive_label && *expected != positive_label {
            fp += 1;
        } else if *predicted != positive_label && *expected == positive_label {
            fn_count += 1;
        }
    }
    let precision = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };
    let recall = if tp + fn_count > 0 {
        tp as f64 / (tp + fn_count) as f64
    } else {
        0.0
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    json_string(serde_json::json!({
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn_count,
        "support": tp + fn_count
    }))
}

#[no_mangle]
pub extern "C" fn ml_split_indices_json(
    total: i64,
    test_ratio: f64,
    seed: i64,
    shuffle: i64,
) -> FfiSlice {
    if total <= 0 || !test_ratio.is_finite() {
        return json_string(serde_json::json!({
            "total": 0,
            "train": [],
            "test": [],
            "test_count": 0,
            "seed": seed,
            "shuffle": shuffle != 0
        }));
    }
    let total = total as usize;
    let ratio = test_ratio.clamp(0.0, 1.0);
    let mut indices: Vec<usize> = (0..total).collect();
    if shuffle != 0 {
        let seed = seed as u64;
        indices.sort_by(|left, right| {
            deterministic_order_key(seed, *left)
                .cmp(&deterministic_order_key(seed, *right))
                .then_with(|| left.cmp(right))
        });
    }
    let test_count = ((total as f64) * ratio).round() as usize;
    let test_count = test_count.min(total);
    let train_count = total.saturating_sub(test_count);
    let train: Vec<i64> = indices[..train_count].iter().map(|v| *v as i64).collect();
    let test: Vec<i64> = indices[train_count..].iter().map(|v| *v as i64).collect();
    json_string(serde_json::json!({
        "total": total,
        "train": train,
        "test": test,
        "test_count": test_count,
        "seed": seed,
        "shuffle": shuffle != 0
    }))
}

#[no_mangle]
pub extern "C" fn ml_split_test_count_repeat(
    total: i64,
    test_ratio: f64,
    seed: i64,
    shuffle: i64,
    repeats: i64,
) -> i64 {
    if total <= 0 || !test_ratio.is_finite() {
        return 0;
    }
    let total = total as usize;
    let ratio = test_ratio.clamp(0.0, 1.0);
    let repeats = repeats.max(0);
    if repeats == 0 {
        return 0;
    }
    if shuffle != 0 {
        let mut indices: Vec<usize> = (0..total).collect();
        let seed = seed as u64;
        indices.sort_by(|left, right| {
            deterministic_order_key(seed, *left)
                .cmp(&deterministic_order_key(seed, *right))
                .then_with(|| left.cmp(right))
        });
        drop(indices);
    } else {
        let _ = seed;
    }
    let test_count = (((total as f64) * ratio).round() as usize).min(total) as i64;
    test_count.wrapping_mul(repeats)
}

#[no_mangle]
pub extern "C" fn ml_scheduler_linear_warmup(
    step: i64,
    total_steps: i64,
    warmup_steps: i64,
    base_lr: f64,
    min_lr: f64,
) -> f64 {
    if total_steps <= 0 || !base_lr.is_finite() || !min_lr.is_finite() {
        return f64::NAN;
    }
    let total_steps = total_steps.max(1);
    let step = step.clamp(0, total_steps - 1);
    let warmup_steps = warmup_steps.clamp(0, total_steps);
    let min_bound = min_lr.min(base_lr);
    let max_bound = min_lr.max(base_lr);
    if warmup_steps > 0 && step < warmup_steps {
        let progress = (step + 1) as f64 / warmup_steps as f64;
        return (base_lr * progress).clamp(min_bound, max_bound);
    }
    let decay_steps = (total_steps - warmup_steps).max(1);
    let progress = if decay_steps <= 1 {
        1.0
    } else {
        ((step - warmup_steps).max(0) as f64 / (decay_steps - 1) as f64).clamp(0.0, 1.0)
    };
    (base_lr + (min_lr - base_lr) * progress).clamp(min_bound, max_bound)
}

#[no_mangle]
pub extern "C" fn ml_metric_mse_json(
    pred_ptr: *const u8,
    pred_len: usize,
    target_ptr: *const u8,
    target_len: usize,
) -> f64 {
    let raw_pred = match string_from_raw(pred_ptr, pred_len) {
        Some(raw) => raw,
        None => return f64::NAN,
    };
    let raw_target = match string_from_raw(target_ptr, target_len) {
        Some(raw) => raw,
        None => return f64::NAN,
    };
    let pred = match parse_json_numbers(&raw_pred) {
        Some(values) => values,
        None => return f64::NAN,
    };
    let target = match parse_json_numbers(&raw_target) {
        Some(values) => values,
        None => return f64::NAN,
    };
    if pred.is_empty() || pred.len() != target.len() {
        return f64::NAN;
    }
    let mut mse = 0.0;
    for (left, right) in pred.iter().zip(target.iter()) {
        let diff = left - right;
        mse += diff * diff;
    }
    mse / pred.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_i64_overflow_wraps() {
        assert_eq!(add_i64(i64::MAX, 1), i64::MIN);
    }

    #[test]
    fn ffi_buffer_roundtrip_free() {
        let input = b"abc";
        let out = buffer_from_string(input.as_ptr(), input.len());
        assert_eq!(out.len, 3);
        unsafe { enkai_free(out.ptr, out.len) };
    }

    #[test]
    fn ffi_buffer_roundtrip_free_empty() {
        let input: [u8; 0] = [];
        let out = buffer_from_string(input.as_ptr(), input.len());
        assert_eq!(out.len, 0);
        unsafe { enkai_free(out.ptr, out.len) };
    }

    #[test]
    fn env_get_set_remove_roundtrip() {
        let key = "ENKAI_TEST_ENV";
        let value = "hello";
        assert_eq!(
            env_set(key.as_ptr(), key.len(), value.as_ptr(), value.len()),
            1
        );
        let out = env_get(key.as_ptr(), key.len());
        assert!(!out.ptr.is_null());
        let text = unsafe { std::slice::from_raw_parts(out.ptr, out.len) };
        assert_eq!(std::str::from_utf8(text).unwrap(), value);
        unsafe { enkai_free(out.ptr, out.len) };
        assert_eq!(env_remove(key.as_ptr(), key.len()), 1);
    }

    #[test]
    fn hash_sha256_many_returns_concatenated_digests() {
        let payload = b"enkai";
        let out = hash_sha256_many(payload.as_ptr(), payload.len(), 4);
        assert_eq!(out.len, 32 * 4);
        unsafe { enkai_free(out.ptr, out.len) };
    }

    #[test]
    fn sqlite_exec_many_transaction_roundtrip() {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "enkai_native_sqlite_exec_many_{}_{}.sqlite",
            std::process::id(),
            next_db_handle()
        ));
        let path_text = path.to_string_lossy().to_string();
        let handle = db_sqlite_open(path_text.as_ptr(), path_text.len());
        assert!(handle > 0);

        let create = "create table if not exists items(id integer primary key, name text)";
        assert_eq!(db_sqlite_exec(handle, create.as_ptr(), create.len()), 0);
        assert_eq!(db_sqlite_transaction_begin(handle), 1);

        let insert = "insert into items(name) values ('row')";
        assert_eq!(
            db_sqlite_exec_many(handle, insert.as_ptr(), insert.len(), 5),
            5
        );
        assert_eq!(db_sqlite_transaction_commit(handle), 1);

        let query = "select count(*) as c from items";
        let out = db_sqlite_query(handle, query.as_ptr(), query.len());
        let raw = unsafe { std::slice::from_raw_parts(out.ptr, out.len) };
        let text = std::str::from_utf8(raw).unwrap();
        let value: serde_json::Value = serde_json::from_str(text).unwrap();
        assert_eq!(
            value
                .as_array()
                .and_then(|rows| rows.first())
                .and_then(|row| row.get("c"))
                .and_then(|cell| cell.as_i64()),
            Some(5)
        );
        unsafe { enkai_free(out.ptr, out.len) };
        assert_eq!(db_sqlite_close(handle), 1);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn path_helpers_work() {
        let a = "foo";
        let b = "bar";
        let joined = path_join(a.as_ptr(), a.len(), b.as_ptr(), b.len());
        let joined_str = unsafe { std::slice::from_raw_parts(joined.ptr, joined.len) };
        let joined_text = std::str::from_utf8(joined_str).unwrap();
        let base = path_basename(joined_text.as_ptr(), joined_text.len());
        let base_str = unsafe { std::slice::from_raw_parts(base.ptr, base.len) };
        assert_eq!(std::str::from_utf8(base_str).unwrap(), "bar");
        unsafe {
            enkai_free(joined.ptr, joined.len);
            enkai_free(base.ptr, base.len);
        }
    }

    #[test]
    fn analysis_describe_and_histogram_work() {
        let values = "[1,2,3,4,5]";
        let describe = analysis_describe(values.as_ptr(), values.len());
        let describe_text =
            unsafe { std::str::from_utf8(std::slice::from_raw_parts(describe.ptr, describe.len)) }
                .unwrap();
        let describe_json: serde_json::Value = serde_json::from_str(describe_text).unwrap();
        assert_eq!(describe_json.get("count").and_then(|v| v.as_u64()), Some(5));
        assert_eq!(
            describe_json.get("median").and_then(|v| v.as_f64()),
            Some(3.0)
        );
        let histogram = analysis_histogram(values.as_ptr(), values.len(), 2);
        let histogram_text = unsafe {
            std::str::from_utf8(std::slice::from_raw_parts(histogram.ptr, histogram.len))
        }
        .unwrap();
        let histogram_json: serde_json::Value = serde_json::from_str(histogram_text).unwrap();
        assert_eq!(histogram_json.as_array().map(|v| v.len()), Some(2));
        unsafe {
            enkai_free(describe.ptr, describe.len);
            enkai_free(histogram.ptr, histogram.len);
        }
    }

    #[test]
    fn analysis_schema_join_and_pipeline_work() {
        let rows = r#"[{"team":"red","score":"10"},{"team":"blue","score":"5"}]"#;
        let schema = r#"{"team":"string","score":{"type":"int","nullable":false}}"#;
        let validated =
            analysis_validate_schema(rows.as_ptr(), rows.len(), schema.as_ptr(), schema.len());
        let validated_text = unsafe {
            std::str::from_utf8(std::slice::from_raw_parts(validated.ptr, validated.len))
        }
        .unwrap();
        let validated_json: serde_json::Value = serde_json::from_str(validated_text).unwrap();
        assert_eq!(
            validated_json.get("ok").and_then(|v| v.as_bool()),
            Some(true)
        );
        assert_eq!(
            validated_json
                .get("rows")
                .and_then(|v| v.as_array())
                .and_then(|rows| rows.first())
                .and_then(|row| row.get("score"))
                .and_then(|v| v.as_i64()),
            Some(10)
        );

        let left = r#"[{"id":1,"name":"a"},{"id":2,"name":"b"}]"#;
        let right = r#"[{"id":1,"city":"x"},{"id":3,"city":"z"}]"#;
        let how = "left";
        let joined = analysis_join(
            left.as_ptr(),
            left.len(),
            right.as_ptr(),
            right.len(),
            "id".as_ptr(),
            2,
            "id".as_ptr(),
            2,
            how.as_ptr(),
            how.len(),
        );
        let joined_text =
            unsafe { std::str::from_utf8(std::slice::from_raw_parts(joined.ptr, joined.len)) }
                .unwrap();
        let joined_json: serde_json::Value = serde_json::from_str(joined_text).unwrap();
        assert_eq!(joined_json.as_array().map(|v| v.len()), Some(2));

        let pipeline_rows =
            r#"[{"team":"red","score":10},{"team":"blue","score":5},{"team":"red","score":3}]"#;
        let pipeline = r#"{"name":"team_sum","stages":[{"op":"group_agg","key":"team","field":"score","agg":"sum"}]}"#;
        let piped = analysis_pipeline_run(
            pipeline_rows.as_ptr(),
            pipeline_rows.len(),
            pipeline.as_ptr(),
            pipeline.len(),
        );
        let piped_text =
            unsafe { std::str::from_utf8(std::slice::from_raw_parts(piped.ptr, piped.len)) }
                .unwrap();
        let piped_json: serde_json::Value = serde_json::from_str(piped_text).unwrap();
        assert_eq!(
            piped_json
                .get("manifest")
                .and_then(|v| v.get("schema_version"))
                .and_then(|v| v.as_u64()),
            Some(1)
        );
        assert_eq!(
            piped_json
                .get("manifest")
                .and_then(|v| v.get("stage_count"))
                .and_then(|v| v.as_u64()),
            Some(1)
        );

        let quantiles = analysis_quantiles("[1,2,3,4,5]".as_ptr(), 11, "[0.5,0.9]".as_ptr(), 9);
        let quantiles_text = unsafe {
            std::str::from_utf8(std::slice::from_raw_parts(quantiles.ptr, quantiles.len))
        }
        .unwrap();
        let quantiles_json: serde_json::Value = serde_json::from_str(quantiles_text).unwrap();
        assert_eq!(quantiles_json.as_array().map(|v| v.len()), Some(2));

        let rolling = analysis_rolling_mean("[1,2,3,4]".as_ptr(), 9, 2);
        let rolling_text =
            unsafe { std::str::from_utf8(std::slice::from_raw_parts(rolling.ptr, rolling.len)) }
                .unwrap();
        let rolling_json: serde_json::Value = serde_json::from_str(rolling_text).unwrap();
        assert_eq!(rolling_json.as_array().map(|v| v.len()), Some(4));

        unsafe {
            enkai_free(validated.ptr, validated.len);
            enkai_free(joined.ptr, joined.len);
            enkai_free(piped.ptr, piped.len);
            enkai_free(quantiles.ptr, quantiles.len);
            enkai_free(rolling.ptr, rolling.len);
        }
    }

    #[test]
    fn algo_and_ml_metrics_work() {
        let values = "[5,1,3,2]";
        let sorted = algo_sort_ints(values.as_ptr(), values.len());
        let sorted_text =
            unsafe { std::str::from_utf8(std::slice::from_raw_parts(sorted.ptr, sorted.len)) }
                .unwrap();
        assert_eq!(sorted_text, "[1,2,3,5]");
        let idx = algo_binary_search_ints(sorted_text.as_ptr(), sorted_text.len(), 3);
        assert_eq!(idx, 2);
        let topk = algo_top_k_ints(values.as_ptr(), values.len(), 2);
        let topk_text =
            unsafe { std::str::from_utf8(std::slice::from_raw_parts(topk.ptr, topk.len)) }.unwrap();
        assert_eq!(topk_text, "[5,3]");
        let merged = algo_merge_sorted_ints("[1,3,5]".as_ptr(), 7, "[2,4,6]".as_ptr(), 7);
        let merged_text =
            unsafe { std::str::from_utf8(std::slice::from_raw_parts(merged.ptr, merged.len)) }
                .unwrap();
        assert_eq!(merged_text, "[1,2,3,4,5,6]");
        let merged_counts = algo_merge_count_maps_json(
            "{\"a\":2,\"b\":1}".as_ptr(),
            13,
            "{\"a\":3,\"c\":4}".as_ptr(),
            13,
        );
        let merged_counts_text = unsafe {
            std::str::from_utf8(std::slice::from_raw_parts(
                merged_counts.ptr,
                merged_counts.len,
            ))
        }
        .unwrap();
        let merged_counts_json: serde_json::Value =
            serde_json::from_str(merged_counts_text).unwrap();
        assert_eq!(merged_counts_json["a"].as_i64(), Some(5));
        assert_eq!(merged_counts_json["c"].as_i64(), Some(4));
        let window_mean = algo_window_mean_json("[1,2,3,4]".as_ptr(), 9, 2);
        let window_mean_text = unsafe {
            std::str::from_utf8(std::slice::from_raw_parts(window_mean.ptr, window_mean.len))
        }
        .unwrap();
        assert_eq!(window_mean_text, "[1.5,2.5,3.5]");
        let cumulative = algo_cumulative_sum_json("[1,2,3]".as_ptr(), 7);
        let cumulative_text = unsafe {
            std::str::from_utf8(std::slice::from_raw_parts(cumulative.ptr, cumulative.len))
        }
        .unwrap();
        assert_eq!(cumulative_text, "[1.0,3.0,6.0]");
        let accuracy = ml_metric_accuracy_json("[1,0,1]".as_ptr(), 7, "[1,1,1]".as_ptr(), 7);
        assert!((accuracy - (2.0 / 3.0)).abs() < 1e-6);
        let prf =
            ml_metric_precision_recall_f1_json("[1,0,1]".as_ptr(), 7, "[1,1,1]".as_ptr(), 7, 1);
        let prf_text =
            unsafe { std::str::from_utf8(std::slice::from_raw_parts(prf.ptr, prf.len)) }.unwrap();
        let prf_json: serde_json::Value = serde_json::from_str(prf_text).unwrap();
        assert!(prf_json["precision"].as_f64().unwrap() > 0.9);
        assert!(prf_json["recall"].as_f64().unwrap() > 0.6);
        let mse = ml_metric_mse_json("[1,2,3]".as_ptr(), 7, "[1,2,4]".as_ptr(), 7);
        assert!((mse - (1.0 / 3.0)).abs() < 1e-6);
        let mae = ml_metric_mae_json("[1,2,3]".as_ptr(), 7, "[1,2,4]".as_ptr(), 7);
        assert!((mae - (1.0 / 3.0)).abs() < 1e-6);
        let rmse = ml_metric_rmse_json("[1,2,3]".as_ptr(), 7, "[1,2,4]".as_ptr(), 7);
        assert!((rmse - (1.0 / 3.0_f64).sqrt()).abs() < 1e-6);
        let split = ml_split_indices_json(10, 0.2, 42, 1);
        let split_text =
            unsafe { std::str::from_utf8(std::slice::from_raw_parts(split.ptr, split.len)) }
                .unwrap();
        let split_json: serde_json::Value = serde_json::from_str(split_text).unwrap();
        assert_eq!(split_json["test_count"].as_u64(), Some(2));
        assert_eq!(split_json["train"].as_array().map(|v| v.len()), Some(8));
        let split2 = ml_split_indices_json(10, 0.2, 42, 1);
        let split2_text =
            unsafe { std::str::from_utf8(std::slice::from_raw_parts(split2.ptr, split2.len)) }
                .unwrap();
        assert_eq!(split_text, split2_text);
        let topk_sum = algo_top_k_sum_ints_repeat("[9,1,7,3,8]".as_ptr(), 11, 3, 4);
        assert_eq!(topk_sum, 96);
        let split_count_sum = ml_split_test_count_repeat(12, 0.25, 99, 1, 10);
        assert_eq!(split_count_sum, 30);
        let lr0 = ml_scheduler_linear_warmup(0, 100, 10, 0.001, 0.0001);
        let lr20 = ml_scheduler_linear_warmup(20, 100, 10, 0.001, 0.0001);
        assert!(lr0 > 0.0 && lr0 < 0.001);
        assert!(lr20 < 0.001 && lr20 > 0.0001);
        unsafe {
            enkai_free(sorted.ptr, sorted.len);
            enkai_free(topk.ptr, topk.len);
            enkai_free(merged.ptr, merged.len);
            enkai_free(merged_counts.ptr, merged_counts.len);
            enkai_free(window_mean.ptr, window_mean.len);
            enkai_free(cumulative.ptr, cumulative.len);
            enkai_free(prf.ptr, prf.len);
            enkai_free(split.ptr, split.len);
            enkai_free(split2.ptr, split2.len);
        }
    }
}
