use memmap2::Mmap;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
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
}
