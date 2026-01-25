use memmap2::Mmap;
use sha2::{Digest, Sha256};

#[repr(C)]
pub struct FfiSlice {
    pub ptr: *mut u8,
    pub len: usize,
}

#[no_mangle]
/// # Safety
/// The caller must pass a pointer and length originally allocated by `morph_native`
/// and must ensure the buffer is not freed more than once.
pub unsafe extern "C" fn morph_free(ptr: *mut u8, len: usize) {
    if ptr.is_null() {
        return;
    }
    let _ = Vec::from_raw_parts(ptr, len, len);
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

fn make_slice(mut bytes: Vec<u8>) -> FfiSlice {
    let len = bytes.len();
    let ptr = bytes.as_mut_ptr();
    std::mem::forget(bytes);
    FfiSlice { ptr, len }
}

fn null_slice() -> FfiSlice {
    FfiSlice {
        ptr: std::ptr::null_mut(),
        len: 0,
    }
}

#[no_mangle]
pub extern "C" fn add_i64(a: i64, b: i64) -> i64 {
    a + b
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
