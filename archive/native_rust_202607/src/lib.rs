use pyo3::prelude::*;
use std::collections::HashMap;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use parking_lot::RwLock;
use once_cell::sync::Lazy;

/// 全局资源单例
static REGISTRY: Lazy<RwLock<NativeRegistry>> = Lazy::new(|| RwLock::new(NativeRegistry::default()));
static NEXT_HANDLE: AtomicU32 = AtomicU32::new(1);

#[derive(Default)]
struct NativeRegistry {
    processes: HashMap<u32, Child>,
    models: HashMap<u32, PyObject>,
    in_use: HashMap<u32, bool>,
}

#[pyclass]
pub struct ResourceRegistry;

#[pymethods]
impl ResourceRegistry {
    #[staticmethod]
    fn register_model(obj: PyObject) -> u32 {
        let handle = NEXT_HANDLE.fetch_add(1, Ordering::SeqCst);
        let mut reg = REGISTRY.write();
        reg.models.insert(handle, obj);
        reg.in_use.insert(handle, false);
        handle
    }

    #[staticmethod]
    fn acquire_model(handle: u32) -> bool {
        let mut reg = REGISTRY.write();
        if let Some(in_use) = reg.in_use.get_mut(&handle) {
            if !*in_use {
                *in_use = true;
                return true;
            }
        }
        false
    }

    #[staticmethod]
    fn release_model(handle: u32) {
        let mut reg = REGISTRY.write();
        if let Some(in_use) = reg.in_use.get_mut(&handle) {
            *in_use = false;
        }
    }

    #[staticmethod]
    fn get_model(handle: u32) -> PyResult<PyObject> {
        let reg = REGISTRY.read();
        reg.models.get(&handle)
            .cloned()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Handle not found"))
    }

    #[staticmethod]
    fn unload_model(handle: u32) {
        let mut reg = REGISTRY.write();
        reg.models.remove(&handle);
        reg.in_use.remove(&handle);
    }

    #[staticmethod]
    fn spawn_process(cmd: Vec<String>) -> PyResult<u32> {
        if cmd.is_empty() { return Ok(0); }
        let handle = NEXT_HANDLE.fetch_add(1, Ordering::SeqCst);
        let child = Command::new(&cmd[0])
            .args(&cmd[1..])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to spawn process: {}", e)))?;
        
        let mut reg = REGISTRY.write();
        reg.processes.insert(handle, child);
        Ok(handle)
    }

    #[staticmethod]
    fn wait_process(handle: u32) -> PyResult<(i32, Vec<u8>, Vec<u8>)> {
        let mut reg = REGISTRY.write();
        if let Some(mut child) = reg.processes.remove(&handle) {
            let output = child.wait_with_output()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            Ok((output.status.code().unwrap_or(-1), output.stdout, output.stderr))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>("Process handle not found"))
        }
    }

    #[staticmethod]
    fn shutdown() {
        let mut reg = REGISTRY.write();
        // 强行清理所有子进程
        for (_, mut child) in reg.processes.drain() {
            let _ = child.kill();
        }
        // 清理所有模型引用
        reg.models.clear();
        reg.in_use.clear();
    }
}

#[pymodule]
fn writansub_native(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ResourceRegistry>()?;
    Ok(())
}
