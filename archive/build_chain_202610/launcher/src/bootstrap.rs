use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::os::windows::ffi::OsStrExt;
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::SystemTime;

use windows_sys::Win32::UI::WindowsAndMessaging::{MessageBoxW, MB_ICONERROR, MB_OK};

pub const CREATE_NO_WINDOW: u32 = 0x0800_0000;
const LOG_ROLLOVER_BYTES: u64 = 1_048_576;
const TAIL_BUF_BYTES: usize = 8192;
const TAIL_SHOW_BYTES: usize = 4000;

pub struct Layout {
    pub root: PathBuf,
    pub python: PathBuf,
    pub app: PathBuf,
    pub log_path: PathBuf,
}

pub fn resolve(use_w: bool) -> Layout {
    let exe = std::env::current_exe().expect("current_exe");
    let root = exe.parent().expect("exe has no parent").to_path_buf();
    let python_name = if use_w { "pythonw.exe" } else { "python.exe" };
    let python = root.join("runtime").join(python_name);
    let app = root.join("app");

    let log_dir = resolve_log_dir();
    let _ = std::fs::create_dir_all(&log_dir);
    let log_path = log_dir.join("launcher.log");
    rollover_if_large(&log_path);

    Layout { root, python, app, log_path }
}

fn resolve_log_dir() -> PathBuf {
    if let Ok(appdata) = std::env::var("APPDATA") {
        return PathBuf::from(appdata).join("WritanSub");
    }
    if let Ok(userprofile) = std::env::var("USERPROFILE") {
        return PathBuf::from(userprofile)
            .join("AppData")
            .join("Roaming")
            .join("WritanSub");
    }
    PathBuf::from(".")
}

fn rollover_if_large(log_path: &Path) {
    if let Ok(meta) = std::fs::metadata(log_path) {
        if meta.len() > LOG_ROLLOVER_BYTES {
            let _ = std::fs::write(log_path, b"");
        }
    }
}

pub fn build_command(layout: &Layout, module: &str, args: &[String]) -> Command {
    let mut cmd = Command::new(&layout.python);
    cmd.current_dir(&layout.app)
        .arg("-m")
        .arg(module)
        .args(args)
        .env("WRITANSUB_HOME", &layout.root)
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env("PYTHONPATH", &layout.app);
    cmd
}

pub fn run_with_tee(
    mut cmd: Command,
    log_path: &Path,
    no_window: bool,
    show_box_on_error: bool,
) -> i32 {
    if no_window {
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd.stderr(Stdio::piped());

    append_log(log_path, &format!("--- launch @ {} ---\n", unix_time()));

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            let body = format!(
                "Failed to spawn python: {e}\nExecutable: {}",
                cmd.get_program().to_string_lossy()
            );
            append_log(log_path, &format!("{body}\n"));
            if show_box_on_error {
                show_error_box("WritanSub 启动失败", &body);
            }
            return 1;
        }
    };

    let stderr = child.stderr.take().expect("piped stderr");
    let log_path_clone = log_path.to_path_buf();
    let (tx, rx) = mpsc::channel::<Vec<u8>>();
    let tee = thread::spawn(move || {
        let mut reader = stderr;
        let mut buf = [0u8; 4096];
        let mut tail: VecDeque<u8> = VecDeque::with_capacity(TAIL_BUF_BYTES);
        let mut sink = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path_clone)
            .ok();
        loop {
            match reader.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    if let Some(ref mut f) = sink {
                        let _ = f.write_all(&buf[..n]);
                    }
                    for &b in &buf[..n] {
                        if tail.len() == TAIL_BUF_BYTES {
                            tail.pop_front();
                        }
                        tail.push_back(b);
                    }
                }
            }
        }
        let _ = tx.send(tail.into_iter().collect());
    });

    let status = child.wait();
    let _ = tee.join();
    let tail_bytes = rx.recv().unwrap_or_default();

    let code = match status {
        Ok(s) => s.code().unwrap_or(-1),
        Err(e) => {
            append_log(log_path, &format!("child.wait failed: {e}\n"));
            -1
        }
    };

    if code != 0 && show_box_on_error {
        let tail_str = String::from_utf8_lossy(&tail_bytes);
        let shown = trim_tail(&tail_str, TAIL_SHOW_BYTES);
        let body = format!(
            "Python 进程以代码 {code} 退出。\n完整日志: {}\n\n--- 最近 stderr ---\n{shown}",
            log_path.display()
        );
        show_error_box("WritanSub 启动失败", &body);
    }

    code
}

fn trim_tail(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let start = s.len() - max_bytes;
    let boundary = (start..s.len())
        .find(|&i| s.is_char_boundary(i))
        .unwrap_or(start);
    format!("... (前略) ...\n{}", &s[boundary..])
}

fn append_log(log_path: &Path, msg: &str) {
    if let Ok(mut f) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
    {
        let _ = f.write_all(msg.as_bytes());
    }
}

fn unix_time() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub fn show_error_box(title: &str, body: &str) {
    let title_w: Vec<u16> = std::ffi::OsStr::new(title)
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let body_w: Vec<u16> = std::ffi::OsStr::new(body)
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    unsafe {
        MessageBoxW(
            std::ptr::null_mut(),
            body_w.as_ptr(),
            title_w.as_ptr(),
            MB_OK | MB_ICONERROR,
        );
    }
}
