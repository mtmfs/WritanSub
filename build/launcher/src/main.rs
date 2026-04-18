#![cfg_attr(windows, windows_subsystem = "windows")]

mod bootstrap;

fn main() {
    let layout = bootstrap::resolve(true);
    let cmd = bootstrap::build_command(&layout, "writansub", &[]);
    let code = bootstrap::run_with_tee(cmd, &layout.log_path, true, true);
    std::process::exit(code);
}
