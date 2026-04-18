mod bootstrap;

fn main() {
    let layout = bootstrap::resolve(false);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = bootstrap::build_command(&layout, "writansub.cli", &args);
    let code = bootstrap::run_with_tee(cmd, &layout.log_path, false, false);
    std::process::exit(code);
}
