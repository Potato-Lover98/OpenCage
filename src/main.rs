mod ai;
mod app;
mod core;
mod ui;

use anyhow::Result;
use app::App;
use ai::rag::RagStore;
use core::config::SettingsStore;
use core::shell_install::ensure_shell_launch_command;

fn main() -> Result<()> {
    if let Err(e) = ensure_shell_launch_command() {
        eprintln!("Opencage shell install warning: {e}");
    }
    let settings = SettingsStore::load_or_create()?;
    let rag = RagStore::open_default()?;
    let args: Vec<String> = std::env::args().collect();
    let resume_session = if args.len() == 3 && args[1] == "resume" {
        Some(args[2].clone())
    } else {
        None
    };
    let mut app = App::new(settings, rag, resume_session);
    app.run()
}
