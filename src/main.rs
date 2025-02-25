#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

use anyhow::Result;
use eframe::egui;

mod endpoint_type;
mod llmclient;
mod chatapp;
mod chatapp_ui;
mod sdclient;

use chatapp::ChatApp;

#[tokio::main]
async fn main() -> Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "LLM Chat",
        options,
        Box::new(|cc| Box::new(ChatApp::new(cc))),
    )
    .map_err(|e| anyhow::anyhow!("Failed to run app: {}", e))
}
