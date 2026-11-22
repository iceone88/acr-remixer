//! acr-stagegen GUI (egui). Phase 1: drives the Python pipeline
//! (`python -m stagegen.cli build ...`) and previews the result; the pipeline is
//! being ported to Rust module by module.
#![windows_subsystem = "windows"]

mod steam;

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{channel, Receiver, Sender};

use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints, VLine};
use serde::{Deserialize, Serialize};

const STAGES: &[(&str, &str)] = &[
    ("WalesS4HafrenSouth", "Wales – Hafren South"),
    ("WalesS3HafrenNorth", "Wales – Hafren North"),
    ("SwedenS1Vargasen", "Sweden – Vargåsen"),
    ("SwedenS3Fredriksberg", "Sweden – Fredriksberg"),
    ("MonteCarloS1Bollene", "Monte Carlo – Bollène"),
    ("MonteCarloS2Sisteron", "Monte Carlo – Sisteron"),
    ("GreeceS3Elatia", "Greece – Elatia"),
    ("GreeceS4Loutraki", "Greece – Loutraki"),
    ("AlsaceS2Munster", "Alsace – Munster"),
    ("AlsaceS4Saverne", "Alsace – Saverne"),
];
/// Stages verified in-game with the hands-off test; the others are experimental.
const VERIFIED: &[&str] = &["WalesS4HafrenSouth"];

#[derive(Serialize, Deserialize, Clone)]
#[serde(default)]
struct Prefs {
    acr_dir: String,
    python: String,
    backend_dir: String,
    out_dir: String,
    stage: String,
    length_m: f32,
    seed: u32,
    batch: u32,
    forest: bool,
    grass: bool,
    foliage: bool,
    install: bool,
    launch: bool,
}

impl Default for Prefs {
    fn default() -> Self {
        let acr = steam::detect_acr().map(|p| p.display().to_string()).unwrap_or_default();
        let py = std::env::var("LOCALAPPDATA")
            .map(|l| PathBuf::from(l).join(r"Programs\Python\Python314\python.exe"))
            .ok()
            .filter(|p| p.is_file())
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "python".into());
        let backend = if acr.is_empty() { String::new() } else { format!(r"{acr}\acr-stagegen") };
        let out = if backend.is_empty() { String::new() } else { format!(r"{backend}\out") };
        Prefs {
            acr_dir: acr,
            python: py,
            backend_dir: backend,
            out_dir: out,
            stage: STAGES[0].0.into(),
            length_m: 4500.0,
            seed: 1,
            batch: 1,
            forest: true,
            grass: true,
            foliage: true,
            install: true,
            launch: false,
        }
    }
}

/// The finished route of a built stage (from layout.json).
struct Preview {
    name: String,
    pts: Vec<[f64; 3]>,
    s: Vec<f64>,
    joins: Vec<f64>,
    length: f64,
    notes: usize,
}

fn load_preview(dir: &Path) -> Result<Preview, String> {
    let txt = std::fs::read_to_string(dir.join("layout.json")).map_err(|e| e.to_string())?;
    let v: serde_json::Value = serde_json::from_str(&txt).map_err(|e| e.to_string())?;
    let pts: Vec<[f64; 3]> = v["preview"]
        .as_array()
        .ok_or("layout.json has no preview (rebuild with a current backend)")?
        .iter()
        .filter_map(|p| {
            let a = p.as_array()?;
            Some([a.first()?.as_f64()?, a.get(1)?.as_f64()?, a.get(2)?.as_f64()?])
        })
        .collect();
    let mut s = vec![0.0];
    for w in pts.windows(2) {
        let d = ((w[1][0] - w[0][0]).powi(2) + (w[1][1] - w[0][1]).powi(2)).sqrt();
        s.push(s.last().unwrap() + d);
    }
    let joins = v["pieces"]
        .as_array()
        .map(|a| a.iter().skip(1).filter_map(|p| p["s_start"].as_f64()).collect())
        .unwrap_or_default();
    Ok(Preview {
        name: dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
        length: v["length_m"].as_f64().unwrap_or(*s.last().unwrap()),
        notes: v["pacenotes"].as_array().map(|a| a.len()).unwrap_or(0),
        pts,
        s,
        joins,
    })
}

enum Msg {
    Line(String),
    Done { ok: bool, out: PathBuf },
}

struct Job {
    rx: Receiver<Msg>,
    remaining: Vec<u32>,
}

struct App {
    prefs: Prefs,
    log: String,
    job: Option<Job>,
    preview: Option<Preview>,
    tab: Tab,
}

#[derive(PartialEq, Clone, Copy)]
enum Tab {
    Map,
    Elevation,
    Log,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let prefs: Prefs = cc.storage.and_then(|s| eframe::get_value(s, eframe::APP_KEY)).unwrap_or_default();
        App { prefs, log: String::new(), job: None, preview: None, tab: Tab::Map }
    }

    fn out_for(&self, seed: u32) -> PathBuf {
        PathBuf::from(&self.prefs.out_dir).join(format!("{}_seed{}", self.prefs.stage, seed))
    }

    fn start(&mut self) {
        let seeds: Vec<u32> = (0..self.prefs.batch.max(1)).map(|i| self.prefs.seed + i).collect();
        self.log.clear();
        self.tab = Tab::Log;
        let (_, rx) = channel();
        self.job = Some(Job { rx, remaining: seeds });
        self.next();
    }

    /// Launch the build for the next seed of the batch (install/launch only on the last).
    fn next(&mut self) {
        let Some(job) = self.job.as_mut() else { return };
        let Some(seed) = (if job.remaining.is_empty() { None } else { Some(job.remaining.remove(0)) }) else {
            self.job = None;
            return;
        };
        let last = job.remaining.is_empty();
        let p = self.prefs.clone();
        let mut args: Vec<String> = vec![
            "-m".into(), "stagegen.cli".into(), "build".into(),
            "--level".into(), p.stage.clone(),
            "--length".into(), format!("{:.0}", p.length_m),
            "--seed".into(), seed.to_string(),
            "--out".into(), p.out_dir.clone(),
        ];
        if !p.forest { args.push("--no-forest".into()); }
        if !p.grass { args.push("--no-grass".into()); }
        if !p.foliage { args.push("--no-foliage".into()); }
        if last && p.install { args.push("--install".into()); }
        if last && p.install && p.launch { args.push("--launch".into()); }
        let paks = PathBuf::from(&p.acr_dir).join(r"acr\Content\Paks");
        args.push("--paks".into());
        args.push(paks.display().to_string());
        let out = PathBuf::from(&p.out_dir).join(format!("{}_seed{}", p.stage, seed));
        let (tx, rx) = channel();
        job.rx = rx;
        self.log.push_str(&format!("\n=== {} seed {} ===\n", p.stage, seed));
        spawn_build(p.python, p.backend_dir, args, out, tx);
    }

    fn poll(&mut self) {
        let mut finished = None;
        if let Some(job) = &self.job {
            while let Ok(m) = job.rx.try_recv() {
                match m {
                    Msg::Line(l) => {
                        self.log.push_str(&l);
                        self.log.push('\n');
                    }
                    Msg::Done { ok, out } => finished = Some((ok, out)),
                }
            }
        }
        if let Some((ok, out)) = finished {
            if ok {
                match load_preview(&out) {
                    Ok(pv) => {
                        self.preview = Some(pv);
                        self.tab = Tab::Map;
                    }
                    Err(e) => self.log.push_str(&format!("preview: {e}\n")),
                }
                self.next();
            } else {
                self.log.push_str("build failed (see above)\n");
                self.job = None;
            }
        }
    }
}

fn spawn_build(python: String, cwd: String, args: Vec<String>, out: PathBuf, tx: Sender<Msg>) {
    std::thread::spawn(move || {
        use std::os::windows::process::CommandExt;
        let child = Command::new(&python)
            .args(&args)
            .current_dir(&cwd)
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTHONIOENCODING", "utf-8")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .creation_flags(0x0800_0000) // CREATE_NO_WINDOW
            .spawn();
        let mut child = match child {
            Ok(c) => c,
            Err(e) => {
                let _ = tx.send(Msg::Line(format!("cannot start {python}: {e}")));
                let _ = tx.send(Msg::Done { ok: false, out });
                return;
            }
        };
        let err = child.stderr.take().unwrap();
        let tx2 = tx.clone();
        let t = std::thread::spawn(move || {
            for l in BufReader::new(err).lines().map_while(Result::ok) {
                let _ = tx2.send(Msg::Line(l));
            }
        });
        for l in BufReader::new(child.stdout.take().unwrap()).lines().map_while(Result::ok) {
            let _ = tx.send(Msg::Line(l));
        }
        let _ = t.join();
        let ok = child.wait().map(|s| s.success()).unwrap_or(false);
        let _ = tx.send(Msg::Done { ok, out });
    });
}

fn path_row(ui: &mut egui::Ui, label: &str, value: &mut String, ok: bool) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(egui::TextEdit::singleline(value).desired_width(230.0));
        ui.label(if ok { "✔" } else { "✖" });
    });
}

impl eframe::App for App {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, &self.prefs);
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll();
        if self.job.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
        let busy = self.job.is_some();

        egui::SidePanel::left("opts").min_width(330.0).show(ctx, |ui| {
            ui.heading("acr-stagegen");
            ui.add_space(6.0);
            egui::CollapsingHeader::new("Setup").default_open(self.prefs.acr_dir.is_empty()).show(ui, |ui| {
                let p = &mut self.prefs;
                let acr_ok = steam::is_acr_dir(Path::new(&p.acr_dir));
                path_row(ui, "ACR", &mut p.acr_dir, acr_ok);
                if ui.small_button("detect").clicked() {
                    if let Some(d) = steam::detect_acr() {
                        p.acr_dir = d.display().to_string();
                    }
                }
                let be_ok = Path::new(&p.backend_dir).join("stagegen").is_dir();
                path_row(ui, "Backend", &mut p.backend_dir, be_ok);
                path_row(ui, "Python", &mut p.python, true);
                path_row(ui, "Output", &mut p.out_dir, true);
            });
            ui.separator();
            ui.add_enabled_ui(!busy, |ui| {
                let p = &mut self.prefs;
                let cur = STAGES.iter().find(|s| s.0 == p.stage).map(|s| s.1).unwrap_or("?");
                egui::ComboBox::from_label("Stage").width(220.0).selected_text(cur).show_ui(ui, |ui| {
                    for (id, name) in STAGES {
                        let tag = if VERIFIED.contains(id) { "" } else { "  (experimental)" };
                        ui.selectable_value(&mut p.stage, id.to_string(), format!("{name}{tag}"));
                    }
                });
                ui.add(egui::Slider::new(&mut p.length_m, 1000.0..=15000.0).step_by(100.0).suffix(" m").text("Length"))
                    .on_hover_text("Rounded to the closest length the source stage allows");
                ui.horizontal(|ui| {
                    ui.label("Seed");
                    ui.add(egui::DragValue::new(&mut p.seed).range(1..=999_999));
                    if ui.small_button("🎲").on_hover_text("random seed").clicked() {
                        p.seed = (std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.subsec_nanos())
                            .unwrap_or(1)
                            % 999_999)
                            + 1;
                    }
                    ui.label("Batch");
                    ui.add(egui::DragValue::new(&mut p.batch).range(1..=50))
                        .on_hover_text("Build this many stages (seed, seed+1, ...)");
                });
                ui.label("Foliage (untick to leave out for performance):");
                ui.horizontal(|ui| {
                    ui.checkbox(&mut p.forest, "Forest");
                    ui.checkbox(&mut p.grass, "Grass & rocks");
                    ui.checkbox(&mut p.foliage, "Ground grass");
                });
                ui.checkbox(&mut p.install, "Install into the game (closes ACR)");
                ui.add_enabled(p.install, egui::Checkbox::new(&mut p.launch, "Start ACR afterwards"));
            });
            ui.add_space(8.0);
            let ready = steam::is_acr_dir(Path::new(&self.prefs.acr_dir))
                && Path::new(&self.prefs.backend_dir).join("stagegen").is_dir();
            ui.horizontal(|ui| {
                if busy {
                    ui.spinner();
                    ui.label("building ...");
                } else if ui.add_enabled(ready, egui::Button::new("▶  Generate").min_size([140.0, 30.0].into())).clicked() {
                    self.start();
                }
                if !busy && ui.button("Open last").clicked() {
                    match load_preview(&self.out_for(self.prefs.seed)) {
                        Ok(pv) => self.preview = Some(pv),
                        Err(e) => self.log.push_str(&format!("preview: {e}\n")),
                    }
                }
            });
            if !VERIFIED.contains(&self.prefs.stage.as_str()) {
                ui.colored_label(egui::Color32::from_rgb(230, 170, 60), "Experimental stage: not yet verified in-game.");
            }
            if let Some(pv) = &self.preview {
                ui.separator();
                ui.label(format!("{}: {:.2} km, {} pieces, {} pacenotes", pv.name, pv.length / 1000.0,
                                 pv.joins.len() + 1, pv.notes));
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.tab, Tab::Map, "Map");
                ui.selectable_value(&mut self.tab, Tab::Elevation, "Elevation");
                ui.selectable_value(&mut self.tab, Tab::Log, "Log");
            });
            ui.separator();
            match self.tab {
                Tab::Log => {
                    egui::ScrollArea::vertical().stick_to_bottom(true).auto_shrink([false; 2]).show(ui, |ui| {
                        ui.add(egui::Label::new(egui::RichText::new(&self.log).monospace()).wrap());
                    });
                }
                Tab::Map | Tab::Elevation if self.preview.is_none() => {
                    ui.centered_and_justified(|ui| ui.label("Generate a stage (or open the last one) to see it here."));
                }
                Tab::Map => {
                    let pv = self.preview.as_ref().unwrap();
                    Plot::new("map").data_aspect(1.0).legend(Legend::default()).show(ui, |pu| {
                        let mut bounds = vec![0.0];
                        bounds.extend(pv.joins.iter().copied());
                        bounds.push(f64::INFINITY);
                        for (k, w) in bounds.windows(2).enumerate() {
                            let pts: Vec<[f64; 2]> = pv.pts.iter().zip(&pv.s)
                                .filter(|(_, s)| **s >= w[0] - 10.0 && **s <= w[1] + 10.0)
                                .map(|(p, _)| [p[0], p[1]]).collect();
                            pu.line(Line::new(PlotPoints::from(pts)).width(3.0_f32).name(format!("piece {k}")));
                        }
                        if let (Some(a), Some(b)) = (pv.pts.first(), pv.pts.last()) {
                            pu.points(egui_plot::Points::new(vec![[a[0], a[1]]]).radius(6.0_f32).name("start"));
                            pu.points(egui_plot::Points::new(vec![[b[0], b[1]]]).radius(6.0_f32).shape(egui_plot::MarkerShape::Square).name("finish"));
                        }
                    });
                }
                Tab::Elevation => {
                    let pv = self.preview.as_ref().unwrap();
                    Plot::new("elev").show(ui, |pu| {
                        let pts: Vec<[f64; 2]> = pv.pts.iter().zip(&pv.s).map(|(p, s)| [*s, p[2]]).collect();
                        pu.line(Line::new(PlotPoints::from(pts)).width(2.0_f32).name("elevation (m)"));
                        for j in &pv.joins {
                            pu.vline(VLine::new(*j).name("join"));
                        }
                    });
                }
            }
        });
    }
}

fn main() -> eframe::Result<()> {
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1100.0, 700.0]).with_title("acr-stagegen"),
        ..Default::default()
    };
    eframe::run_native("acr-stagegen", opts, Box::new(|cc| Ok(Box::new(App::new(cc)))))
}
