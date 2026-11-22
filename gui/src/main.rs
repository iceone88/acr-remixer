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

#[cfg(embedded_backend)]
const BACKEND: Option<&[u8]> = Some(include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../packaging/dist/stagegen-backend.exe")));
#[cfg(not(embedded_backend))]
const BACKEND: Option<&[u8]> = None;

/// Unpack the embedded backend once (the name carries a content hash, so a
/// new release replaces it) and return its path.
fn embedded_backend() -> Result<PathBuf, String> {
    let bytes = BACKEND.ok_or("this build has no embedded backend")?;
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes.iter().step_by(64) {
        h = (h ^ *b as u64).wrapping_mul(0x100_0000_01b3);
    }
    let dir = PathBuf::from(std::env::var("LOCALAPPDATA").map_err(|e| e.to_string())?).join("acr-stagegen");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let exe = dir.join(format!("stagegen-backend-{:x}-{:016x}.exe", bytes.len(), h));
    if !exe.is_file() {
        let tmp = exe.with_extension("tmp");
        std::fs::write(&tmp, bytes).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, &exe).map_err(|e| e.to_string())?;
    }
    Ok(exe)
}

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
    /// development: run `python -m stagegen.cli` from backend_dir instead of the embedded backend
    use_python: bool,
    /// block ACR's network access (Windows Firewall) while a generated stage is installed
    block_online: bool,
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
            use_python: BACKEND.is_none(),
            block_online: false,
        }
    }
}

/// The generated stage currently installed in the game's ~mods folder (level name).
fn installed_stage(acr_dir: &str) -> Option<String> {
    let mods = PathBuf::from(acr_dir).join(r"acr\Content\Paks\~mods");
    std::fs::read_dir(mods).ok()?.filter_map(|e| e.ok()).find_map(|e| {
        let n = e.file_name().to_string_lossy().into_owned();
        n.strip_prefix("pakchunk99-")?.strip_suffix("_Stagegen_9_P.utoc").map(|s| s.to_string())
    })
}

const FW_RULE: &str = "acr-stagegen: block ACR online";

fn quiet(cmd: &mut Command) -> &mut Command {
    use std::os::windows::process::CommandExt;
    cmd.creation_flags(0x0800_0000) // CREATE_NO_WINDOW
}

/// Is the firewall rule that blocks ACR's network access present?
fn fw_blocked() -> bool {
    quiet(Command::new("netsh").args(["advfirewall", "firewall", "show", "rule", &format!("name={FW_RULE}")]))
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Add (in + out) or remove the blocking rule for acr.exe; one UAC prompt.
fn fw_set(acr_dir: &str, block: bool) -> Result<(), String> {
    let exe = PathBuf::from(acr_dir).join(r"acr\Binaries\Win64\acr.exe");
    let mut cmds = format!("netsh advfirewall firewall delete rule name=\"{FW_RULE}\"");
    if block {
        if !exe.is_file() {
            return Err(format!("{} not found", exe.display()));
        }
        for dir in ["out", "in"] {
            cmds.push_str(&format!(
                " & netsh advfirewall firewall add rule name=\"{FW_RULE}\" dir={dir} action=block program=\"{}\" enable=yes",
                exe.display()
            ));
        }
    }
    let ps = format!(
        "Start-Process cmd -Verb RunAs -Wait -WindowStyle Hidden -ArgumentList '/c {}'",
        cmds.replace('\'', "''")
    );
    let st = quiet(Command::new("powershell").args(["-NoProfile", "-Command", &ps]))
        .status()
        .map_err(|e| e.to_string())?;
    if !st.success() {
        return Err("cancelled or no admin rights".into());
    }
    if fw_blocked() != block {
        return Err("the firewall rule did not change (admin prompt declined?)".into());
    }
    Ok(())
}

const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), " (", env!("GIT_HASH"), ")");

/// Hide the Windows user name in paths before text leaves the PC.
fn redact(s: &str) -> String {
    let mut out = s.to_string();
    if let Ok(home) = std::env::var("USERPROFILE") {
        out = out.replace(&home, "%USERPROFILE%");
    }
    if let Ok(user) = std::env::var("USERNAME") {
        if user.len() >= 3 {
            out = out.replace(&user, "<user>");
        }
    }
    out
}

/// Short code for stage + length + seed + foliage: "WalesS4HafrenSouth/4500/939252/FGL".
fn share_code(p: &Prefs) -> String {
    let fol: String = [(p.forest, 'F'), (p.grass, 'G'), (p.foliage, 'L')]
        .iter().map(|(on, c)| if *on { *c } else { '-' }).collect();
    format!("{}/{:.0}/{}/{}", p.stage, p.length_m, p.seed, fol)
}

fn apply_share_code(p: &mut Prefs, code: &str) -> Result<(), String> {
    let parts: Vec<&str> = code.trim().split('/').collect();
    if parts.len() != 4 {
        return Err("expected Stage/Length/Seed/Foliage, e.g. WalesS4HafrenSouth/4500/1/FGL".into());
    }
    if !STAGES.iter().any(|s| s.0 == parts[0]) {
        return Err(format!("unknown stage {:?}", parts[0]));
    }
    let len: f32 = parts[1].parse().map_err(|_| "bad length")?;
    let seed: u32 = parts[2].parse().map_err(|_| "bad seed")?;
    p.stage = parts[0].to_string();
    p.length_m = len.clamp(1000.0, 15000.0);
    p.seed = seed.max(1);
    p.forest = parts[3].contains('F');
    p.grass = parts[3].contains('G');
    p.foliage = parts[3].contains('L');
    Ok(())
}

/// Summary of the newest ACR crash reports (error + top of the call stack).
fn acr_crashes(n: usize) -> String {
    let Ok(base) = std::env::var("LOCALAPPDATA") else { return "(no LOCALAPPDATA)\n".into() };
    let dir = PathBuf::from(base).join(r"acr\Saved\Crashes");
    let Ok(rd) = std::fs::read_dir(&dir) else { return "(no crash reports)\n".into() };
    let mut ds: Vec<_> = rd.filter_map(|e| e.ok()).filter(|e| e.path().is_dir())
        .filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path()))).collect();
    ds.sort_by(|a, b| b.0.cmp(&a.0));
    let mut out = String::new();
    for (t, d) in ds.into_iter().take(n) {
        let age = t.elapsed().map(|e| e.as_secs() / 60).unwrap_or(0);
        let txt = std::fs::read_to_string(d.join("CrashContext.runtime-xml")).unwrap_or_default();
        let tag = |name: &str| -> String {
            let (a, b) = (format!("<{name}>"), format!("</{name}>"));
            txt.find(&a).and_then(|i| txt[i + a.len()..].find(&b).map(|j| txt[i + a.len()..i + a.len() + j].trim().to_string()))
                .unwrap_or_default()
        };
        let stack: Vec<String> = tag("PCallStack").lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).take(15).collect();
        out.push_str(&format!("--- crash {} min ago: {}\n{}\n", age, tag("ErrorMessage"), stack.join("\n")));
    }
    if out.is_empty() { "(no crash reports)\n".into() } else { out }
}

fn windows_version() -> String {
    quiet(Command::new("cmd").args(["/c", "ver"])).output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "?".into())
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
    Done { ok: bool, out: Option<PathBuf> },
}

struct Job {
    rx: Receiver<Msg>,
    remaining: Vec<u32>,
    total: usize,
    progress: f32,
    phase: String,
}

struct App {
    prefs: Prefs,
    log: String,
    job: Option<Job>,
    preview: Option<Preview>,
    tab: Tab,
    /// generated stage in ~mods, firewall block present (refreshed on events)
    installed: Option<String>,
    blocked: bool,
    /// share-code text field (paste a code here to load it)
    code_input: String,
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
        let mut app = App { prefs, log: String::new(), job: None, preview: None, tab: Tab::Map,
                            installed: None, blocked: false, code_input: String::new() };
        app.preview = load_preview(&app.out_for(app.prefs.seed)).ok();     // reopen the last stage
        app.refresh_install_state();
        app
    }

    fn out_for(&self, seed: u32) -> PathBuf {
        PathBuf::from(&self.prefs.out_dir).join(format!("{}_seed{}", self.prefs.stage, seed))
    }

    fn start(&mut self) {
        let seeds: Vec<u32> = (0..self.prefs.batch.max(1)).map(|i| self.prefs.seed + i).collect();
        self.log.clear();
        self.tab = Tab::Log;
        let (_, rx) = channel();
        let total = seeds.len();
        self.job = Some(Job { rx, remaining: seeds, total, progress: 0.0, phase: "starting".into() });
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
            "build".into(),
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
        job.progress = 0.0;
        job.phase = "starting".into();
        self.log.push_str(&format!("\n=== {} seed {} ===\n", p.stage, seed));
        match backend_cmd(&p) {
            Ok((prog, mut full, cwd)) => {
                full.extend(args);
                spawn_build(prog, cwd, full, Some(out), tx);
            }
            Err(e) => {
                self.log.push_str(&format!("backend: {e}\n"));
                self.job = None;
            }
        }
    }

    fn refresh_install_state(&mut self) {
        self.installed = installed_stage(&self.prefs.acr_dir);
        self.blocked = fw_blocked();
    }

    /// Make the firewall block match: on while a generated stage is installed
    /// (if wanted), off otherwise. Asks for admin rights only when it changes.
    fn sync_firewall(&mut self) {
        self.refresh_install_state();
        let want = self.prefs.block_online && self.installed.is_some();
        if want != self.blocked {
            match fw_set(&self.prefs.acr_dir, want) {
                Ok(()) => self.log.push_str(if want { "ACR's internet access is blocked while the generated stage is installed.\n" }
                                            else { "ACR's internet access is allowed again.\n" }),
                Err(e) => self.log.push_str(&format!("firewall: {e}\n")),
            }
            self.blocked = fw_blocked();
        }
    }

    /// Write one text file with everything useful for a bug report; returns its path.
    fn export_bug_report(&self) -> Result<PathBuf, String> {
        let p = &self.prefs;
        let mut r = String::new();
        r.push_str(&format!("acr-stagegen bug report\nversion: {VERSION}\nbackend: {}\nwindows: {}\n\n",
            if p.use_python || BACKEND.is_none() { "python (development)" } else { "embedded" }, windows_version()));
        r.push_str(&format!("share code: {}\n", share_code(p)));
        r.push_str(&format!("installed stage: {}\nonline blocked: {}\n\n",
            self.installed.as_deref().unwrap_or("none"), self.blocked));
        r.push_str("=== settings ===\n");
        r.push_str(&serde_json::to_string_pretty(p).unwrap_or_default());
        r.push_str("\n\n=== last stage ===\n");
        match &self.preview {
            Some(pv) => r.push_str(&format!("{}: {:.2} km, {} pieces (joins at {:?} m), {} pacenotes\n",
                pv.name, pv.length / 1000.0, pv.joins.len() + 1,
                pv.joins.iter().map(|j| j.round()).collect::<Vec<_>>(), pv.notes)),
            None => r.push_str("(none loaded)\n"),
        }
        r.push_str("\n=== log of the last run ===\n");
        r.push_str(if self.log.is_empty() { "(empty)\n" } else { &self.log });
        r.push_str("\n=== ACR crash reports (newest first) ===\n");
        r.push_str(&acr_crashes(3));
        let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let dir = if Path::new(&p.out_dir).is_dir() { PathBuf::from(&p.out_dir) } else { std::env::temp_dir() };
        let path = dir.join(format!("acr-stagegen-bugreport-{stamp}.txt"));
        std::fs::write(&path, redact(&r)).map_err(|e| e.to_string())?;
        Ok(path)
    }

    /// Remove installed generated stages from the game (restores the originals).
    fn uninstall(&mut self) {
        let p = self.prefs.clone();
        self.log.push_str("\n=== uninstall ===\n");
        self.tab = Tab::Log;
        let paks = PathBuf::from(&p.acr_dir).join(r"acr\Content\Paks").display().to_string();
        match backend_cmd(&p) {
            Ok((prog, mut full, cwd)) => {
                full.extend(["uninstall".to_string(), "--paks".into(), paks]);
                let (tx, rx) = channel();
                self.job = Some(Job { rx, remaining: vec![], total: 1, progress: 0.5, phase: "uninstalling".into() });
                spawn_build(prog, cwd, full, None, tx);
            }
            Err(e) => self.log.push_str(&format!("backend: {e}\n")),
        }
    }

    fn poll(&mut self) {
        let mut finished = None;
        if let Some(job) = &mut self.job {
            while let Ok(m) = job.rx.try_recv() {
                match m {
                    Msg::Line(l) => {
                        if let Some(rest) = l.strip_prefix("@progress ") {
                            let (f, phase) = rest.split_once(' ').unwrap_or((rest, ""));
                            job.progress = f.trim().parse().unwrap_or(job.progress);
                            job.phase = phase.to_string();
                            continue;
                        }
                        self.log.push_str(&l);
                        self.log.push('\n');
                    }
                    Msg::Done { ok, out } => finished = Some((ok, out)),
                }
            }
        }
        if let Some((ok, out)) = finished {
            if ok {
                if let Some(out) = out {
                    match load_preview(&out) {
                        Ok(pv) => {
                            self.preview = Some(pv);
                            self.tab = Tab::Map;
                        }
                        Err(e) => self.log.push_str(&format!("preview: {e}\n")),
                    }
                }
                self.next();
            } else {
                self.log.push_str("build failed (see above)\n");
                self.job = None;
            }
            if self.job.is_none() {
                self.sync_firewall();
            }
        }
    }
}

/// (program, leading args, working dir) of the backend: the embedded exe, or
/// `python -m stagegen.cli` in development.
fn backend_cmd(p: &Prefs) -> Result<(String, Vec<String>, String), String> {
    if p.use_python || BACKEND.is_none() {
        if !Path::new(&p.backend_dir).join("stagegen").is_dir() {
            return Err(format!("no stagegen package in {:?}", p.backend_dir));
        }
        return Ok((p.python.clone(), vec!["-m".into(), "stagegen.cli".into()], p.backend_dir.clone()));
    }
    let exe = embedded_backend()?;
    let cwd = exe.parent().map(|d| d.display().to_string()).unwrap_or_default();
    Ok((exe.display().to_string(), vec![], cwd))
}

fn spawn_build(python: String, cwd: String, args: Vec<String>, out: Option<PathBuf>, tx: Sender<Msg>) {
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

        if let Some(stage) = self.installed.clone() {
            let mut do_uninstall = false;
            let mut do_block = false;
            egui::TopBottomPanel::top("installed").show(ctx, |ui| {
                ui.add_space(4.0);
                ui.horizontal_wrapped(|ui| {
                    ui.colored_label(egui::Color32::from_rgb(230, 150, 40),
                        format!("⚠ Generated stage installed ({stage}): it replaces the original stage. \
                                 Play offline; times would count for the original."));
                    if self.blocked {
                        ui.colored_label(egui::Color32::from_rgb(90, 180, 90), "🔒 ACR online blocked");
                    } else if !busy && ui.small_button("Block ACR online").clicked() {
                        do_block = true;
                    }
                    if !busy && ui.small_button("Uninstall").clicked() {
                        do_uninstall = true;
                    }
                });
                ui.add_space(4.0);
            });
            if do_block {
                self.prefs.block_online = true;
                self.sync_firewall();
            }
            if do_uninstall {
                self.uninstall();
            }
        }

        let mut block_changed = false;
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
                path_row(ui, "Output", &mut p.out_dir, true);
                if BACKEND.is_some() {
                    ui.checkbox(&mut p.use_python, "Use Python backend (development)");
                }
                if p.use_python || BACKEND.is_none() {
                    let be_ok = Path::new(&p.backend_dir).join("stagegen").is_dir();
                    path_row(ui, "Backend", &mut p.backend_dir, be_ok);
                    path_row(ui, "Python", &mut p.python, true);
                }
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
                block_changed = ui.checkbox(&mut p.block_online, "Block ACR online while a stage is installed")
                    .on_hover_text("Adds a Windows Firewall rule for acr.exe while a generated stage is installed and \
                                    removes it on uninstall, so no times reach the online leaderboards. \
                                    Windows asks for admin rights each time it changes.")
                    .changed();
            });
            ui.add_space(8.0);
            let ready = steam::is_acr_dir(Path::new(&self.prefs.acr_dir))
                && (BACKEND.is_some() && !self.prefs.use_python
                    || Path::new(&self.prefs.backend_dir).join("stagegen").is_dir());
            ui.horizontal(|ui| {
                if let Some(job) = &self.job {
                    ui.vertical(|ui| {
                        let k = job.total - job.remaining.len();
                        let prefix = if job.total > 1 { format!("stage {k} of {}: ", job.total) } else { String::new() };
                        ui.add(egui::ProgressBar::new(job.progress).desired_width(300.0).animate(true)
                            .text(format!("{prefix}{} ({:.0} %)", job.phase, job.progress * 100.0)));
                    });
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
            if !busy
                && ui.add_enabled(ready, egui::Button::new("Uninstall generated stage"))
                    .on_hover_text("Remove the installed stage from the game (closes ACR); the original stage is back")
                    .clicked()
            {
                self.uninstall();
            }
            if !VERIFIED.contains(&self.prefs.stage.as_str()) {
                ui.colored_label(egui::Color32::from_rgb(230, 170, 60), "Experimental stage: not yet verified in-game.");
            }
            if let Some(pv) = &self.preview {
                ui.separator();
                ui.label(format!("{}: {:.2} km, {} pieces, {} pacenotes", pv.name, pv.length / 1000.0,
                                 pv.joins.len() + 1, pv.notes));
            }
            ui.separator();
            egui::CollapsingHeader::new("Share & bug report").show(ui, |ui| {
                let code = share_code(&self.prefs);
                ui.horizontal(|ui| {
                    ui.label("Share code:");
                    ui.monospace(&code);
                    if ui.small_button("Copy").on_hover_text("Same code = same stage for anyone").clicked() {
                        ui.ctx().copy_text(code.clone());
                    }
                });
                ui.horizontal(|ui| {
                    ui.add(egui::TextEdit::singleline(&mut self.code_input).hint_text("paste a share code")
                        .desired_width(220.0));
                    if ui.add_enabled(!busy, egui::Button::new("Load")).clicked() {
                        let input = self.code_input.clone();
                        match apply_share_code(&mut self.prefs, &input) {
                            Ok(()) => self.log.push_str(&format!("loaded share code {}\n", input.trim())),
                            Err(e) => self.log.push_str(&format!("share code: {e}\n")),
                        }
                    }
                });
                if ui.button("Export bug report").on_hover_text(
                    "Writes one text file (settings, last log, ACR crash reports; your user name is hidden) \
                     to attach to a bug report").clicked()
                {
                    match self.export_bug_report() {
                        Ok(path) => {
                            self.log.push_str(&format!("bug report written: {}\n", redact(&path.display().to_string())));
                            let _ = Command::new("explorer").arg(format!("/select,{}", path.display())).spawn();
                        }
                        Err(e) => self.log.push_str(&format!("bug report: {e}\n")),
                    }
                }
                ui.small(format!("acr-stagegen {VERSION}"));
            });
        });

        if block_changed && !busy {
            self.sync_firewall();
        }

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
