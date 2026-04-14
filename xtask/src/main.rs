use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, anyhow, bail};
use cargo_metadata::{Metadata, MetadataCommand, Package, Target, TargetKind};
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Profile {
    Debug,
    Release,
}

impl Profile {
    fn cargo_args(self) -> &'static [&'static str] {
        match self {
            Self::Debug => &[],
            Self::Release => &["--release"],
        }
    }

    fn dir_name(self) -> &'static str {
        match self {
            Self::Debug => "debug",
            Self::Release => "release",
        }
    }
}

#[derive(Debug)]
enum CommandKind {
    BuildGallery { only: Option<String> },
    BuildApp { selected: String },
    BuildApps { only: Option<String> },
}

#[derive(Debug)]
struct Options {
    command: CommandKind,
    profile: Profile,
}

#[derive(Debug, Default, Deserialize)]
struct WorkspaceMetadataDoc {
    gallery: Option<WorkspaceGalleryConfigRaw>,
}

#[derive(Debug, Default, Deserialize)]
struct WorkspaceGalleryConfigRaw {
    frontend_dir: Option<String>,
    demo_apps_dir: Option<String>,
    examples_dir: Option<String>,
    dist_dir: Option<String>,
    shared_assets_dir: Option<String>,
    wasm_dist_dir: Option<String>,
}

#[derive(Debug, Clone)]
struct WorkspaceGalleryConfig {
    frontend_dir: String,
    demo_apps_dir: String,
    examples_dir: String,
    dist_dir: String,
    shared_assets_dir: String,
    wasm_dist_dir: String,
}

impl WorkspaceGalleryConfig {
    fn from_metadata(metadata: &Metadata) -> Result<Self> {
        let doc: WorkspaceMetadataDoc = serde_json::from_value(metadata.workspace_metadata.clone())
            .context("failed to parse [workspace.metadata]")?;
        let raw = doc.gallery.unwrap_or_default();
        Ok(Self {
            frontend_dir: raw
                .frontend_dir
                .unwrap_or_else(|| "demo_apps/gallery".to_string()),
            demo_apps_dir: raw.demo_apps_dir.unwrap_or_else(|| "demo_apps".to_string()),
            examples_dir: raw.examples_dir.unwrap_or_else(|| "examples".to_string()),
            dist_dir: raw.dist_dir.unwrap_or_else(|| "dist".to_string()),
            shared_assets_dir: raw
                .shared_assets_dir
                .unwrap_or_else(|| "examples/assets".to_string()),
            wasm_dist_dir: raw.wasm_dist_dir.unwrap_or_else(|| "wasm".to_string()),
        })
    }
}

#[derive(Debug, Deserialize)]
struct ExampleMetadataDoc {
    gallery: ExampleGalleryMetadata,
}

#[derive(Debug, Clone, Deserialize)]
struct ExampleGalleryMetadata {
    name: String,
    category: String,
    description: String,
    order: Option<i32>,
    web: Option<bool>,
    note: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PackageMetadataDoc {
    gallery: Option<DemoAppGalleryMetadata>,
}

#[derive(Debug, Clone, Deserialize)]
struct DemoAppGalleryMetadata {
    name: String,
    category: String,
    description: String,
    order: Option<i32>,
    web: Option<bool>,
    note: Option<String>,
}

#[derive(Debug, Clone, Copy)]
enum GalleryItemKind {
    Iframe,
    Standalone,
}

impl GalleryItemKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Iframe => "iframe",
            Self::Standalone => "standalone",
        }
    }
}

#[derive(Debug, Clone)]
struct ExampleSpec {
    id: String,
    target_name: String,
    name: String,
    category: String,
    description: String,
    order: i32,
    source_path: String,
    source_url: Option<String>,
    web_supported: bool,
    note: Option<String>,
}

#[derive(Debug, Clone)]
struct AppSpec {
    id: String,
    package_name: String,
    name: String,
    category: String,
    description: String,
    order: i32,
    source_path: String,
    source_url: Option<String>,
    url: String,
    web_supported: bool,
    note: Option<String>,
    manifest_dir: PathBuf,
}

#[derive(Debug, Clone)]
struct ManifestEntry {
    id: String,
    name: String,
    category: String,
    description: String,
    order: i32,
    kind: GalleryItemKind,
    source_path: String,
    source_url: Option<String>,
    url: Option<String>,
    web_supported: bool,
    note: Option<String>,
}

#[derive(Debug, Serialize)]
struct ManifestCategory {
    category: String,
    items: Vec<ManifestItem>,
}

#[derive(Debug, Serialize)]
struct ManifestItem {
    id: String,
    name: String,
    #[serde(rename = "type")]
    kind: String,
    description: String,
    source_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    url: Option<String>,
    web_supported: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    note: Option<String>,
}

fn main() -> Result<()> {
    let options = parse_args()?;
    let workspace_root = workspace_root()?;
    let metadata = metadata(&workspace_root)?;
    let config = WorkspaceGalleryConfig::from_metadata(&metadata)?;

    match &options.command {
        CommandKind::BuildGallery { only } => build_gallery(
            &workspace_root,
            &metadata,
            &config,
            options.profile,
            only.as_deref(),
        ),
        CommandKind::BuildApp { selected } => build_apps(
            &workspace_root,
            &metadata,
            &config,
            options.profile,
            Some(selected.as_str()),
        ),
        CommandKind::BuildApps { only } => build_apps(
            &workspace_root,
            &metadata,
            &config,
            options.profile,
            only.as_deref(),
        ),
    }
}

fn parse_args() -> Result<Options> {
    let mut args = env::args().skip(1);
    let command = args.next().ok_or_else(|| anyhow!(
        "usage: cargo xtask <build-gallery|build-app <id>> [--debug|--release] [--only <example_id>]"
    ))?;

    match command.as_str() {
        "build-gallery" => {
            let (profile, only) = parse_build_flags(args.collect())?;
            Ok(Options {
                command: CommandKind::BuildGallery { only },
                profile,
            })
        }
        "build-app" => {
            let selected = args.next().context("build-app requires an app id")?;
            let (profile, _) = parse_build_flags(args.collect())?;
            Ok(Options {
                command: CommandKind::BuildApp { selected },
                profile,
            })
        }
        "build-apps" => {
            let (profile, only) = parse_build_flags(args.collect())?;
            Ok(Options {
                command: CommandKind::BuildApps { only },
                profile,
            })
        }
        other => bail!("unknown xtask command: {other}"),
    }
}

fn parse_build_flags(args: Vec<String>) -> Result<(Profile, Option<String>)> {
    let mut profile = Profile::Release;
    let mut only = None;
    let mut index = 0;

    while index < args.len() {
        match args[index].as_str() {
            "--debug" => profile = Profile::Debug,
            "--release" => profile = Profile::Release,
            "--only" => {
                let value = args
                    .get(index + 1)
                    .context("--only requires an id")?
                    .clone();
                only = Some(value);
                index += 1;
            }
            other => bail!("unknown xtask argument: {other}"),
        }

        index += 1;
    }

    Ok((profile, only))
}

fn workspace_root() -> Result<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .map(Path::to_path_buf)
        .context("failed to resolve workspace root")
}

fn build_gallery(
    workspace_root: &Path,
    metadata: &Metadata,
    config: &WorkspaceGalleryConfig,
    profile: Profile,
    only: Option<&str>,
) -> Result<()> {
    let repo_url = repository_url(metadata)?;
    let mut examples = collect_examples(workspace_root, metadata, config, &repo_url)?;
    let apps = collect_demo_apps(workspace_root, metadata, config, &repo_url)?;

    if let Some(selected) = only {
        examples.retain(|spec| spec.id == selected);
        if examples.is_empty() {
            bail!("example '{selected}' was not found in gallery metadata");
        }
    }

    let dist_root = prepare_clean_dir(&workspace_root.join(&config.dist_dir))?;
    sync_shared_assets(workspace_root, config, &dist_root)?;

    let gallery_root = dist_root.clone();
    stage_gallery_frontend(workspace_root, config, &gallery_root)?;

    let manifest_entries =
        build_manifest_entries(&examples, if only.is_none() { &apps } else { &[] });
    write_manifest(&gallery_root.join("examples.json"), &manifest_entries)?;

    let wasm_root = prepare_clean_dir(&gallery_root.join(&config.wasm_dist_dir))?;
    for spec in examples.iter().filter(|spec| spec.web_supported) {
        build_example_target(workspace_root, profile, spec, &wasm_root)?;
    }

    if only.is_none() {
        for app in apps.iter().filter(|spec| spec.web_supported) {
            build_app_spec(workspace_root, profile, app, &dist_root)?;
        }
    }

    Ok(())
}

fn build_apps(
    workspace_root: &Path,
    metadata: &Metadata,
    config: &WorkspaceGalleryConfig,
    profile: Profile,
    selected: Option<&str>,
) -> Result<()> {
    let repo_url = repository_url(metadata)?;
    let mut apps = collect_demo_apps(workspace_root, metadata, config, &repo_url)?;
    if let Some(id) = selected {
        apps.retain(|app| app.id == id || app.package_name == id);
        if apps.is_empty() {
            bail!("app '{id}' was not found in demo_apps metadata");
        }
    }

    let dist_root = prepare_dist(&workspace_root.join(&config.dist_dir))?;
    sync_shared_assets(workspace_root, config, &dist_root)?;

    for app in apps.iter().filter(|spec| spec.web_supported) {
        build_app_spec(workspace_root, profile, app, &dist_root)?;
    }

    Ok(())
}

fn metadata(workspace_root: &Path) -> Result<Metadata> {
    MetadataCommand::new()
        .current_dir(workspace_root)
        .exec()
        .context("failed to query cargo metadata")
}

fn repository_url(metadata: &Metadata) -> Result<String> {
    let root_package = metadata
        .root_package()
        .context("workspace root package not found")?;
    root_package
        .repository
        .clone()
        .or_else(|| root_package.homepage.clone())
        .context("workspace package is missing repository metadata")
}

fn collect_examples(
    workspace_root: &Path,
    metadata: &Metadata,
    config: &WorkspaceGalleryConfig,
    repo_url: &str,
) -> Result<Vec<ExampleSpec>> {
    let root_package = metadata
        .root_package()
        .context("workspace root package not found")?;
    let examples_root = workspace_root.join(&config.examples_dir);

    let mut specs = Vec::new();
    for target in root_example_targets(root_package)? {
        let source_path = target.src_path.as_std_path();
        if !source_path.starts_with(&examples_root) {
            continue;
        }

        // skip examples that are not having [gallery] metadata block
        let Some(gallery) = parse_example_metadata(source_path)? else {
            continue;
        };

        let source_rel = normalize_path(source_path.strip_prefix(workspace_root)?);

        specs.push(ExampleSpec {
            id: target.name.clone(),
            target_name: target.name.clone(),
            name: gallery.name,
            category: gallery.category,
            description: gallery.description,
            order: gallery.order.unwrap_or(0),
            source_path: source_rel.clone(),
            source_url: Some(source_url(repo_url, &source_rel)),
            web_supported: gallery.web.unwrap_or(true),
            note: gallery.note,
        });
    }

    specs.sort_by(|left, right| {
        left.order
            .cmp(&right.order)
            .then_with(|| left.category.cmp(&right.category))
            .then_with(|| left.name.cmp(&right.name))
    });

    Ok(specs)
}

fn collect_demo_apps(
    workspace_root: &Path,
    metadata: &Metadata,
    config: &WorkspaceGalleryConfig,
    repo_url: &str,
) -> Result<Vec<AppSpec>> {
    let demo_apps_root = workspace_root.join(&config.demo_apps_dir);
    let mut specs = Vec::new();

    for package in &metadata.packages {
        let manifest_path = package.manifest_path.as_std_path();
        if !manifest_path.starts_with(&demo_apps_root) {
            continue;
        }

        let metadata_doc: PackageMetadataDoc = serde_json::from_value(package.metadata.clone())
            .with_context(|| format!("failed to parse package metadata for {}", package.name))?;
        let Some(gallery) = metadata_doc.gallery else {
            continue;
        };

        let manifest_dir = manifest_path
            .parent()
            .context("package manifest has no parent directory")?
            .to_path_buf();

        let source_path = package
            .targets
            .iter()
            .find(|target| target.kind.iter().any(|kind| kind == &TargetKind::Bin))
            .map(|target| target.src_path.as_std_path().to_path_buf())
            .unwrap_or_else(|| manifest_dir.join("main.rs"));
        let source_rel = normalize_path(source_path.strip_prefix(workspace_root)?);

        specs.push(AppSpec {
            id: package.name.to_string(),
            package_name: package.name.to_string(),
            name: gallery.name,
            category: gallery.category,
            description: gallery.description,
            order: gallery.order.unwrap_or(0),
            source_path: source_rel.clone(),
            source_url: Some(source_url(repo_url, &source_rel)),
            url: format!("../{}/index.html", package.name),
            web_supported: gallery.web.unwrap_or(true),
            note: gallery.note,
            manifest_dir,
        });
    }

    specs.sort_by(|left, right| {
        left.order
            .cmp(&right.order)
            .then_with(|| left.category.cmp(&right.category))
            .then_with(|| left.name.cmp(&right.name))
    });

    Ok(specs)
}

fn parse_example_metadata(source_path: &Path) -> Result<Option<ExampleGalleryMetadata>> {
    let contents = fs::read_to_string(source_path)
        .with_context(|| format!("failed to read {}", source_path.display()))?;

    let mut block_lines = Vec::new();
    let mut in_block = false;
    for line in contents.lines().take(48) {
        let trimmed = line.trim_start();
        let Some(rest) = trimmed.strip_prefix("//!") else {
            if in_block || !trimmed.is_empty() {
                break;
            }
            continue;
        };

        let content = rest.trim_start();
        if !in_block {
            if content == "[gallery]" {
                in_block = true;
                block_lines.push(content.to_string());
            }
            continue;
        }

        if content.is_empty() {
            break;
        }
        block_lines.push(content.to_string());
    }

    if block_lines.is_empty() {
        return Ok(None);
    }

    let doc: ExampleMetadataDoc = toml::from_str(&block_lines.join("\n")).with_context(|| {
        format!(
            "failed to parse [gallery] metadata in {}",
            source_path.display()
        )
    })?;

    Ok(Some(doc.gallery))
}

fn build_manifest_entries(examples: &[ExampleSpec], apps: &[AppSpec]) -> Vec<ManifestEntry> {
    let mut entries = Vec::with_capacity(examples.len() + apps.len());

    for example in examples {
        entries.push(ManifestEntry {
            id: example.id.clone(),
            name: example.name.clone(),
            category: example.category.clone(),
            description: example.description.clone(),
            order: example.order,
            kind: GalleryItemKind::Iframe,
            source_path: example.source_path.clone(),
            source_url: example.source_url.clone(),
            url: None,
            web_supported: example.web_supported,
            note: example.note.clone(),
        });
    }

    for app in apps {
        entries.push(ManifestEntry {
            id: app.id.clone(),
            name: app.name.clone(),
            category: app.category.clone(),
            description: app.description.clone(),
            order: app.order,
            kind: GalleryItemKind::Standalone,
            source_path: app.source_path.clone(),
            source_url: app.source_url.clone(),
            url: Some(app.url.clone()),
            web_supported: app.web_supported,
            note: app.note.clone(),
        });
    }

    entries.sort_by(|left, right| {
        left.order
            .cmp(&right.order)
            .then_with(|| left.category.cmp(&right.category))
            .then_with(|| left.name.cmp(&right.name))
    });

    entries
}

fn write_manifest(path: &Path, entries: &[ManifestEntry]) -> Result<()> {
    let mut grouped: BTreeMap<String, Vec<&ManifestEntry>> = BTreeMap::new();
    for entry in entries {
        grouped
            .entry(entry.category.clone())
            .or_default()
            .push(entry);
    }

    let categories: Vec<ManifestCategory> = grouped
        .into_iter()
        .map(|(category, mut items)| {
            items.sort_by(|left, right| {
                left.order
                    .cmp(&right.order)
                    .then_with(|| left.name.cmp(&right.name))
            });
            ManifestCategory {
                category,
                items: items
                    .into_iter()
                    .map(|entry| ManifestItem {
                        id: entry.id.clone(),
                        name: entry.name.clone(),
                        kind: entry.kind.as_str().to_string(),
                        description: entry.description.clone(),
                        source_path: entry.source_path.clone(),
                        source_url: entry.source_url.clone(),
                        url: entry.url.clone(),
                        web_supported: entry.web_supported,
                        note: entry.note.clone(),
                    })
                    .collect(),
            }
        })
        .collect();

    let json = serde_json::to_string_pretty(&categories).context("failed to serialize manifest")?;
    fs::write(path, json).with_context(|| format!("failed to write {}", path.display()))
}

fn root_example_targets(root_package: &Package) -> Result<Vec<&Target>> {
    let targets: Vec<&Target> = root_package
        .targets
        .iter()
        .filter(|target| target.kind.iter().any(|kind| kind == &TargetKind::Example))
        .collect();

    if targets.is_empty() {
        bail!("no root example targets were discovered");
    }

    Ok(targets)
}

fn prepare_dist(dist_root: &Path) -> Result<PathBuf> {
    fs::create_dir_all(dist_root)
        .with_context(|| format!("failed to create {}", dist_root.display()))?;
    Ok(dist_root.to_path_buf())
}

fn prepare_clean_dir(dir: &Path) -> Result<PathBuf> {
    if dir.exists() {
        fs::remove_dir_all(dir).with_context(|| format!("failed to clear {}", dir.display()))?;
    }
    fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
    Ok(dir.to_path_buf())
}

fn sync_shared_assets(
    workspace_root: &Path,
    config: &WorkspaceGalleryConfig,
    dist_root: &Path,
) -> Result<()> {
    let source = workspace_root.join(&config.shared_assets_dir);
    let destination = dist_root.join("assets");
    replace_dir(&source, &destination)
}

fn stage_gallery_frontend(
    workspace_root: &Path,
    config: &WorkspaceGalleryConfig,
    gallery_root: &Path,
) -> Result<()> {
    let frontend_dir = workspace_root.join(&config.frontend_dir);
    replace_dir(&frontend_dir, gallery_root)
}

fn build_example_target(
    workspace_root: &Path,
    profile: Profile,
    spec: &ExampleSpec,
    wasm_root: &Path,
) -> Result<()> {
    let mut cargo = Command::new("cargo");
    cargo
        .current_dir(workspace_root)
        .env("MYTH_ASSET_PATH", "../assets/")
        .args([
            "build",
            "--package",
            "myth-engine",
            "--target",
            "wasm32-unknown-unknown",
            "--example",
            &spec.target_name,
        ])
        .args(profile.cargo_args());
    run_command(&mut cargo, &format!("building gallery example {}", spec.id))?;

    let wasm_path = workspace_root
        .join("target")
        .join("wasm32-unknown-unknown")
        .join(profile.dir_name())
        .join("examples")
        .join(format!("{}.wasm", spec.target_name));

    wasm_bindgen(&wasm_path, wasm_root)?;
    optimize_wasm(profile, wasm_root, &spec.target_name)
}

fn build_app_spec(
    workspace_root: &Path,
    profile: Profile,
    spec: &AppSpec,
    dist_root: &Path,
) -> Result<()> {
    let mut cargo = Command::new("cargo");
    cargo
        .current_dir(workspace_root)
        .env("MYTH_ASSET_PATH", "../assets/")
        .args([
            "build",
            "--package",
            spec.package_name.as_str(),
            "--target",
            "wasm32-unknown-unknown",
        ])
        .args(profile.cargo_args());
    run_command(&mut cargo, &format!("building app {}", spec.id))?;

    let wasm_path = workspace_root
        .join("target")
        .join("wasm32-unknown-unknown")
        .join(profile.dir_name())
        .join(format!("{}.wasm", spec.package_name));

    let app_root = dist_root.join(&spec.id);
    let web_source = spec.manifest_dir.join("web");
    replace_dir(&web_source, &app_root)?;

    let output_dir = prepare_clean_dir(&app_root.join("pkg"))?;
    wasm_bindgen(&wasm_path, &output_dir)?;
    optimize_wasm(profile, &output_dir, &spec.package_name)
}

fn wasm_bindgen(wasm_path: &Path, output_dir: &Path) -> Result<()> {
    fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    let mut command = Command::new("wasm-bindgen");
    command.arg(wasm_path).args([
        "--out-dir",
        output_dir.to_str().context("non-utf8 output dir")?,
        "--target",
        "web",
        "--no-typescript",
    ]);

    run_command(
        &mut command,
        &format!("running wasm-bindgen for {}", wasm_path.display()),
    )
}

fn optimize_wasm(profile: Profile, output_dir: &Path, module_name: &str) -> Result<()> {
    if profile != Profile::Release {
        return Ok(());
    }

    let input = output_dir.join(format!("{module_name}_bg.wasm"));
    if !input.exists() {
        return Ok(());
    }

    let status = Command::new("wasm-opt")
        .args(["-Oz", "-o"])
        .arg(&input)
        .arg(&input)
        .status();

    match status {
        Ok(result) if result.success() => Ok(()),
        Ok(_) => bail!("wasm-opt failed for {}", input.display()),
        Err(_) => Ok(()),
    }
}

fn replace_dir(source: &Path, destination: &Path) -> Result<()> {
    if !destination.exists() {
        fs::create_dir_all(destination)
            .with_context(|| format!("failed to create {}", destination.display()))?;
    }

    for entry in WalkDir::new(source) {
        let entry = entry?;
        let path = entry.path();
        let relative = path.strip_prefix(source)?;
        let target = destination.join(relative);

        if entry.file_type().is_dir() {
            fs::create_dir_all(&target)
                .with_context(|| format!("failed to create {}", target.display()))?;
            continue;
        }

        copy_file(path, &target)?;
    }

    Ok(())
}

fn copy_file(source: &Path, destination: &Path) -> Result<()> {
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    fs::copy(source, destination).with_context(|| {
        format!(
            "failed to copy {} -> {}",
            source.display(),
            destination.display()
        )
    })?;
    Ok(())
}

fn source_url(repository: &str, source_path: &str) -> String {
    let repo = repository.trim_end_matches(".git");
    let source_path = source_path.replace('\\', "/");
    if repo.contains("github.com") {
        format!("{repo}/blob/main/{source_path}")
    } else {
        format!("{repo}/{source_path}")
    }
}

fn normalize_path(path: &Path) -> String {
    path.components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

fn run_command(command: &mut Command, context_message: &str) -> Result<()> {
    let status = command
        .status()
        .with_context(|| format!("failed while {context_message}"))?;
    if !status.success() {
        bail!("command failed while {context_message}");
    }
    Ok(())
}
