// Stage the web build for Capacitor: the trainer is ONE self-contained page (fonts, packs,
// SVG all inlined), so the app's web assets are just the repo-root index.html.
// Rebuild it first with:  PYTHONPATH=src python demo/build_trainer.py
import { copyFileSync, mkdirSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const src = join(here, "..", "..", "index.html");
const www = join(here, "..", "www");

mkdirSync(www, { recursive: true });
copyFileSync(src, join(www, "index.html"));
const mb = (statSync(src).size / 1e6).toFixed(2);
console.log(`staged index.html (${mb} MB) -> mobile/www/`);
