

import * as fs from 'fs'
import * as path from 'path'
import { fileURLToPath } from "url"

console.log("Copying data files from plugin...")

const thisDir = path.dirname(fileURLToPath(import.meta.url))
const pluginDir = path.join(thisDir, "../../ai_diffusion")
const srcDir = path.join(thisDir, "../src")
const dataDir = path.join(srcDir, "content/data")
const iconsDir = path.join(srcDir, "icons/plugin")

if (!fs.existsSync(dataDir)) {
    fs.mkdirSync(dataDir)
}
fs.copyFileSync(
    path.join(pluginDir, "presets/models.json"),
    path.join(dataDir, "models.json")
)

if (!fs.existsSync(iconsDir)) {
    fs.mkdirSync(iconsDir)
}
fs.readdirSync(path.join(pluginDir, "icons")).forEach(file => {
    if (file.includes("-dark")) {
        fs.copyFileSync(
            path.join(pluginDir, "icons", file),
            path.join(iconsDir, file)
        )
    }
})