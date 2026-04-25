# Screenshots to capture

After your first successful run, capture and commit these images. The README
references them by these exact filenames.

| Filename | What it should show |
|----------|--------------------|
| `screenshot.png` | Full app at landing — sidebar visible, sample buttons, no upload yet |
| `heatmap.png` | A loaded sample showing original image + Grad-CAM heatmap + probability bar |
| `report.png` *(optional)* | The downloaded PDF opened in a viewer |
| `demo.gif` *(optional)* | 10–20 s screen recording of the full flow: click sample → see prediction → download PDF |

## Quick capture (macOS)

```bash
# Single window screenshot (Cmd+Shift+4 then Space, or:)
screencapture -W docs/screenshot.png

# Full screen GIF: use Kap (https://getkap.co) or Gifox, export to docs/demo.gif
```

Aim for **1600×1000 px** or larger so the screenshots stay sharp on retina
displays in the GitHub README.
