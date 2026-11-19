# Birdcount slides

Open `index.html` in a browser. The deck works offline with the CSS, JavaScript and `media/` folder beside it.

- **← / →** or **Space** to advance through the deck.
- **O** opens the slide overview. **N** shows speaker notes. **F** toggles fullscreen.
- Click the play buttons on the detection and embedding slides to hear the examples. Select a song pair on the MLP slide to inspect its score, or select a step on the average-linkage slide to inspect its distances.

## Add a slide

Copy a `<section class="slide" …>` in `index.html`. Give the section and heading new IDs, and point `aria-labelledby` to the new heading. Sections are shown in file order and their IDs double as deep links. Put details and credits in the section’s `<aside class="notes">`. Put new images and audio in `media/` and reference them by relative path. The slide overview and counter update automatically.

## Refresh the figures

Run `.venv/bin/python slides/refresh-data.py` from the repo root to rebuild `data.js` and the demo MP3s. The script uses the repo’s experiment outputs, cached HawkEars detections and source WAVs, and needs FFmpeg and the experiment’s Python dependencies. Review slide text and speaker notes if you change the experiment or demo recordings.
