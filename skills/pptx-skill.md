---
name: pptx-skill
title: PowerPoint presentations with python-pptx
description: Create .pptx slide decks (research talks, summaries, lecture slides) by writing and running a python-pptx script with run_python, including figures generated with matplotlib.
triggers: powerpoint, pptx, slides, slide deck, deck, presentation, keynote, talk slides, lecture slides
tools: run_python, list_files, search_arxiv
---

# PowerPoint presentations (python-pptx)

STEM OS has no dedicated presentation command. Build decks by calling **run_python** with a
python-pptx script. The script runs inside the project workspace, so saved files appear as
project artifacts automatically (the user downloads the .pptx from the Outputs panel).

## Workflow
1. Plan the deck first: audience, 6–12 slides, one message per slide. Typical research-talk order:
   Title → Motivation → Background → Method → Results (1–3 slides) → Limitations → Conclusion → References.
2. Gather content before writing slides (search_arxiv / evidence already in the conversation).
   Put citations as short "[n] Author, year" lines on the slide where a claim appears.
3. If figures help, create them in the SAME run_python call with matplotlib, `savefig('figures/<name>.png', dpi=200)`, then insert them.
4. Save to `slides/<short-name>.pptx` and print the path. Never claim a deck exists without a successful run (exit_code 0).

## Template script
```python
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

os.makedirs("slides", exist_ok=True)
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)   # 16:9

def title_slide(title, subtitle):
    s = prs.slides.add_slide(prs.slide_layouts[0])
    s.shapes.title.text = title
    s.placeholders[1].text = subtitle

def bullet_slide(title, bullets, notes=""):
    s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = title
    tf = s.placeholders[1].text_frame
    tf.clear()
    for i, b in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        level = 1 if b.startswith("  ") else 0
        p.text, p.level = b.strip(), level
        p.font.size = Pt(22 if level == 0 else 18)
    if notes:
        s.notes_slide.notes_text_frame.text = notes

def image_slide(title, image_path, caption=""):
    s = prs.slides.add_slide(prs.slide_layouts[5])          # title only
    s.shapes.title.text = title
    s.shapes.add_picture(image_path, Inches(1.5), Inches(1.4), height=Inches(5.2))
    if caption:
        tb = s.shapes.add_textbox(Inches(1.5), Inches(6.7), Inches(10), Inches(0.5))
        tb.text_frame.text = caption
        tb.text_frame.paragraphs[0].font.size = Pt(14)
        tb.text_frame.paragraphs[0].font.color.rgb = RGBColor(0x55, 0x55, 0x55)

title_slide("Diffusion Models", "From DDPM to fast samplers")
bullet_slide("Motivation", ["Generate high-fidelity samples", "  stable training vs GANs [1]"], notes="Speaker notes here")
prs.save("slides/diffusion-models.pptx")
print("saved slides/diffusion-models.pptx", len(prs.slides), "slides")
```

## Rules
- At most ~6 bullets per slide, ≤ 12 words each; put details in speaker notes.
- Equations: python-pptx cannot render LaTeX. Render the equation to PNG with matplotlib mathtext
  (`fig.text(0.5, 0.5, r"$E=mc^2$", fontsize=28)`; `savefig(..., bbox_inches="tight", transparent=True)`) and insert it with `image_slide`.
- Layout indexes in the default template: 0 title, 1 title+content, 5 title only, 6 blank.
- After the run, tell the user the file name, slide count and slide titles.
