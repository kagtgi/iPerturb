"""Render a PDF as a grid of page thumbnails: py scripts/contact_sheet.py in.pdf out.png [first] [last] [cols]"""
import sys

import fitz

pdf, out = sys.argv[1], sys.argv[2]
first = int(sys.argv[3]) if len(sys.argv) > 3 else 1
last = int(sys.argv[4]) if len(sys.argv) > 4 else None
cols = int(sys.argv[5]) if len(sys.argv) > 5 else 6
d = fitz.open(pdf)
last = last or len(d)
pages = list(range(first - 1, last))
z = 0.55
w0, h0 = d[0].rect.width * z, d[0].rect.height * z
rows = (len(pages) + cols - 1) // cols
sheet = fitz.open()
pg = sheet.new_page(width=w0 * cols, height=h0 * rows)
for k, i in enumerate(pages):
    r, c = divmod(k, cols)
    rect = fitz.Rect(c * w0, r * h0, (c + 1) * w0, (r + 1) * h0)
    pg.show_pdf_page(rect, d, i)
    pg.draw_rect(rect, color=(0.6, 0.6, 0.6), width=0.5)
    pg.insert_text((rect.x0 + 4, rect.y0 + 10), str(i + 1), fontsize=9, color=(1, 0, 0))
pg.get_pixmap(dpi=110).save(out)
print(out, len(pages))
