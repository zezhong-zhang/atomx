"""Check PDF boundaries and create a local contact sheet (optional renderer deps).

Run in a separate environment with pymupdf and pillow. Text boundaries are a
mechanical check; they do not replace viewing the pages.
"""
import json
from pathlib import Path
import pymupdf
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parent
doc=pymupdf.open(ROOT/"研究体系与案例设计.pdf")
outside=[];tiles=[]
for index,page in enumerate(doc):
    pix=page.get_pixmap(matrix=pymupdf.Matrix(.6,.6))
    im=Image.frombytes("RGB",[pix.width,pix.height],pix.samples)
    im.thumbnail((300,425))
    tile=Image.new("RGB",(320,455),"#dce4e9")
    tile.paste(im,((320-im.width)//2,8))
    ImageDraw.Draw(tile).text((14,436),f"Page {index+1}",fill="black");tiles.append(tile)
    for block in page.get_text("dict")["blocks"]:
        if block["type"]!=0:continue
        for line in block["lines"]:
            for span in line["spans"]:
                x0,y0,x1,y1=span["bbox"]
                if x0 < -1 or y0 < -1 or x1 > page.rect.width+1 or y1 > page.rect.height+1:
                    outside.append({"page":index+1,"bbox":list(span["bbox"]),"text":span["text"]})
sheet=Image.new("RGB",(1280,((len(tiles)+3)//4)*455),"#b8c6d0")
for i,tile in enumerate(tiles):sheet.paste(tile,((i%4)*320,(i//4)*455))
sheet.save(ROOT/"layout_contact_sheet.jpg")
doc[-1].get_pixmap(matrix=pymupdf.Matrix(1.4,1.4)).save(ROOT/"layout_last_page.png")
text="\n".join(page.get_text() for page in doc)
all_cases=all(f"S{i:02}" in text for i in range(1,13))
summary=pymupdf.open(ROOT/"故事线摘要.pdf")
report={"main_pages":len(doc),"summary_pages":len(summary),"all_case_ids_in_pdf":all_cases,"out_of_page_text":outside,"scope":"PDF extraction and page-boundary check; visual inspection separately recorded"}
(ROOT/"layout_check.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(report,ensure_ascii=False))
assert all_cases and not outside
