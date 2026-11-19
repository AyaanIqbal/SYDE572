"""Build the PDF report from its Markdown source. Requires reportlab."""
from pathlib import Path
import re
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'output/pdf/assignment4_report.pdf'


def inline(text):
    text = escape(text)
    text = re.sub(r'`([^`]+)`', r'<font name="Courier">\1</font>', text)
    return re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)


def footer(canvas, document):
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.HexColor('#52616b'))
    canvas.drawString(48, 30, 'Ayaan Iqbal | SYDE 572 | Assignment 4')
    canvas.drawRightString(564, 30, str(document.page))


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles['Title'].fontSize = 20
    styles['Title'].leading = 25
    styles['Title'].alignment = TA_LEFT
    styles['BodyText'].fontSize = 10
    styles['BodyText'].leading = 14
    styles['BodyText'].spaceAfter = 7
    styles['Heading2'].keepWithNext = True
    styles['Heading2'].fontSize = 13
    styles['Heading2'].leading = 17
    styles['Heading2'].spaceBefore = 12
    styles.add(ParagraphStyle('TableCell', parent=styles['BodyText'], fontSize=9, leading=12))
    lines = (ROOT/'report.md').read_text().splitlines()
    story = []
    position = 0
    while position < len(lines):
        line = lines[position]
        if not line.strip():
            position += 1
            continue
        if line.startswith('# '):
            story.append(Paragraph(inline(line[2:]), styles['Title']))
            story.append(Paragraph('Ayaan Iqbal | SYDE 572', styles['BodyText']))
        elif line.startswith('## '):
            story.append(Paragraph(inline(line[3:]), styles['Heading2']))
        elif line.startswith('|'):
            rows = []
            while position < len(lines) and lines[position].startswith('|'):
                cells = [c.strip() for c in lines[position].strip('|').split('|')]
                if not all(re.fullmatch(r'[-: ]+', c) for c in cells):
                    rows.append([Paragraph(inline(c), styles['TableCell']) for c in cells])
                position += 1
            table = Table(rows, colWidths=[516/len(rows[0])]*len(rows[0]), repeatRows=1)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e9eeea')),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('BOTTOMPADDING', (0,0), (-1,-1), 7),
                ('TOPPADDING', (0,0), (-1,-1), 7),
                ('LINEBELOW', (0,0), (-1,0), .5, colors.HexColor('#658176'))]))
            story.extend([table, Spacer(1, 9)])
            continue
        elif line.startswith('!['):
            match = re.fullmatch(r'!\[(.*?)\]\((.*?)\)', line)
            story.append(Image(str(ROOT/match[2]), width=516, height=206.4))
            story.append(Paragraph('Figure 1. Learning curve and held-out confusion matrix.', styles['BodyText']))
        else:
            paragraph = [line]
            position += 1
            while position < len(lines) and lines[position].strip() and not re.match(r'^(#|\||!\[|\d+\. )', lines[position]):
                paragraph.append(lines[position].strip())
                position += 1
            story.append(Paragraph(inline(' '.join(paragraph)), styles['BodyText']))
            continue
        position += 1
    document = SimpleDocTemplate(str(OUTPUT), pagesize=(612,792), leftMargin=48,
                                 rightMargin=48, topMargin=42, bottomMargin=48,
                                 title='Assignment 4: Text Classification with Naive Bayes', author='Ayaan Iqbal')
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == '__main__':
    main()
