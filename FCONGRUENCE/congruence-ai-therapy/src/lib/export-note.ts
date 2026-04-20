/**
 * Client-side note export. v1: zero dependencies, uses window.print() for PDF
 * and a Word-compatible HTML blob for .doc. If we want pixel-perfect server-
 * rendered PDFs later, swap this out for an edge function that returns a PDF.
 */

interface ExportArgs {
  title: string;
  dateIso: string;
  markdown: string;
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function markdownToHtml(markdown: string): string {
  // Lightweight markdown rendering sufficient for clinical notes.
  // Headings, bold, italic, lists, paragraphs. No external dep.
  const lines = markdown.split("\n");
  const out: string[] = [];
  let inList = false;

  const closeList = () => {
    if (inList) {
      out.push("</ul>");
      inList = false;
    }
  };

  for (const raw of lines) {
    const line = raw.trimEnd();

    if (!line.trim()) {
      closeList();
      out.push("");
      continue;
    }

    const h1 = /^#\s+(.*)$/.exec(line);
    const h2 = /^##\s+(.*)$/.exec(line);
    const h3 = /^###\s+(.*)$/.exec(line);
    const bullet = /^[-*]\s+(.*)$/.exec(line);

    if (h1) { closeList(); out.push(`<h1>${escapeHtml(h1[1])}</h1>`); continue; }
    if (h2) { closeList(); out.push(`<h2>${escapeHtml(h2[1])}</h2>`); continue; }
    if (h3) { closeList(); out.push(`<h3>${escapeHtml(h3[1])}</h3>`); continue; }

    if (bullet) {
      if (!inList) { out.push("<ul>"); inList = true; }
      out.push(`<li>${formatInline(escapeHtml(bullet[1]))}</li>`);
      continue;
    }

    closeList();
    out.push(`<p>${formatInline(escapeHtml(line))}</p>`);
  }
  closeList();
  return out.join("\n");
}

function formatInline(s: string): string {
  return s
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>");
}

function buildHtml({ title, dateIso, markdown }: ExportArgs): string {
  const body = markdownToHtml(markdown);
  return `<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>${escapeHtml(title)}</title>
    <style>
      body { font-family: Arial, Helvetica, sans-serif; font-size: 11pt; line-height: 1.55; color: #111; margin: 1in; }
      h1 { font-size: 16pt; margin: 0 0 6pt 0; border-bottom: 1px solid #888; padding-bottom: 4pt; }
      h2 { font-size: 12pt; margin: 18pt 0 6pt; text-transform: uppercase; letter-spacing: .04em; border-bottom: 1px solid #bbb; padding-bottom: 3pt; }
      h3 { font-size: 11pt; margin: 12pt 0 4pt; }
      p { margin: 0 0 8pt; }
      ul { margin: 0 0 10pt 1.2em; padding: 0; }
      li { margin: 0 0 4pt; }
      .meta { color: #666; font-size: 9pt; margin-bottom: 14pt; }
      @media print { body { margin: 0.75in; } }
    </style>
  </head>
  <body>
    <h1>${escapeHtml(title)}</h1>
    <div class="meta">Generated ${escapeHtml(new Date(dateIso).toLocaleString())}</div>
    ${body}
  </body>
</html>`;
}

/**
 * Open a print dialog with the rendered note. User picks "Save as PDF" from
 * the print dialog. Zero-dependency v1 approach.
 */
export function exportNoteAsPdf(args: ExportArgs): void {
  const html = buildHtml(args);
  const w = window.open("", "_blank", "noopener,noreferrer");
  if (!w) {
    // Popup blocked — fall back to opening in the current tab.
    const blob = new Blob([html], { type: "text/html" });
    const url = URL.createObjectURL(blob);
    window.location.href = url;
    return;
  }
  w.document.open();
  w.document.write(html);
  w.document.close();
  // Give the window time to render, then trigger print.
  w.onload = () => {
    setTimeout(() => {
      w.focus();
      w.print();
    }, 250);
  };
}

/**
 * Download the rendered note as a .doc file (Word-compatible HTML).
 */
export function exportNoteAsDoc(args: ExportArgs): void {
  const html = buildHtml(args);
  const blob = new Blob([html], { type: "application/msword" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const safeTitle = args.title.replace(/[^\w\-. ]+/g, "_");
  const dateStr = args.dateIso.split("T")[0];
  a.href = url;
  a.download = `${safeTitle} - Note - ${dateStr}.doc`;
  a.click();
  URL.revokeObjectURL(url);
}
