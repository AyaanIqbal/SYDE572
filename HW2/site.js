// Read the actual assignment outputs; no duplicate website copies to maintain.
async function readFile(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`Unable to load ${path} (${response.status})`);
  return response.text();
}

function element(tag, text) {
  const node = document.createElement(tag);
  node.textContent = text;
  return node;
}

// Render the plain headings, paragraphs, lists and pipe tables in our notes.
// Text is inserted as textContent, so note/source files cannot inject HTML.
function renderNotes(text) {
  const fragment = document.createDocumentFragment();
  const lines = text.replace(/\r/g, "").split("\n");
  let i = 0;
  const clean = value => value.replace(/\*\*(.*?)\*\*/g, "$1");
  while (i < lines.length) {
    const line = lines[i].trim();
    if (!line) { i++; continue; }
    if (line.startsWith("|")) {
      const rows = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        const row = lines[i++].trim().split("|").slice(1, -1).map(x => x.trim());
        if (!row.every(cell => /^:?-+:?$/.test(cell))) rows.push(row);
      }
      fragment.append(renderTable(rows));
    } else if (/^#{1,3} /.test(line)) {
      const depth = line.match(/^#+/)[0].length;
      // The surrounding section already names the part.
      if (depth > 1) fragment.append(element("h3", line.replace(/^#+ /, "")));
      i++;
    } else if (line.startsWith("- ")) {
      const list = document.createElement("ul");
      while (i < lines.length && lines[i].trim().startsWith("- ")) {
        let item = lines[i++].trim().slice(2);
        while (i < lines.length && /^\s+\S/.test(lines[i])) item += " " + lines[i++].trim();
        list.append(element("li", clean(item)));
      }
      fragment.append(list);
    } else {
      const paragraph = [line];
      i++;
      while (i < lines.length && lines[i].trim() && !/^(#|\||- )/.test(lines[i].trim())) {
        paragraph.push(lines[i++].trim());
      }
      fragment.append(element("p", clean(paragraph.join(" "))));
    }
  }
  return fragment;
}

function renderTable(rows) {
  const wrapper = element("div", "");
  wrapper.className = "table-wrap";
  const table = document.createElement("table");
  rows.forEach((row, index) => {
    const group = document.createElement(index === 0 ? "thead" : "tbody");
    const tr = document.createElement("tr");
    row.forEach(value => {
      const cell = element(index === 0 ? "th" : "td", value);
      if (index === 0) cell.scope = "col";
      tr.append(cell);
    });
    group.append(tr);
    table.append(group);
  });
  wrapper.append(table);
  return wrapper;
}

async function populate(node, attribute, render) {
  try {
    const text = await readFile(node.dataset[attribute]);
    node.replaceChildren(render(text));
  } catch (error) {
    // Keep any existing direct download link usable on a loading failure.
    node.append(element("p", error.message));
  }
}

document.querySelectorAll("[data-notes]").forEach(node => populate(node, "notes", renderNotes));
document.querySelectorAll("[data-csv]").forEach(node => populate(node, "csv", text =>
  renderTable(text.trim().split(/\r?\n/).map(line => line.split(",")))));
document.querySelectorAll("[data-source]").forEach(node => populate(node, "source", text =>
  document.createTextNode(text)));
