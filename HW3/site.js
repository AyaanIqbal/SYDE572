// Load the original notes, result tables, and runnable source files.
async function populate(node, path, render) {
  try {
    const response = await fetch(path, {cache: 'no-store'});
    if (!response.ok) throw new Error(`Unable to load ${path}`);
    node.replaceChildren(render(await response.text()));
  } catch (error) {
    const message = document.createElement('p');
    message.textContent = error.message;
    node.append(message);
  }
}
function textNode(tag, text) {
  const node = document.createElement(tag);
  node.textContent = text;
  return node;
}
document.querySelectorAll('[data-notes]').forEach(node => {
  populate(node, node.dataset.notes, text => {
    const fragment = document.createDocumentFragment();
    text.trim().split(/\n\s*\n/).forEach(paragraph => {
      fragment.append(textNode('p', paragraph.replace(/\n/g, ' ')));
    });
    return fragment;
  });
});
document.querySelectorAll('[data-source]').forEach(node => {
  populate(node, node.dataset.source, text => document.createTextNode(text));
});
document.querySelectorAll('[data-csv]').forEach(node => {
  populate(node, node.dataset.csv, text => {
    const wrapper = document.createElement('div');
    wrapper.className = 'table-wrap';
    const table = document.createElement('table');
    const head = document.createElement('thead');
    const body = document.createElement('tbody');
    text.trim().split(/\r?\n/).forEach((line, index) => {
      const row = document.createElement('tr');
      if (node.dataset.csv.endsWith('pca_axes.csv')) {
        const label = textNode('th', index === 0 ? 'Axis' : `PC${index}`);
        label.scope = index === 0 ? 'col' : 'row';
        row.append(label);
      }
      line.split(',').forEach(value => {
        const number = Number(value);
        const displayed = index > 0 && value.trim() && Number.isFinite(number)
          ? (Number.isInteger(number) ? String(number) : number.toFixed(6)) : value;
        const cell = textNode(index === 0 ? 'th' : 'td', displayed);
        if (index === 0) cell.scope = 'col';
        row.append(cell);
      });
      (index === 0 ? head : body).append(row);
    });
    table.append(head, body);
    wrapper.append(table);
    return wrapper;
  });
});
