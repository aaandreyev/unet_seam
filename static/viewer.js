async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

async function loadSamples() {
  const split = document.getElementById("filter-split").value;
  const tag = document.getElementById("filter-tag").value;
  const orientation = document.getElementById("filter-orientation").value;
  const op = document.getElementById("filter-op").value;
  const params = new URLSearchParams();
  if (split) params.set("split", split);
  if (tag) params.set("tag", tag);
  if (orientation) params.set("orientation", orientation);
  if (op) params.set("op", op);
  const rows = await fetchJson(`/api/samples?${params.toString()}`);
  const list = document.getElementById("sample-list");
  list.innerHTML = "";
  rows.forEach((row) => {
    const item = document.createElement("button");
    item.className = "sample";
    item.textContent = `${row.id} | ${row.split} | ${row.strip.axis}`;
    item.onclick = () => inspect(row.id);
    list.appendChild(item);
  });
}

async function inspect(id) {
  const meta = await fetchJson(`/api/sample/${id}`);
  document.getElementById("input-img").src = `/api/strip/${id}/input.png`;
  document.getElementById("target-img").src = `/api/strip/${id}/target.png`;
  document.getElementById("meta").textContent = JSON.stringify(meta, null, 2);
}

async function loadStats() {
  document.getElementById("stats-pre").textContent = JSON.stringify(await fetchJson("/api/stats"), null, 2);
}

async function loadRuns() {
  document.getElementById("runs-pre").textContent = JSON.stringify(await fetchJson("/api/runs"), null, 2);
}

function setupTabs() {
  document.querySelectorAll(".tabs button").forEach((button) => {
    button.onclick = () => {
      document.querySelectorAll(".tab").forEach((tab) => tab.classList.remove("on"));
      document.getElementById(button.dataset.tab).classList.add("on");
    };
  });
}

document.getElementById("reload").onclick = async () => {
  await loadSamples();
  await loadStats();
  await loadRuns();
};

setupTabs();
loadSamples();
loadStats();
loadRuns();
