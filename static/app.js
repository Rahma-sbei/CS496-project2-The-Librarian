// The Librarian chat page: sends questions to /api/ask and renders the cited answer and its sources.
const form = document.getElementById("ask");
const box = document.getElementById("question");
const send = document.getElementById("send");
const thread = document.getElementById("thread");
const empty = document.getElementById("empty");
const rewrite = document.getElementById("rewrite");
const types = document.getElementById("types");
let turns = 0;

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

// Minimal Markdown for LLM answers: code fences, inline code, bold, bullet lists, links, [n] citations.
function inline(text, turn, nSources) {
  const codes = [];
  let s = esc(text).replace(/`([^`]+)`/g, (_, c) => { codes.push(c); return `\u0000${codes.length - 1}\u0000`; });
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
       .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
       .replace(/\[(\d+)\]/g, (_, n) => +n >= 1 && +n <= nSources
         ? `<a class="cite" href="#src-${turn}-${n}" data-n="${n}">${n}</a>`
         : `<span class="cite bad" title="This source number does not exist">${n}</span>`);
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${codes[i]}</code>`);
}

function markdown(text, turn, nSources) {
  const out = [];
  const parts = text.split(/```[\w-]*\n?([\s\S]*?)```/g);   // odd indexes are code blocks
  parts.forEach((part, i) => {
    if (i % 2) { out.push(`<pre><code>${esc(part.replace(/\n$/, ""))}</code></pre>`); return; }
    for (const block of part.split(/\n{2,}/)) {
      const lines = block.split("\n").filter((l) => l.trim());
      if (!lines.length) continue;
      if (lines.every((l) => /^\s*([-*]|\d+\.)\s+/.test(l))) {
        out.push("<ul>" + lines.map((l) => `<li>${inline(l.replace(/^\s*([-*]|\d+\.)\s+/, ""), turn, nSources)}</li>`).join("") + "</ul>");
      } else {
        out.push(`<p>${lines.map((l) => inline(l, turn, nSources)).join("<br>")}</p>`);
      }
    }
  });
  return out.join("");
}

function sourceCard(s, turn) {
  const where = s.section ? ` <span class="src-section">› ${esc(s.section.replace(/\*\*/g, ""))}</span>` : "";
  const date = s.date ? ` · ${esc(s.date.slice(0, 10))}` : "";
  return `<div class="src ${s.cited ? "cited" : "uncited"}" id="src-${turn}-${s.n}">
    <div class="src-head">
      <span class="src-n">[${s.n}]</span>
      <span class="badge ${esc(s.type)}">${esc(s.type)}</span>
      <span class="src-title">${esc(s.title)}${where}</span>
      <a class="src-link" href="${esc(s.url)}" target="_blank" rel="noopener">open ↗</a>
    </div>
    <details><summary>${s.cited ? "Cited" : "Retrieved, not cited"} · similarity ${s.score}${date} · show passage</summary><pre>${esc(s.text)}</pre></details>
  </div>`;
}

function renderAnswer(el, data, turn) {
  const idk = /^i don['’]t know\.?$/i.test(data.answer.trim());
  const body = idk
    ? `<div class="idk">I don't know. The retrieved passages don't answer this question.</div>`
    : `<div class="answer">${markdown(data.answer, turn, data.sources.length)}</div>`;
  const sources = data.sources.length
    ? `<div class="sources"><h3>Sources</h3>${data.sources.map((s) => sourceCard(s, turn)).join("")}</div>` : "";
  const rewritten = data.query !== undefined && data.query.trim() !== data.question
    ? `<span>searched for: “${esc(data.query)}”</span>` : "";
  el.innerHTML = `${body}${sources}
    <div class="meta">
      <span>${data.latency_s.toFixed(1)} s</span><span>${data.tokens} tokens</span>
      <span>$${data.cost_usd.toFixed(5)}</span><span>${esc(data.model)}</span>${data.cached ? "<span>cached answer</span>" : ""}${rewritten}
      <span class="feedback">Helpful?
        <button type="button" data-helpful="true" aria-label="Helpful">👍</button>
        <button type="button" data-helpful="false" aria-label="Not helpful">👎</button>
      </span>
    </div>`;

  el.querySelectorAll(".cite[data-n]").forEach((a) => a.addEventListener("click", (ev) => {
    ev.preventDefault();
    const card = document.getElementById(`src-${turn}-${a.dataset.n}`);
    card.scrollIntoView({ behavior: "smooth", block: "center" });
    card.classList.add("flash");
    setTimeout(() => card.classList.remove("flash"), 1200);
  }));
  el.querySelectorAll(".feedback button").forEach((b) => b.addEventListener("click", async () => {
    const buttons = el.querySelectorAll(".feedback button");
    buttons.forEach((x) => (x.disabled = true));
    b.classList.add("chosen");
    const helpful = b.dataset.helpful === "true";
    const comment = helpful ? "" : (prompt("What was wrong with this answer? (optional)") || "");
    try {
      await fetch("/api/feedback", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ request_id: data.request_id, helpful, comment: comment.slice(0, 1000) }) });
      b.closest(".feedback").firstChild.textContent = "Thanks! ";
    } catch { b.closest(".feedback").firstChild.textContent = "Could not send feedback "; }
  }));
}

async function ask(question) {
  empty.hidden = true;
  const turn = ++turns;
  const node = document.getElementById("turn-tpl").content.firstElementChild.cloneNode(true);
  node.querySelector(".q").textContent = question;
  const answerEl = node.querySelector(".a");
  thread.appendChild(node);
  node.scrollIntoView({ behavior: "smooth", block: "start" });

  send.disabled = true;
  try {
    const r = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, rewrite: rewrite.checked, types: types.value ? [types.value] : null }) });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(typeof data.detail === "string" ? data.detail : `Request failed (${r.status})`);
    data.question = question;
    renderAnswer(answerEl, data, turn);
  } catch (err) {
    answerEl.innerHTML = `<div class="error">${esc(err.message || "Could not reach the server.")}</div>`;
  } finally {
    send.disabled = false;
    box.focus();
  }
}

form.addEventListener("submit", (ev) => {
  ev.preventDefault();
  const q = box.value.trim();
  if (!q || send.disabled) return;
  box.value = "";
  box.style.height = "auto";
  ask(q);
});
box.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); form.requestSubmit(); }
});
box.addEventListener("input", () => { box.style.height = "auto"; box.style.height = `${box.scrollHeight}px`; });
document.querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => ask(c.textContent)));
