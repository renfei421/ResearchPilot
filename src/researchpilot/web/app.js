"use strict";
const el = id => document.getElementById(id);
const stages = {queued:"等待执行", planning:"规划检索", searching:"检索学术文献", reranking:"评估文献相关性",
  acquiring:"获取文档", retrieving:"检索证据段落", selecting:"选择证据", synthesizing:"整理结论",
  verifying:"核验结论", assessing:"评估证据缺口", rescuing:"补救已下载文档", follow_up:"规划补充检索", finalizing:"整理最终结果", completed:"已完成", failed:"运行失败"};
const statuses = {queued:"排队中", running:"研究中", completed:"已完成", failed:"失败"};
let activeRun = null, generation = 0, pollTimer = null, historyLimit = 20;

async function request(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail = typeof body.detail === "string" ? body.detail : "请求失败，请重试。";
    throw new Error(`${detail}${body.issues ? " (" + body.issues.map(item => item.field).join(", ") + ")" : ""}`);
  }
  return response;
}
function error(message = "") { el("error").textContent = message; el("error").hidden = !message; }
function clearSelection() {
  generation++; clearTimeout(pollTimer); activeRun = null;
  el("run-panel").hidden = true; el("result").replaceChildren(); el("exports").hidden = true;
  error(); history.replaceState(null, "", location.pathname);
}
async function historyList() {
  const items = await (await request(`/research?limit=${historyLimit}`)).json();
  el("history").replaceChildren();
  if (!items.length) { const p = document.createElement("p"); p.className="muted"; p.textContent="还没有研究记录"; el("history").append(p); }
  for (const item of items) {
    const button = document.createElement("button"); button.type="button";
    button.className = "history-item" + (item.run_id === activeRun ? " active" : "");
    const title = document.createElement("strong"); title.textContent = item.question;
    const meta = document.createElement("small"); meta.textContent = `${item.mode === "claim_check" ? "Idea Check" : "Research"} · ${statuses[item.status]} · ${new Date(item.created_at).toLocaleString()}`;
    button.append(title, meta); button.addEventListener("click", () => openRun(item.run_id)); el("history").append(button);
  }
  el("more-history").hidden = items.length < historyLimit || historyLimit >= 100;
}
async function poll(runId, token) {
  try {
    const run = await (await request(`/research/${encodeURIComponent(runId)}`)).json();
    if (token !== generation) return;
    error(); el("run-panel").hidden=false; el("run-question").textContent=run.question;
    el("run-status").textContent=statuses[run.status]; el("stage").textContent=stages[run.current_stage] || run.current_stage;
    el("round").textContent=run.current_round; el("papers").textContent=run.selected_papers; el("evidence").textContent=run.evidence_collected;
    el("run-meta").textContent=`运行 ${run.run_id.slice(0, 8)} · 更新于 ${new Date(run.updated_at).toLocaleTimeString()}`;
    el("warnings").replaceChildren(); el("warnings").hidden=!run.warnings.length;
    for (const message of run.warnings) { const li=document.createElement("li"); li.textContent=message; el("warnings").append(li); }
    if (run.status === "completed") {
      const html = await (await request(`/research/${encodeURIComponent(runId)}/view`)).text();
      if (token !== generation) return;
      // This endpoint emits escaped HTML; source/model strings never become markup.
      el("result").innerHTML=html; el("exports").hidden=false;
      el("export-json").href=`/research/${encodeURIComponent(runId)}/export/json`;
      el("export-markdown").href=`/research/${encodeURIComponent(runId)}/export/markdown`;
      await historyList();
      if (window.refreshProject) await window.refreshProject(run.project_id);
    } else if (run.status === "failed") { error(run.error || "研究未完成，请重试。"); await historyList(); }
    else { pollTimer=setTimeout(() => poll(runId, token), 1200); }
  } catch (err) {
    if (token !== generation) return;
    error(`无法读取运行状态：${err.message} 正在重试…`);
    pollTimer=setTimeout(() => poll(runId, token), 5000);
  }
}
function openRun(runId) {
  clearSelection(); activeRun=runId; history.replaceState(null,"",`?run=${encodeURIComponent(runId)}`);
  poll(runId, generation); historyList().catch(err => error(err.message));
}
el("research-form").addEventListener("submit", async event => {
  event.preventDefault(); error(); el("submit").disabled=true;
  const number = id => el(id).value === "" ? null : Number(el(id).value);
  try {
    const idea = el("mode").value === "claim_check";
    const body = {year_from:number("year-from"), year_to:number("year-to"), max_search_rounds:number("max-rounds")};
    if (el("run-project").value) Object.assign(body, {project_id:el("run-project").value, refresh_search:el("refresh-search").checked});
    if (idea) Object.assign(body, {claim:el("question").value, field:el("claim-field").value || null,
      context:el("claim-context").value || null, max_papers_per_claim:number("max-papers")});
    else Object.assign(body, {question:el("question").value, max_papers:number("max-papers")});
    const response = await request(idea ? "/claim-check" : "/research", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
    openRun((await response.json()).run_id);
  } catch (err) { error(err.message); }
  finally { el("submit").disabled=false; }
});
el("mode").addEventListener("change", () => {
  const idea = el("mode").value === "claim_check";
  el("idea-options").hidden = !idea;
  el("question-label").textContent = idea ? "你的研究主张、假设或方法构想" : "你想研究什么？";
  el("question").placeholder = idea ? "描述核心想法，以及适用条件、方法和预期结论。" : "输入你的研究问题…";
  el("submit").textContent = idea ? "Check Prior Work" : "开始研究 ↗";
  el("paper-budget-label").textContent = idea ? "每项主张最多文献" : "每轮最多文献";
  el("max-papers").max = idea ? "8" : "20"; el("max-papers").value = idea ? "8" : "6";
  el("max-rounds").max = idea ? "2" : "3"; el("max-rounds").value = "2";
  el("budget-help").textContent = idea ? "最多两轮，整个运行最多获取 16 篇文献。重复查询与文档共享复用。" : "有证据缺口时才继续下一轮。总文献上限为每轮文献数 × 轮次数。";
});
el("new-research").addEventListener("click", () => { clearSelection(); el("question").value=""; el("question").focus(); historyList().catch(err=>error(err.message)); });
el("refresh-history").addEventListener("click", () => historyList().catch(err=>error(err.message)));
el("more-history").addEventListener("click", () => { historyLimit=Math.min(100,historyLimit+20); historyList().catch(err=>error(err.message)); });
el("result").addEventListener("click", event => {
  const link=event.target.closest("a.citation"); if (!link) return;
  const target=document.getElementById(link.getAttribute("href").slice(1));
  if (target) { event.preventDefault(); target.open=true; target.scrollIntoView({behavior:"smooth",block:"center"}); target.querySelector("summary").focus(); }
});
async function init() {
  try {
    const health=await (await request("/health")).json();
    if (!health.openai_configured) { el("configuration").textContent="当前进程未配置 OPENAI_API_KEY。请在启动终端设置环境变量并重启服务器后进行真实研究。"; el("configuration").hidden=false; }
    await historyList(); const id=new URLSearchParams(location.search).get("run"); if (id) openRun(id);
  } catch (err) { error(`无法连接本地服务：${err.message}`); }
}
init();
