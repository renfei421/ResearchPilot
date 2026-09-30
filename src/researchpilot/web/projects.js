"use strict";
let currentProject = null, projectData = null, continueTarget = null, projectGeneration = 0;
const projectPath = () => `/projects/${encodeURIComponent(currentProject)}`;
const jsonOptions = (method, body) => ({method, headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
async function projectList() {
  const projects = await (await request("/projects")).json();
  const selected = el("run-project").value;
  el("project-list").replaceChildren();
  el("run-project").replaceChildren(new Option("独立研究（不加入项目）", ""));
  for (const p of projects) {
    const button=document.createElement("button"); button.type="button"; button.className="history-item";
    button.textContent=p.title + (p.status === "archived" ? " · 已归档" : "");
    button.addEventListener("click", () => openProject(p.project_id).catch(err=>error(err.message)));
    el("project-list").append(button);
    if (p.status === "active") el("run-project").append(new Option(p.title, p.project_id));
  }
  el("run-project").value=selected;
}
async function openProject(id) {
  const token=++projectGeneration;
  const data=await (await request(`/projects/${encodeURIComponent(id)}`)).json();
  const html=await (await request(`/projects/${encodeURIComponent(id)}/view`)).text();
  if (token !== projectGeneration) return;
  currentProject=id; projectData=data;
  // Server template escapes all user, model and source text.
  el("project-detail").innerHTML=html; el("project-panel").hidden=false;
  const active=data.project.status === "active";
  el("run-project").value=active ? id : "";
  el("project-record-form").hidden=!active;
  el("project-ask").disabled=!active; el("project-idea").disabled=!active;
  el("project-archive").textContent=active ? "归档项目" : "恢复项目";
  el("project-export-json").href=projectPath()+"/export/json";
  el("project-export-md").href=projectPath()+"/export/markdown";
  el("project-edit-form").hidden=true; el("project-continue-form").hidden=true;
  el("project-edit-title").value=data.project.title;
  el("project-edit-description").value=data.project.description;
  el("project-edit-field").value=data.project.field || "";
}
window.refreshProject = async id => {
  await projectList();
  if (currentProject && (!id || id === currentProject)) await openProject(currentProject);
};
el("new-project").addEventListener("click", () => {el("project-create").hidden=!el("project-create").hidden; el("project-title").focus();});
el("project-create").addEventListener("submit", async event => {
  event.preventDefault(); const button=event.submitter; button.disabled=true;
  try {
    const p=await (await request("/projects", jsonOptions("POST", {title:el("project-title").value,
      description:el("project-description").value, field:el("project-field").value || null}))).json();
    el("project-create").reset(); el("project-create").hidden=true;
    await projectList(); await openProject(p.project_id); error();
  } catch(err) {error(err.message);} finally {button.disabled=false;}
});
el("project-edit").addEventListener("click", () => {el("project-edit-form").hidden=!el("project-edit-form").hidden;});
el("project-edit-form").addEventListener("submit", async event => {
  event.preventDefault();
  try {
    await request(projectPath(), jsonOptions("PATCH", {title:el("project-edit-title").value,
      description:el("project-edit-description").value, field:el("project-edit-field").value || null}));
    await window.refreshProject();
  } catch(err) {error(err.message);}
});
el("project-archive").addEventListener("click", async () => {
  try {await request(projectPath(), jsonOptions("PATCH", {status:projectData.project.status === "active" ? "archived" : "active"})); await window.refreshProject();}
  catch(err) {error(err.message);}
});
el("project-record-form").addEventListener("submit", async event => {
  event.preventDefault(); const button=event.submitter; button.disabled=true;
  try {await request(projectPath()+"/"+el("project-record-kind").value,
    jsonOptions("POST", {text:el("project-record-text").value})); el("project-record-text").value=""; await openProject(currentProject);}
  catch(err) {error(err.message);} finally {button.disabled=false;}
});
function projectCompose(mode) {
  clearSelection(); el("run-project").value=currentProject; el("mode").value=mode;
  el("mode").dispatchEvent(new Event("change")); el("question").value="";
  if (mode === "claim_check") el("claim-field").value=projectData.project.field || "";
  el("research-form").scrollIntoView({behavior:"smooth"}); el("question").focus();
}
el("project-ask").addEventListener("click", () => projectCompose("research"));
el("project-idea").addEventListener("click", () => projectCompose("claim_check"));
el("run-project").addEventListener("change", () => {
  if (el("run-project").value) openProject(el("run-project").value).catch(err=>error(err.message));
  else {currentProject=null; projectGeneration++; el("project-panel").hidden=true;}
});
el("project-detail").addEventListener("click", async event => {
  const button=event.target.closest("button");
  const link=event.target.closest("a[data-evidence-link]");
  if (link) {const target=el(link.getAttribute("href").slice(1)); if (target) target.open=true;}
  if (!button) return;
  try {
    if (button.dataset.run) {openRun(button.dataset.run); el("run-panel").scrollIntoView({behavior:"smooth"});}
    if (button.dataset.archive) {
      await request(projectPath()+`/${button.dataset.archive}/${encodeURIComponent(button.dataset.id)}`, jsonOptions("PATCH", {status:"archived"}));
      await openProject(currentProject);
    }
    if (button.dataset.continue) {
      continueTarget={target_type:button.dataset.continue, target_id:button.dataset.id, mode:button.dataset.mode};
      el("project-continue-label").textContent=button.closest("article").querySelector("p").textContent;
      el("project-continue-instruction").value=""; el("project-continue-refresh").checked=false;
      el("project-continue-form").hidden=false; el("project-continue-form").scrollIntoView({behavior:"smooth"});
    }
  } catch(err) {error(err.message);}
});
el("project-continue-form").addEventListener("submit", async event => {
  event.preventDefault(); const button=event.submitter; button.disabled=true;
  try {
    const run=await (await request(projectPath()+"/continue", jsonOptions("POST", {...continueTarget,
      instruction:el("project-continue-instruction").value, refresh_search:el("project-continue-refresh").checked}))).json();
    el("project-continue-form").hidden=true; openRun(run.run_id);
  } catch(err) {error(err.message);} finally {button.disabled=false;}
});
projectList().catch(err=>error(err.message));
